import math
import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.modular_attack import ModularEvasionAttackFixedEps
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from raid.attacks.loss.avg_loss import AvgEnsembleLoss

MOMENTUM_ALPHA = 0.75
RHO = 0.75


def _check_oscillation(loss_steps, j, k, k3=0.75):
    """Verbatim from Croce & Hein 2020 reference implementation."""
    t = torch.zeros_like(loss_steps[0])
    for counter5 in range(k):
        t.add_(loss_steps[j - counter5] > loss_steps[j - counter5 - 1])
    return t <= k * k3


class EnsembleAPGD(ModularEvasionAttackFixedEps):
    """Auto-PGD (Croce & Hein, 2020) adapted for ensemble attacks.

    The optimization loop follows the official _apgd implementation
    (fra31/auto-attack) exactly: one forward pass per iteration (the
    gradient pass doubles as the loss evaluation), counter-based checkpoint
    schedule, and check_oscillation for step-size halving.

    The only adaptation is the ensemble interface: forward_loss /
    manipulation_function / gradient_processing from secmlt's
    ModularEvasionAttackFixedEps replace direct model() / clamp / sign calls.
    """

    def __init__(
            self,
            perturbation_model: str,
            epsilon: float,
            num_steps: int,
            step_size: float,
            random_start: bool,
            loss_function: torch.nn.Module = AvgEnsembleLoss(),
            y_target: int | None = None,
            lb: float = 0.0,
            ub: float = 1.0,
            trackers: list[Tracker] | None = None,
            **kwargs,
    ) -> None:
        perturbation_models = {
            LpPerturbationModels.L1: L1Constraint,
            LpPerturbationModels.L2: L2Constraint,
            LpPerturbationModels.LINF: LInfConstraint,
        }

        if random_start:
            initializer = RandomLpInitializer(
                perturbation_model=perturbation_model,
                radius=epsilon,
            )
        else:
            initializer = Initializer()
        self.epsilon = epsilon
        gradient_processing = LinearProjectionGradientProcessing(perturbation_model)
        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=self.epsilon),
        ]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )
        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=loss_function,
            optimizer_cls=OptimizerFactory.create_sgd(step_size),
            manipulation_function=manipulation_function,
            gradient_processing=gradient_processing,
            initializer=initializer,
            trackers=trackers,
        )

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
        optim_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        if init_deltas is not None:
            delta = init_deltas.data.clone()
        elif isinstance(self.initializer, BaseEvasionAttack):
            _, delta = self.initializer._run(model, samples, target)
        else:
            delta = self.initializer(samples.data)

        batch = samples.shape[0]
        x_adv, delta = self.manipulation_function(samples.data, delta.data)

        # ── initial forward pass (official _apgd lines 412-425) ──
        x_adv.requires_grad_(True)
        scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
        loss_indiv = losses * multiplier
        grad = torch.autograd.grad(loss_indiv.sum(), x_adv)[0]
        x_adv.detach_()
        loss_indiv.detach_()

        x_best = x_adv.clone()
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()
        x_adv_old = x_adv.clone()

        # ── checkpoint schedule (official _apgd line 390, 430-431) ──
        n_iter_2 = max(int(0.22 * self.num_steps), 1)
        n_iter_min = max(int(0.06 * self.num_steps), 1)
        size_decr = max(int(0.03 * self.num_steps), 1)

        step_size = torch.full((batch,), float(self.step_size))
        k = n_iter_2
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.zeros(batch, dtype=torch.bool)
        loss_steps = torch.zeros(self.num_steps, batch)

        for i in range(self.num_steps):
            # ── gradient step (official lines 449-490, Linf case) ──
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv

            step = atleast_kd(step_size, x_adv.dim())
            grad_dir = self.gradient_processing(grad)

            z_next, _ = self.manipulation_function(
                samples.data, (x_adv - samples.data) - step * grad_dir
            )

            if i == 0:
                x_adv_1 = z_next
            else:
                combined = (
                    x_adv
                    + MOMENTUM_ALPHA * (z_next - x_adv)
                    + (1 - MOMENTUM_ALPHA) * grad2
                )
                x_adv_1, _ = self.manipulation_function(
                    samples.data, combined - samples.data
                )

            x_adv = x_adv_1

            # ── get gradient (official lines 492-501) ──
            x_adv.requires_grad_(True)
            scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            loss_indiv = losses * multiplier
            grad = torch.autograd.grad(loss_indiv.sum(), x_adv)[0]
            x_adv.detach_()
            loss_indiv.detach_()

            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(
                        i, losses.detach().cpu(), scores.detach().cpu(),
                        x_adv.detach().cpu(), (x_adv - samples.data).detach().cpu(),
                        grad.detach().cpu(),
                    )

            # ── update best + check step size (official lines 508-542) ──
            loss_steps[i] = -loss_indiv
            ind = loss_indiv < loss_best
            x_best[ind] = x_adv[ind]
            grad_best[ind] = grad[ind]
            loss_best[ind] = loss_indiv[ind]

            counter3 += 1

            if counter3 == k:
                fl_reduce_no_impr = (
                    (~reduced_last_check)
                    & ((-loss_best_last_check) >= (-loss_best))
                )
                reduced_last_check = (
                    _check_oscillation(loss_steps, i, k, k3=RHO)
                    | fl_reduce_no_impr
                )
                loss_best_last_check = loss_best.clone()

                if reduced_last_check.any():
                    step_size[reduced_last_check] /= 2.0
                    x_adv = x_adv.clone()
                    x_adv[reduced_last_check] = x_best[reduced_last_check]
                    grad[reduced_last_check] = grad_best[reduced_last_check]

                k = max(k - size_decr, n_iter_min)
                counter3 = 0

        best_x = x_best.cpu()
        return best_x, best_x - samples.data.cpu()
