import torch
import torch.nn.functional as F
from secmlt.adv.evasion.modular_attack import ModularEvasionAttackFixedEps
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import ClipConstraint, LInfConstraint
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from raid.attacks.loss.avg_loss import AvgEnsembleLoss


class EnsembleCWA(ModularEvasionAttackFixedEps):
    """Common Weakness Attack, MI-CWA (Chen et al., "Rethinking Model
    Ensemble in Transfer-based Adversarial Attacks", ICLR 2024).

    Follows MI_CommonWeakness.attack from the official implementation
    (huanranchen/AdversarialAttacks, attacks/AdversarialInput/CommonWeakness.py),
    Linf, untargeted, no DI/TI. Each outer step:
      1. SAM reverse step: x0 = x - r * sign(grad CE(sum_i f_i(x), y))
      2. CSE inner loop, one model at a time:
         m = mu * m + g_i / ||g_i||_2 ;  x_i = x_{i-1} + beta * m
      3. outer MI step on the pseudo-gradient G = x_n - x:
         M = mu * M + G / ||G||_1 ;  x = x + alpha * sign(M)
    with [lb, ub] clipping and projection onto the eps-ball after every
    update, as in the official code. The inner momentum persists across
    outer steps (it is not reset), also as in the official code.

    Deviations from the official code:
      - outer step alpha = `step_size` (official hard-codes 16/255/5), so it
        runs at the same epsilon/num_steps/step_size as EnsemblePGD/APGD;
      - reverse step defaults to epsilon/15 (official: 16/255/15, i.e.
        epsilon/15 at its default epsilon=16/255);
      - the outer L1 normalization is per sample (official normalizes over
        the whole batch, which couples samples in a batch; per sample is
        the MI-FGSM formulation and makes results batch-independent);
      - the ensemble model is RAID's EnsembleModel: per-model logits come
        from its members (with the ensemble's logit scaling, if any).
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
            reverse_step_size: float | None = None,
            inner_step_size: float = 250.0,
            mu: float = 1.0,
            **kwargs,
    ) -> None:
        """
        Create the ensemble Common Weakness attack.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model for the attack. Only inf is supported.
        epsilon : float
            Radius of the constraint for the Lp ball.
        num_steps : int
            Number of outer iterations.
        step_size : float
            Outer (momentum sign) step size.
        random_start : bool
            Whether to use a random initialization onto the Lp ball.
        loss_function : torch.nn.Module, optional
            Only used for tracking; the attack itself uses cross-entropy, as
            the official implementation.
        y_target : int | None, optional
            Target label for a targeted attack, None
            for untargeted attack, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers).
        reverse_step_size : float | None, optional
            SAM reverse step size, by default epsilon / 15.
        inner_step_size : float, optional
            CSE inner step size on L2-normalized gradients, by default 250.
        mu : float, optional
            Momentum for both the inner and outer updates, by default 1.
        """
        if perturbation_model != LpPerturbationModels.LINF:
            raise ValueError("EnsembleCWA only supports the Linf perturbation model.")

        if random_start:
            initializer = RandomLpInitializer(
                perturbation_model=perturbation_model,
                radius=epsilon,
            )
        else:
            initializer = Initializer()
        self.epsilon = epsilon
        self.reverse_step_size = (
            epsilon / 15 if reverse_step_size is None else reverse_step_size
        )
        self.inner_step_size = inner_step_size
        self.mu = mu
        gradient_processing = LinearProjectionGradientProcessing(perturbation_model)
        manipulation_function = AdditiveManipulation(
            domain_constraints=[ClipConstraint(lb=lb, ub=ub)],
            perturbation_constraints=[LInfConstraint(radius=self.epsilon)],
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

    def _project(self, samples: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Clip x to [lb, ub] and project it onto the eps-ball around samples."""
        x_adv, _ = self.manipulation_function(samples.data, x.data - samples.data)
        return x_adv.detach()

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
        optim_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # +1 ascends the CE of the true label (untargeted), -1 descends the
        # CE of the target label (targeted); the SAM step goes the other way.
        direction = 1 if self.y_target is None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        if init_deltas is not None:
            delta = init_deltas.data.clone()
        else:
            delta = self.initializer(samples.data)
        x = self._project(samples, samples.data + delta)

        members = model.models
        scaling = model.ensemble_function._apply_scaling
        inner_momentum = torch.zeros_like(x)
        outer_momentum = torch.zeros_like(x)

        for i in range(self.num_steps):
            x_outer = x.clone()

            # ── 1. SAM reverse step on the logit-summed ensemble ──
            x.requires_grad_(True)
            scores = model.decision_function(x)  # (n_models, batch, classes)
            loss = F.cross_entropy(scores.sum(dim=0), target)
            grad = torch.autograd.grad(loss, x)[0]
            x = self._project(samples, x.detach() - direction * self.reverse_step_size * grad.sign())

            if self.trackers is not None:
                with torch.no_grad():
                    losses = self.loss_function(scores.detach(), target)
                for tracker in self.trackers:
                    tracker.track(
                        i, losses.cpu(), scores.detach().cpu(),
                        x_outer.cpu(), (x_outer - samples.data).cpu(),
                        grad.detach().cpu(),
                    )

            # ── 2. CSE inner loop, one ensemble member at a time ──
            for name, member in members.items():
                x.requires_grad_(True)
                logits = scaling(member(x.to(member._get_device())), name)
                loss = F.cross_entropy(logits, target.to(logits.device))
                grad = torch.autograd.grad(loss, x)[0]
                grad_norm = grad.flatten(1).norm(p=2, dim=1)
                inner_momentum = self.mu * inner_momentum + direction * grad / atleast_kd(grad_norm, grad.dim())
                x = self._project(samples, x.detach() + self.inner_step_size * inner_momentum)

            # ── 3. outer MI step on the pseudo-gradient ──
            fake_grad = x - x_outer
            fake_norm = fake_grad.flatten(1).norm(p=1, dim=1).clamp_min(1e-12)
            outer_momentum = self.mu * outer_momentum + fake_grad / atleast_kd(fake_norm, fake_grad.dim())
            x = self._project(samples, x_outer + self.step_size * outer_momentum.sign())

        x = x.cpu()
        return x, x - samples.data.cpu()
