import argparse
import pprint
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from attacks import EnsemblePGD, EnsembleAPGD, EnsembleCWA
from data import get_dataloader
from secmlt.adv.backends import Backends
from secmlt.adv.evasion import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy
from secmlt.trackers import (
    LossTracker, PredictionTracker, PerturbationNormTracker, 
    SampleTracker, ScoresTracker, TensorboardTracker, GradientNormTracker
)
from attacks.trackers.ensemble_trackers import EnsembleScoresTracker, EnsemblePredictionTracker, AvgEnsembleScoresTracker, MajorityVotingPredictionTracker
from raid.plots import plot_adv_example
import os
import time
import pickle

from constants import MODELS, ENSEMBLING_STRATEGIES, ENSEMBLE_LOSSES
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import random


THRESHOLD = 0.5
ATTACK_CLASSES = {"pgd": EnsemblePGD, "apgd": EnsembleAPGD, "cwa": EnsembleCWA}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    pprint.pp(vars(args))
    if args.random_start:
        set_seed(args.random_seed)

    device_str = args.device[0] if isinstance(args.device, list) else args.device
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.dry_run:
        # pgd (default) keeps the original "adv_raw_..." naming for backward
        # compatibility with already-generated PGD results; apgd/cwa embed
        # the attack name so they don't collide with PGD's directories.
        name_prefix = args.ensembling_strategy if args.attack == "pgd" else args.attack
        output_dir = Path(
            f"{args.output_dir}/ADV/"
            f"{args.path_to_dataset.split('/')[-1]}{'_'+str(args.subset) if args.subset != -1 else ''}/"
            f"adv_{name_prefix}_{args.models}_{args.epsilon.replace('/','_')}"
            f"_{args.num_steps}_{args.step_size}"
        )
        # Directory existing isn't enough to call a cell "done" — a prior run
        # can leave behind partial output (e.g. the example_*.pdf plots) and
        # die before writing adv_dataset.pkl, which used to make this skip
        # silently forever. When --generate is set, require the pkl itself.
        already_done = output_dir.exists() and (
            not args.generate or (output_dir / "adv_dataset.pkl").exists()
        )
        if (
            already_done
            # and input(f"Output directory: {output_dir} exists, continue and overwrite? (y/n): ") != "y"
            and not args.overwrite
        ):
            return
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # create model evaluation results output directory
        if args.eval_model:
            eval_output_dir = Path(
                f"{args.eval_output_dir}/"
                f"{Path(*Path(output_dir).parts[1:])}/"
                f"{args.eval_model}"
            )
            if (
                eval_output_dir.exists()
                # and input(f"Output directory: {eval_output_dir} exists, continue and overwrite? (y/n): ") != "y"
                and not args.overwrite
            ):
                return
            eval_output_dir.mkdir(exist_ok=True, parents=True)

    #########################################################################
    #                     Loading the dataset and models                    #
    #########################################################################

    if args.scale_logits:
        logit_scaler_file = 'logit_scaler/logits_scalers.pkl'
        with open(logit_scaler_file, "rb") as f:
            scalers = pickle.load(f)
    else:
        scalers = None

    if len(args.models) > 1:
        if isinstance(args.device, list):
            assert len(args.device) == len(args.models), "The number of devices must be equal to the number of models"
            devices = [device if torch.cuda.is_available() else "cpu"
                       for device in args.device]
        else:
            devices = [args.device if torch.cuda.is_available() else "cpu"
                       for _ in args.models]
        model = MODELS['ModelEnsemble'](
            {m: MODELS[m][0](MODELS[m][1], device=devices[i])
             for i, m in enumerate(args.models)},
            ENSEMBLING_STRATEGIES[args.ensembling_strategy](scalers))
        device = devices[0]
    else:
        args.models = args.models[0]
        device_str = args.device[0] if isinstance(args.device, list) else args.device
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        model = MODELS[args.models][0](
            MODELS[args.models][1], device=device)

    if args.eval_model:
        device_str = args.device[0] if isinstance(args.device, list) else args.device
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        eval_model = MODELS[args.eval_model][0](
            MODELS[args.eval_model][1], device=device)

    data_loaders, _ = get_dataloader(path=args.path_to_dataset,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        device=device,
        subset=args.subset,
        sample_seed=args.sample_seed)

    #########################################################################
    #                     Creating and running the attack                   #
    #########################################################################


    if args.ensembling_strategy == "raw" and isinstance(args.models, list):
        loss = args.ensemble_loss
        trackers = [
            LossTracker(),
            MajorityVotingPredictionTracker(),
            PerturbationNormTracker("linf"),
            GradientNormTracker(),
            EnsemblePredictionTracker(),
            EnsembleScoresTracker(),
            AvgEnsembleScoresTracker(),
        ]
    else:
        loss = "ce"
        trackers = [
            LossTracker(),
            PredictionTracker(),
            PerturbationNormTracker("linf"),
            GradientNormTracker(),
            ScoresTracker(),
        ]

    # Create and run attack (PGD attack with Linf and CE loss)
    perturbation_model = LpPerturbationModels.LINF
    y_target = None  # Untargeted attack: push away from ground truth
    epsilon = float(eval(args.epsilon))
    num_steps = args.num_steps
    step_size = args.step_size

    os.makedirs("trackers/logs/pgd", exist_ok=True)
    tensorboard_tracker = TensorboardTracker("trackers/logs/pgd", trackers)

    if isinstance(args.models, list):
        attack_cls = ATTACK_CLASSES[args.attack]
        native_attack = attack_cls(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=True if args.random_start else False,
            loss_function=ENSEMBLE_LOSSES[loss],
            y_target=y_target,
            trackers=tensorboard_tracker,
        )
    else:
        native_attack = PGD(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=True if args.random_start else False,
            y_target=y_target,
            backend=Backends.NATIVE,
            trackers=tensorboard_tracker,
        )

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting attack at: {current_time}")
    native_adv_ds = {}
    if args.generate:
        adv_dataset = {}

    if args.eval_model:
        all_predictions = []
        all_probs = []
        all_p_labels = []
        all_labels = []
        all_logits = []
        metrics = {}

    for k, data_loader in tqdm(data_loaders.items(), desc="Generating adversarial samples"):
        native_adv_subset = native_attack(model, data_loader)

        # Get original and adversarial examples
        orig_images, orig_labels = next(iter(data_loader))
        adv_images, adv_labels = next(iter(native_adv_subset))
        
        # Save and plot adversarial examples
        plot_adv_example(output_dir, orig_images, orig_labels, adv_images, adv_labels, 
                        tensorboard_tracker.trackers, num_steps, epsilon)

        accuracy = Accuracy()(model, native_adv_subset)
        print(f"{k} Initial test accuracy: {Accuracy()(model, data_loader):.2f}, Adversarial test accuracy: {accuracy.item():.2f}")

    #########################################################################
    #       Evaluating the adversarial attack on the provided model         #
    #             Saving the dataset of adversarial images                  #
    #########################################################################

        if args.generate:
            adv_images_list = []
            adv_labels_list = []
        if args.eval_model:
            predictions = []
            probs_list = []
            p_labels = []
            labels = []

        for batch in native_adv_subset:
            adv_images_b, adv_labels_b = batch
            if args.generate:
                adv_images_list.append(adv_images_b.cpu())
                adv_labels_list.append(adv_labels_b.cpu())
            if args.eval_model:
                logits_batch = eval_model.decision_function(adv_images_b)
                all_logits.extend(logits_batch.cpu().tolist())
                p_labels.extend(eval_model.predict(adv_images_b).cpu().detach())
                probs = torch.softmax(logits_batch, dim=1)[:, 1]
                probs_list.extend(probs.cpu().tolist())
                predictions.extend((probs > THRESHOLD).int().cpu().tolist())
                labels.extend(adv_labels_b.cpu().detach())

        if args.generate:
            adv_images = torch.cat(adv_images_list, dim=0)
            adv_labels = torch.cat(adv_labels_list, dim=0)
            adv_dataset[k] = {
                'images' : adv_images,
                'labels' : adv_labels,
            }
        if args.eval_model:
            metrics[k] = {
                "acc": accuracy_score(y_true=labels, y_pred=p_labels)
                }
            all_predictions.extend(predictions)
            all_probs.extend(probs_list)
            all_p_labels.extend(p_labels)
            all_labels.extend(labels)

        del native_adv_subset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Ending attack at: {current_time}")
    if args.generate:
        save_path = os.path.join(output_dir, 'adv_dataset.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(adv_dataset, f)
        print(f"Save on disk pickle file")
    if args.eval_model:
        metrics['All'] = {
            "f1": f1_score(y_true=all_labels, y_pred=all_p_labels),
            "acc": accuracy_score(y_true=all_labels, y_pred=all_p_labels),
            "auc": roc_auc_score(y_true=all_labels, y_score=all_probs),
            }
        with open(f"{eval_output_dir}/results.json", "w") as f:
            json.dump(
                {"config": vars(args), "metrics": metrics, "predictions": predictions, "logits": all_logits},
                f,
            )
    print('done!')


def parse_args():
    parser = argparse.ArgumentParser(description="PGD Attack on Model Ensemble")
    parser.add_argument("--overwrite", type=int, default=0, help="Set to 1 to overwrite existing results")
    parser.add_argument("--random_start", type=int, default=0, help="Whether to use random start during the initilization of PGD")
    parser.add_argument("--random_seed", type=int, help="random seed. Set only if the random start is true for PGD")
    parser.add_argument("--trained_d3", action="store_true", help="Load checkpoints trained on D3")
    parser.add_argument("--generate", action="store_true", help="Generate adversarial tensors dataset if this flag is set")
    parser.add_argument("--generate_images", action="store_true", help="Generate adversarial images dataset if this flag is set")
    parser.add_argument("--eval_model", type=str, choices=MODELS, help="Model to be evaluated")
    parser.add_argument("--eval_output_dir", type=str, default="output/evaluate_detector", help="Directory to save evaluation results") 
    parser.add_argument("--models", type=str, nargs="+", default="chen2024_convnext", help="List of Ensemble models")
    parser.add_argument("--device", type=str, nargs="+", default="cuda",
                        help="Device on which to run the models. If a list is passed, "
                             "its length must be equal to the number of models,"
                             "and each model will be run on the corresponding device.")
    parser.add_argument("--path_to_dataset", default="data/ELSA_TEST")
    parser.add_argument("--dataset_type", default="subfolders")
    parser.add_argument("--output_dir", default="data", help="Directory to save adversarial images")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for data loader")
    parser.add_argument("--epsilon", type=str, default='32/255', help="Perturbation budget for PGD")
    parser.add_argument("--num_steps", type=int, default=10, help="PGD steps number")
    parser.add_argument("--step_size", type=float, default=5e-2, help="PGD Step size")
    parser.add_argument("--subset", type=int, default=-1, help="test subset size (-1 for full test set)")
    parser.add_argument("--sample_seed", type=int, default=None, help="Seed for random dataset subsampling (different seed = different images)")
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "apgd", "cwa"],
                        help="Ensemble attack algorithm to use (only affects ensemble/multi-model runs).")
    parser.add_argument("--ensembling_strategy", type=str, default="raw", choices=["raw", "avg", "random"],
                        help="Defines the ensembling strategy.")
    parser.add_argument("--ensemble_loss", type=str, default="avg_ce", choices=["avg_ce"],
                        help="Defines the ensemble loss to be used in the attack. "
                             "Only used if the ensembling strategy is `raw`.")
    parser.add_argument("--scale_logits", action="store_true",
                        help="scales the logits in the ensemble. "
                             "Only used if the ensembling strategy is `raw`.")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
