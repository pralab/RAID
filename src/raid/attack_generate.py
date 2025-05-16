import argparse
import pprint
from pathlib import Path
import torch
from tqdm import tqdm
from attacks import EnsemblePGD
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
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score
import json


THRESHOLD = 0.5


def main(args):
    pprint.pp(vars(args))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.dry_run:
        output_dir = Path(
            f"{args.output_dir}/ADV/"
            f"{args.path_to_dataset.split('/')[-1]}{'_'+str(args.subset) if args.subset != -1 else ''}/"
            f"adv_{args.ensembling_strategy}_{args.models}_{args.epsilon.replace('/','_')}"
            f"_{args.num_steps}_{args.step_size}"
        )
        if (
            output_dir.exists()
            and input(f"Output directory: {output_dir} exists, continue and overwrite? (y/n): ") != "y"
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
                and input(f"Output directory: {eval_output_dir} exists, continue and overwrite? (y/n): ") != "y"
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
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = MODELS[args.models][0](
            MODELS[args.models][1], device=device)

    if args.eval_model:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        eval_model = MODELS[args.eval_model][0](
            MODELS[args.eval_model][1], device=device)

    data_loaders, _ = get_dataloader(path=args.path_to_dataset, 
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        device=device,
        subset=args.subset)

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
    y_target = 0 # Misclassify AIG images as real
    epsilon = float(eval(args.epsilon))
    num_steps = args.num_steps
    step_size = args.step_size

    os.makedirs("trackers/logs/pgd", exist_ok=True)
    tensorboard_tracker = TensorboardTracker("trackers/logs/pgd", trackers)

    if isinstance(args.models, list):
        native_attack = EnsemblePGD(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=False,
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
            random_start=False,
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
            all_p_labels.extend(p_labels)
            all_labels.extend(labels)

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
            "auc": roc_auc_score(y_true=all_labels, y_score=all_predictions),
            "ap": average_precision_score(y_true=all_labels, y_score=all_predictions),
            }
        with open(f"{eval_output_dir}/results.json", "w") as f:
            json.dump(
                {"config": vars(args), "metrics": metrics, "predictions": predictions, "logits": all_logits},
                f,
            )
    print('done!')


def parse_args():
    parser = argparse.ArgumentParser(description="PGD Attack on Model Ensemble")
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
