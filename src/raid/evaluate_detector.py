import argparse
import datetime
import json
import pprint
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

#from src.detectors_code.sha2023.test import NeuralNet  # noqa
from data import get_dataloader

from constants import MODELS


THRESHOLD = 0.5


@torch.no_grad()
def main(args):
    pprint.pp(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # create output directory
    if not args.dry_run:
        if args.dataset_type == 'subfolders':
            path_name = Path(args.path_to_dataset).parts[1:]
        else:
            path_name = Path(args.path_to_dataset).parts[1:-1]

        output_dir = Path(
            f"{args.output_dir}/"
            f"{Path(*path_name)}{'_'+str(args.subset) if args.subset != -1 else ''}/"
            f"{args.model}"
        )

        if (output_dir.exists()):
            print(f"A directory with this name already exist, pay attention to not overwrite data: {output_dir}")
        else: output_dir.mkdir(exist_ok=True, parents=True)

    # load model
    if 'ModelEnsemble' not in args.model:
        if args.path_to_checkpoint is None:
            checkpoint_path = MODELS[args.model][1]
        elif len(args.path_to_checkpoint) == 1:
            checkpoint_path = args.path_to_checkpoint[0]
        else:
            checkpoint_path = args.path_to_checkpoint
        model = MODELS[args.model][0](checkpoint_path=checkpoint_path, device=device)
    else:
        model = MODELS[args.ensemble_method](
            {m: MODELS[m][0](MODELS[m][1], device=device)
             for m in args.ensemble_models}
        )

    # load dataset
    dataloaders, _ = get_dataloader(path=args.path_to_dataset,
                                    dataset_type=args.dataset_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    subset=args.subset)

    # compute predictions
    all_predictions = []
    all_p_labels = []
    all_labels = []
    all_logits = []

    metrics = {}

    for k, v in dataloaders.items():
        predictions = []
        p_labels = []
        labels = []
        for idx, (images, batch_labels) in enumerate(tqdm(v, desc="Processing batches")):
            logits_batch = model.decision_function(images)
            all_logits.extend(logits_batch.cpu().tolist())
            p_labels.extend(model.predict(images).cpu().detach())
            probs = torch.softmax(logits_batch, dim=1)[:, 1]
            predictions.extend((probs > THRESHOLD).int().cpu().tolist())
            labels.extend(batch_labels.cpu().detach())

        metrics[k] = {
            "acc": accuracy_score(y_true=labels, y_pred=p_labels)
        }

        all_predictions.extend(predictions)
        all_p_labels.extend(p_labels)
        all_labels.extend(labels)

    metrics['All'] = {
        "f1": f1_score(y_true=all_labels, y_pred=all_p_labels),
        "acc": accuracy_score(y_true=all_labels, y_pred=all_p_labels),
        "auc": roc_auc_score(y_true=all_labels, y_score=all_predictions),
        "ap": average_precision_score(y_true=all_labels, y_score=all_predictions),
    }


    # save results
    if not args.dry_run:
        with open(output_dir / "results.json", "w") as f:
            json.dump(
                {"config": vars(args), "metrics": metrics, "predictions": predictions, "logits": all_logits},
                f,
            )
            import os
            print(f'Print: {os.path.join(output_dir, "results.json")}')

    print(f"Result at: {output_dir}")
    print('done!')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_models", action="store_true", help="Load external models")
    parser.add_argument("--N_1_out", action="store_true", help="Define evaluation mode to take out one single detector at a time")
    parser.add_argument("--trained_d3", action="store_true", help="Load checkpoints trained on D3")
    parser.add_argument("--epsilon", type=str, default='32/255', help="Perturbation budget for PGD, used for logging")
    parser.add_argument("--model", choices=MODELS, default="wang2020")
    parser.add_argument(
        "--path-to-checkpoint",
        nargs="+",
    )
    parser.add_argument("--path_to_dataset", default="data/ELSA_D3_1000")
    parser.add_argument("--dataset_type", default="subfolders")
    parser.add_argument("--output_dir", default="output/evaluate_detector")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--subset", type=int, default=-1, help="test subset size (-1 for full test set)")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
