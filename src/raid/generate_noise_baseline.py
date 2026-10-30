"""Generate Gaussian / uniform noise baselines matched by L-inf norm."""
import argparse
import os
import pickle

import torch
from data import get_dataloader


def add_noise(images, epsilon, noise_type, seed=42):
    gen = torch.Generator(device=images.device).manual_seed(seed)
    if noise_type == "gaussian":
        noise = torch.randn(images.shape, generator=gen, device=images.device) * epsilon
        noise = noise.clamp(-epsilon, epsilon)
    elif noise_type == "uniform":
        noise = torch.empty(images.shape, device=images.device).uniform_(
            -epsilon, epsilon, generator=gen
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return (images + noise).clamp(0.0, 1.0)


def main(args):
    device = torch.device("cpu")

    data_loaders, _ = get_dataloader(
        path=args.path_to_dataset,
        dataset_type="subfolders",
        batch_size=args.batch_size,
        device=device,
        subset=args.subset,
    )

    epsilon = float(eval(args.epsilon))

    dataset_name = args.path_to_dataset.rstrip("/").split("/")[-1]
    subset_suffix = f"_{args.subset}" if args.subset != -1 else ""
    output_dir = os.path.join(
        args.output_dir,
        "ADV",
        f"{dataset_name}{subset_suffix}",
        f"noise_{args.noise_type}_{args.epsilon.replace('/', '_')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    adv_dataset = {}
    for k, data_loader in data_loaders.items():
        all_images = []
        all_labels = []
        for images, labels in data_loader:
            noisy = add_noise(images, epsilon, args.noise_type, seed=args.seed)
            all_images.append(noisy)
            all_labels.append(labels)

        adv_dataset[k] = {
            "images": torch.cat(all_images, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }
        print(f"  {k}: {adv_dataset[k]['images'].shape[0]} images")

    save_path = os.path.join(output_dir, "adv_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(adv_dataset, f)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noise baselines for ablation")
    parser.add_argument("--path_to_dataset", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--noise_type", choices=["gaussian", "uniform"], required=True)
    parser.add_argument("--epsilon", required=True, help="e.g. 16/255")
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
