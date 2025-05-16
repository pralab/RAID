from pathlib import Path
from typing import Callable, List, Union
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, TensorDataset
from torchvision.transforms import ToTensor, CenterCrop, Compose, Resize
import pickle


def get_dataloader(
    path: str,
    transform: Callable = [CenterCrop((256, 256)), ToTensor()],
    dataset_type: str = "subfolders",
    image_suffixes: List[str] = [".png", ".jpg", ".jpeg"],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    device: Union[str, torch.device] = "cpu",
    subset: int = -1,
):
    dataloaders = {}
    file_paths = {}
    path = Path(path)

    if dataset_type == "subfolders":
        for folder in sorted(path.iterdir()):
            if not 'modern_generator' in str(folder.resolve()):          
                label = 0 if folder.name == "real" else 1
                paths = [
                    str(file)
                    for file in sorted(folder.iterdir())
                    if file.suffix.lower() in image_suffixes
                ]
                labels = [torch.tensor(label)] * len(paths)

                ds = PathDataset(paths=paths, labels=labels, transform=Compose(transform)) if (subset == -1) else (
                Subset(PathDataset(paths=paths, labels=labels, transform=Compose(transform)), list(range(subset))))

                dataloaders[folder.name] = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=lambda batch: custom_collate_fn(batch, device),
                )
                file_paths[folder.name] = paths

    elif dataset_type == "wang2020": # not used
        for subfolder in sorted(path.iterdir()):
            real_paths = [
                str(file)
                for file in sorted(subfolder.rglob("0_real/*.*"))
                if file.suffix.lower() in image_suffixes
            ]
            fake_paths = [
                str(file)
                for file in sorted(subfolder.rglob("1_fake/*.*"))
                if file.suffix.lower() in image_suffixes
            ]
            labels_real = [torch.tensor(0)] * len(real_paths)
            labels_fake = [torch.tensor(1)] * len(fake_paths)

            dataloaders[f"{subfolder.name}_real"] = DataLoader(
                PathDataset(paths=real_paths, labels=labels_real, transform=Compose(transform)),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=lambda batch: custom_collate_fn(batch, device),
            )
            file_paths[f"{subfolder.name}_real"] = real_paths

            dataloaders[f"{subfolder.name}_fake"] = DataLoader(
                PathDataset(paths=fake_paths, labels=labels_fake, transform=Compose(transform)),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=lambda batch: custom_collate_fn(batch, device),
            )
            file_paths[f"{subfolder.name}_fake"] = fake_paths

    elif dataset_type == "dataset":
        with open(path, "rb") as f:
            datasets = pickle.load(f)
        print(f"Loaded pickle file")

        for k, dataset in datasets.items():
            dataset = dataset if subset == -1 else Subset(dataset, list(range(subset)))

            dataloaders[k] = DataLoader(
                TensorDataset(dataset['images'], dataset['labels']),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=lambda batch: custom_collate_fn(batch, device),
            )
        file_paths = None

    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    return dataloaders, file_paths


class PathDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform: Callable | None = None):
        self.images = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def custom_collate_fn(batch, device):
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    labels = torch.stack(labels).to(device)
    return images, labels
