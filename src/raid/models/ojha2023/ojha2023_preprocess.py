import torch
from PIL import Image
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class Ojha2023PreProcess(DataProcessing):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in x])

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
