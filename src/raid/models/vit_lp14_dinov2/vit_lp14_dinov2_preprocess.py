import torch
from PIL import Image
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class Vit_lp14_dinov2PreProcess(DataProcessing):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(518),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in x])
        
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
