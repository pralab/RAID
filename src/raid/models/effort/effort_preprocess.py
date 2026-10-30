import torch
import torch.nn.functional as F
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class EffortPreProcess(DataProcessing):

    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return torch.stack([self.normalize(img) for img in x])

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
