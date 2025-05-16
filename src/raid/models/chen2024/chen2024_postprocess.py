import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class Chen2024PostProcess(DataProcessing):
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
