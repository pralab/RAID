import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class Koutlis2024PostProcess(DataProcessing):
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x[0],x[0]], dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
