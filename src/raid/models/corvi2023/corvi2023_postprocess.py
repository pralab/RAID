import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class Corvi2023PostProcess(DataProcessing):

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x,x], dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
