from attacks.loss import AvgEnsembleLoss
from torch.nn import CrossEntropyLoss
from huggingface_hub import hf_hub_download
from models import (
    cavia2024,
    chen2024,
    corvi2023,
    koutlis2024,
    ojha2023,
    wang2020,
    EnsembleModel,
    vit_lp14_dinov2,
    vit_lp14_reg_dinov2,
    vit_lp16_siglip_384,
    vit_tp16_224_augreg_in21k,
)
from models.ensemble.ensemble_function import (
    RawEnsembleFunction,
    AvgEnsembleFunction,
    RandomEnsembleFunction
)


def _get_raid_ckpt(filename):
    return hf_hub_download(repo_id="aimagelab/RAID_ckpt", filename=filename)


# models with default checkpoints
MODELS = {
    "cavia2024": (cavia2024, _get_raid_ckpt("cavia2024/model_best.pth.tar")), # retrained
    "chen2024_convnext": (
        chen2024, _get_raid_ckpt("chen2024_convnext/16_acc0.9993.pth")
    ),
    "chen2024_clip": (
        chen2024, _get_raid_ckpt("chen2024_clip/last_acc0.9112.pth")
    ),
    "corvi2023": (corvi2023, _get_raid_ckpt("corvi2023/model_best.pth.tar")),
    "koutlis2024": (
        koutlis2024, _get_raid_ckpt("koutlis2024/model_ldm_trainable.pth")
    ),
    "ojha2023": (ojha2023, _get_raid_ckpt("ojha2023/model_best.pth.tar")),
    "wang2020": (wang2020, _get_raid_ckpt("wang2020/model_best.pth.tar")),

    "vit_lp14_dinov2": (
        vit_lp14_dinov2,
        _get_raid_ckpt("pretrained_linear/dinov2_retrain_d3/model_best.pth.tar")
    ),
    "vit_lp14_reg_dinov2": (
        vit_lp14_reg_dinov2,
        _get_raid_ckpt("pretrained_linear/dinov2_reg_retrain_d3/model_best.pth.tar")
    ),
    "vit_lp16_siglip_384": (
        vit_lp16_siglip_384,
        _get_raid_ckpt("pretrained_linear/siglip_retrain_d3/model_best.pth.tar")
    ),
    "vit_tp16_224_augreg_in21k": (
        vit_tp16_224_augreg_in21k,
        _get_raid_ckpt("pretrained_linear/vit_tiny_retrain_d3/model_best.pth.tar")
    ),
    "vit_tp16_224_code_augreg_in21k": (
        vit_tp16_224_augreg_in21k,
        _get_raid_ckpt("pretrained_linear/vit_tiny_code_retrain_d3/model_best.pth.tar")
    ),

    "ModelEnsemble" : EnsembleModel,

}

ENSEMBLING_STRATEGIES = {
    "raw": RawEnsembleFunction,
    "avg": AvgEnsembleFunction,
    "random": RandomEnsembleFunction
}

ENSEMBLE_LOSSES = {
    "avg_ce": AvgEnsembleLoss(),
    "ce" : CrossEntropyLoss(reduction="none"),
}
