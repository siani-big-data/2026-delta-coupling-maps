from torch import nn

from .architectures import (
    AblationReferenceModel,
    ClinicalReferenceModel,
    DCMReferenceModel,
    ScalarReferenceModel,
)
from .specs import ABLATION_SPECS, MODEL_INPUTS


def build_model(name: str, input_shape: tuple[int, ...]) -> nn.Module:
    if name not in MODEL_INPUTS:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_INPUTS)}")
    representation = MODEL_INPUTS[name]
    if input_shape is None:
        raise ValueError("Model input dimensions must be supplied explicitly")
    shape = tuple(input_shape)
    if representation == "dcm":
        if len(shape) != 3 or shape[0] != 2 or min(shape[1:]) < 4:
            raise ValueError(f"DCM inputs require (2, height>=4, width>=4), got {shape}")
    elif len(shape) != 1 or shape[0] < 1:
        raise ValueError(f"Tabular inputs require a positive feature width, got {shape}")
    if name == "clinical":
        return ClinicalReferenceModel(shape[0])
    if name == "scalar":
        return ScalarReferenceModel(shape[0])
    if name == "dcm":
        return DCMReferenceModel((shape[1], shape[2]))
    _, physiology, covariates, channels = ABLATION_SPECS[name]
    return AblationReferenceModel(
        physiology,
        use_covariates=covariates,
        dcm_channels=channels,
        input_features=shape[0] if len(shape) == 1 else None,
        image_shape=(shape[1], shape[2]) if len(shape) == 3 else None,
    )
