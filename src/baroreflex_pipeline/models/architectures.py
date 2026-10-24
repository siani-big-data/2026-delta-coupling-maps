from __future__ import annotations

import torch
from torch import nn

from .base import CardiovascularRegressor


class CovariateBranch(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())


class ScalarReferenceModel(CardiovascularRegressor):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.physiology = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.covariates = CovariateBranch()
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, scalars: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.physiology(scalars), self.covariates(covariates)), dim=1)
        return self.regressor(combined).squeeze(1)


class ClinicalReferenceModel(CardiovascularRegressor):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.physiology = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.covariates = CovariateBranch()
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, indicators: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.physiology(indicators), self.covariates(covariates)), dim=1)
        return self.regressor(combined).squeeze(1)


class DCMReferenceModel(CardiovascularRegressor):
    def __init__(self, image_shape: tuple[int, int]) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (image_shape[0] // 4) * (image_shape[1] // 4), 128),
            nn.ReLU(),
        )
        self.covariates = CovariateBranch()
        self.regressor = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, dcm: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.image(dcm), self.covariates(covariates)), dim=1)
        return self.regressor(combined).squeeze(1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class AblationReferenceModel(CardiovascularRegressor):
    def __init__(
        self,
        representation: str,
        *,
        use_covariates: bool,
        dcm_channels: tuple[int, ...],
        input_features: int | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        if representation not in {"clinical", "scalar", "dcm", "covariates"}:
            raise ValueError(f"Unknown representation: {representation}")
        if representation == "covariates" and not use_covariates:
            raise ValueError("Covariate-only model requires covariates")
        if (
            not dcm_channels
            or len(set(dcm_channels)) != len(dcm_channels)
            or any(c not in (0, 1) for c in dcm_channels)
        ):
            raise ValueError("DCM channels must be a nonempty unique subset of (0, 1)")
        self.representation = representation
        self.dcm_channels = dcm_channels
        self.physiology: nn.Module | None = None
        width = 0
        if representation == "clinical":
            self.physiology = nn.Sequential(
                nn.Linear(input_features, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
            )
            width = 64
        elif representation == "scalar":
            self.physiology = nn.Sequential(
                nn.Linear(input_features, 128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            width = 64
        elif representation == "dcm":
            self.physiology = nn.Sequential(
                nn.Conv2d(len(dcm_channels), 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * (image_shape[0] // 4) * (image_shape[1] // 4), 128),
                nn.ReLU(),
            )
            width = 128
        self.covariates = CovariateBranch() if use_covariates else None
        width += 64 if use_covariates else 0
        self.regressor = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, inputs: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.physiology is not None:
            if self.representation == "dcm":
                inputs = inputs[:, self.dcm_channels, :, :]
            parts.append(self.physiology(inputs))
        if self.covariates is not None:
            parts.append(self.covariates(covariates))
        return self.regressor(torch.cat(parts, dim=1)).squeeze(1)
