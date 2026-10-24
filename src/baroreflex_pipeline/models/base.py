from abc import ABC, abstractmethod

import torch
from torch import nn


class CardiovascularRegressor(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError
