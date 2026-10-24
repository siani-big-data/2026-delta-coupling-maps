from pathlib import Path

import pandas as pd

from .base import TabularFeatureExtractor
from .clinical import extract_clinical_panel, save_clinical_panel
from .scalars import extract_scalar_panel, save_scalar_panel


class ClinicalExtractor(TabularFeatureExtractor):
    def extract(self, beat_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return extract_clinical_panel(beat_tables)

    def save(self, features: pd.DataFrame, output: Path) -> None:
        save_clinical_panel(features, output)


class TsfreshExtractor(TabularFeatureExtractor):
    def __init__(self, feature_set: str, n_jobs: int) -> None:
        self.feature_set = feature_set
        self.n_jobs = n_jobs

    def extract(self, beat_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return extract_scalar_panel(beat_tables, feature_set=self.feature_set, n_jobs=self.n_jobs)

    def save(self, features: pd.DataFrame, output: Path) -> None:
        save_scalar_panel(features, output, self.feature_set)
