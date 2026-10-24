from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class TabularFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, beat_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, features: pd.DataFrame, output: Path) -> None:
        raise NotImplementedError

    def extract_from_directory(self, beats_dir: Path, output: Path) -> pd.DataFrame:
        paths = sorted(beats_dir.glob("*.csv.gz"))
        if not paths:
            raise FileNotFoundError(f"No beat files found in {beats_dir}")
        tables = {path.name.removesuffix(".csv.gz"): pd.read_csv(path) for path in paths}
        features = self.extract(tables)
        self.save(features, output)
        return features
