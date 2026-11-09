"""Minimal loader. The point of this file is that the safe path is the easy one.

    from load import load_annotated, load_production

    ds = load_annotated()               # leak-free features + labels + splits
    train, val, test = ds.temporal()    # primary protocol
    prod = load_production()            # cloud + gateway streams

Quarantined fields are reachable only by asking for them explicitly, and they
warn when you do.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ROOT, read_json

ANNOTATED = ROOT / "annotated"
PRODUCTION = ROOT / "production"

# Columns present in samples.parquet that are audit metadata, not model inputs.
AUDIT_COLUMNS = [
    "leak_terms", "has_leak_term", "audit_raw_screenshot_count",
    "raw_event_id", "raw_index",
]
LABEL_COLUMNS = ["need_help", "intention"]


@dataclass
class AnnotatedDataset:
    samples: pd.DataFrame
    screens: pd.DataFrame
    history: pd.DataFrame
    splits: dict

    @property
    def feature_columns(self) -> list[str]:
        drop = set(AUDIT_COLUMNS + LABEL_COLUMNS + ["sample_id", "user_name"])
        return [c for c in self.samples.columns if c not in drop]

    def temporal(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Primary protocol: per-user chronological 70/15/15."""
        s = self.splits["temporal"]
        return tuple(
            self.samples[self.samples.sample_id.isin(set(s[k]))].copy()
            for k in ("train", "val", "test")
        )

    def user_holdout(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cold-start protocol: disjoint users. Removes same-user history, so
        it measures generalisation to a new person, not personalisation."""
        s = self.splits["user_holdout"]
        return tuple(
            self.samples[self.samples.user_id.isin(set(s[k]))].copy()
            for k in ("train_users", "val_users", "test_users")
        )

    def context_for(self, sample_id: str) -> dict:
        """Everything a model is allowed to see for one sample."""
        row = self.samples.loc[self.samples.sample_id == sample_id].iloc[0]
        return {
            "user_id": row.user_id,
            "t0": row.t0,
            "app_name": row.app_name,
            "window_title": row.window_title,
            "screens": self.screens[self.screens.sample_id == sample_id]
            .sort_values("screen_order").to_dict("records"),
            "history": self.history[self.history.sample_id == sample_id]
            .sort_values("start_time").to_dict("records"),
        }

    def targets_for(self, sample_id: str) -> dict:
        """Labels plus the reference query, for suggestion-generation eval."""
        row = self.samples.loc[self.samples.sample_id == sample_id].iloc[0]
        conv = load_quarantine("current_conversation", _internal=True)
        hit = conv[conv.sample_id == sample_id]
        return {
            "need_help": bool(row.need_help),
            "intention": row.intention,
            "reference_query": hit["target_query"].iloc[0] if len(hit) else "",
        }


def load_annotated() -> AnnotatedDataset:
    return AnnotatedDataset(
        samples=pd.read_parquet(ANNOTATED / "samples.parquet"),
        screens=pd.read_parquet(ANNOTATED / "context_screens.parquet"),
        history=pd.read_parquet(ANNOTATED / "history_events.parquet"),
        splits=read_json(ANNOTATED / "splits.json"),
    )


def load_quarantine(name: str, _internal: bool = False) -> pd.DataFrame:
    """Load a quarantined table. These fields correlate with the label through
    their mere presence - see annotated/audit.json before using them as inputs.
    """
    path = ANNOTATED / "quarantine" / f"{name}.parquet"
    if not _internal:
        warnings.warn(
            f"'{name}' is quarantined: its presence alone predicts the label "
            "(history queries appear in 79.8% of positives and 0% of "
            "negatives). Valid as a target or under a matched design; not as a "
            "plain input feature.",
            stacklevel=2,
        )
    return pd.read_parquet(path)


@dataclass
class ProductionDataset:
    cloud: pd.DataFrame
    users: pd.DataFrame
    gateway_path: Path
    texts_path: Path

    def gateway(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.gateway_path, columns=columns)

    def texts(self, foreground_only: bool = True) -> pd.DataFrame:
        df = pd.read_parquet(self.texts_path)
        return df[df.is_foreground] if foreground_only else df

    def user_day(self) -> pd.DataFrame:
        """Per user-day aggregates from the cloud stream."""
        df = self.cloud.copy()
        df["day"] = df["timestamp"].dt.date
        agg = df.groupby(["user_id", "day"]).agg(
            n_calls=("request_id", "count"),
            n_errors=("is_error", "sum"),
            total_tokens=("total_tokens", "sum"),
            median_latency_ms=("latency_ms", "median"),
            n_devices=("device_id", "nunique"),
            n_models=("model_alias", "nunique"),
        ).reset_index()
        return agg


def load_production() -> ProductionDataset:
    return ProductionDataset(
        cloud=pd.read_parquet(PRODUCTION / "calls_cloud.parquet"),
        users=pd.read_parquet(PRODUCTION / "users.parquet"),
        gateway_path=PRODUCTION / "calls_gateway.parquet",
        texts_path=PRODUCTION / "texts_gateway.parquet",
    )


if __name__ == "__main__":
    ds = load_annotated()
    tr, va, te = ds.temporal()
    print(f"annotated: {len(ds.samples)} samples, "
          f"{ds.samples.user_id.nunique()} users")
    print(f"  temporal split: {len(tr)}/{len(va)}/{len(te)}")
    print(f"  feature columns: {ds.feature_columns}")
    prod = load_production()
    print(f"production cloud: {len(prod.cloud):,} calls, "
          f"{prod.cloud.user_id.nunique()} users")
    print(f"  user-day rows: {len(prod.user_day())}")
