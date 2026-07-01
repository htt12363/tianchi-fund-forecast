from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .models import (
    MODEL_NAMES,
    apply_v4_weekday_adjustment,
    candidate_matrix,
    ridge_calendar_candidate,
    stable_ensemble,
)


def error_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """计算时间序列预测误差。"""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ape = np.abs(pred - actual) / np.maximum(actual, 1.0)
    return {
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "max_ape": float(ape.max()),
        "over_30pct_ratio": float((ape > 0.30).mean()),
    }


def rolling_month_validation(
    daily: pd.DataFrame,
    months: Iterable[int] = (4, 5, 6, 7, 8),
    year: int = 2014,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """滚动月份验证。

    时间序列不能随机切分，所以这里用历史数据预测后一个月，
    依次模拟 2014 年 4 月到 8 月的预测效果。
    """
    records = []
    predictions = []
    dates = pd.DatetimeIndex(daily["date"])

    for month in months:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        train = daily.loc[dates < start]
        test = daily.loc[(dates >= start) & (dates <= end)]
        if train.empty or test.empty:
            continue

        for target, column in (
            ("purchase", "total_purchase_amt"),
            ("redeem", "total_redeem_amt"),
        ):
            matrix = candidate_matrix(train["date"], train[column], test["date"])
            model_preds = {name: matrix[:, i] for i, name in enumerate(MODEL_NAMES)}
            model_preds["ridge_calendar"] = ridge_calendar_candidate(train["date"], train[column], test["date"])

            base_pred = stable_ensemble(train["date"], train[column], test["date"], target)
            model_preds["stable_ensemble"] = base_pred

            # 第四版提交模型：稳健融合 + 小幅星期几修正。
            if target == "purchase":
                v4_pred, _ = apply_v4_weekday_adjustment(test["date"], base_pred, base_pred)
            else:
                _, v4_pred = apply_v4_weekday_adjustment(test["date"], base_pred, base_pred)
            model_preds["stable_ensemble_v4"] = v4_pred

            actual = test[column].to_numpy(dtype=float)
            for model_name, pred in model_preds.items():
                records.append(
                    {
                        "year_month": f"{year}-{month:02d}",
                        "target": target,
                        "model": model_name,
                        **error_metrics(actual, pred),
                    }
                )
                for d, a, p in zip(test["date"], actual, pred):
                    predictions.append(
                        {
                            "date": d,
                            "target": target,
                            "model": model_name,
                            "actual": a,
                            "prediction": p,
                        }
                    )

    return pd.DataFrame(records), pd.DataFrame(predictions)


def save_validation_summary(metrics: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """保存按模型汇总后的验证结果。"""
    summary = (
        metrics.groupby(["target", "model"], as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            mean_median_ape=("median_ape", "mean"),
            worst_max_ape=("max_ape", "max"),
            mean_over_30pct_ratio=("over_30pct_ratio", "mean"),
        )
        .sort_values(["target", "mean_mape"])
    )
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return summary
