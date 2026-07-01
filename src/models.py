from __future__ import annotations

from datetime import timedelta
from typing import Sequence
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# 由 2014 年 4—8 月滚动验证选择出的稳健融合权重。
# 申购和赎回的规律不同，所以分别设置权重。
PURCHASE_WEIGHTS = np.array([0.01016211, 0.57487546, 0.31864342, 0.02608711, 0.07023190])
REDEEM_WEIGHTS = np.array([0.39749502, 0.00151383, 0.05679872, 0.53024969, 0.01394274])
MODEL_NAMES = ["calendar_seasonal", "same_weekday_4", "same_weekday_8", "same_weekday_12", "holt_winters"]


def calendar_seasonal_base(
    train_dates: Sequence[pd.Timestamp],
    train_y: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    window_days: int = 184,
) -> np.ndarray:
    """最近半年日历模型：星期几因子 × 月内日期因子。"""
    dates = list(pd.to_datetime(train_dates))
    y = np.asarray(train_y, dtype=float)
    cutoff = dates[-1]
    idx = [i for i, d in enumerate(dates) if (cutoff - d).days < window_days]
    td = [dates[i] for i in idx]
    z = y[idx]

    overall = z.mean()
    weekday_mean = np.array([
        np.mean([z[i] for i, d in enumerate(td) if d.weekday() == w])
        for w in range(7)
    ])
    weekday_factor = weekday_mean / overall

    day_mean: dict[int, float] = {}
    day_weekday_factor: dict[int, float] = {}
    for day in range(1, 32):
        j = [i for i, d in enumerate(td) if d.day == day]
        if j:
            day_mean[day] = float(z[j].mean())
            day_weekday_factor[day] = float(np.mean([weekday_factor[td[i].weekday()] for i in j]))

    preds = []
    for d in pd.to_datetime(future_dates):
        base = day_mean.get(d.day, overall) / max(day_weekday_factor.get(d.day, 1.0), 1e-9)
        preds.append(base * weekday_factor[d.weekday()])
    return np.asarray(preds)


def weighted_same_weekday(
    train_dates: Sequence[pd.Timestamp],
    train_y: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    weeks: int,
    decay: float,
) -> np.ndarray:
    """取目标日前若干周同星期几的数值，并按时间远近做指数加权。"""
    dates = list(pd.to_datetime(train_dates))
    position = {d: i for i, d in enumerate(dates)}
    y = np.asarray(train_y, dtype=float)
    preds = []

    for d in pd.to_datetime(future_dates):
        values, weights = [], []
        for k in range(1, weeks + 1):
            old_date = d - timedelta(days=7 * k)
            if old_date in position:
                values.append(y[position[old_date]])
                weights.append(decay ** (k - 1))

        pred = float(np.average(values, weights=weights)) if values else float(y[-28:].mean())

        # 轻微趋势修正，避免预测完全停留在过去水平。
        if len(y) >= 56:
            recent_ratio = np.clip(y[-28:].mean() / max(y[-56:-28].mean(), 1.0), 0.9, 1.1)
            horizon = (d - dates[-1]).days
            pred *= recent_ratio ** ((horizon / 28.0) * 0.4)
        preds.append(pred)
    return np.asarray(preds)


def holt_winters_log(train_y: Sequence[float], horizon: int) -> np.ndarray:
    """Holt-Winters 七日周期模型，在对数尺度上拟合以降低极端值影响。"""
    y = np.asarray(train_y, dtype=float)
    z = np.log1p(y)
    model = ExponentialSmoothing(
        z,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    return np.expm1(model.forecast(horizon))


def _calendar_features(dates: Sequence[pd.Timestamp], origin: pd.Timestamp) -> np.ndarray:
    """构造 Ridge 回归使用的日历特征。"""
    rows = []
    for d in pd.to_datetime(dates):
        t = (d - origin).days
        dow, dom, doy = d.weekday(), d.day, d.dayofyear
        row = [
            t / 400.0,
            (t / 400.0) ** 2,
            dom / 31.0,
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * dom / 31),
            math.cos(2 * math.pi * dom / 31),
            math.sin(2 * math.pi * doy / 365.25),
            math.cos(2 * math.pi * doy / 365.25),
            float(dow >= 5),
            float(dom <= 3),
            float(dom >= 27),
        ]
        row.extend(float(dow == i) for i in range(7))
        row.extend(float(d.month == i) for i in range(1, 13))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def ridge_calendar_candidate(
    train_dates: Sequence[pd.Timestamp],
    train_y: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    alpha: float = 20.0,
    window: int = 365,
) -> np.ndarray:
    """Ridge 日历回归，作为机器学习对照模型。"""
    dates = list(pd.to_datetime(train_dates))[-window:]
    y = np.asarray(train_y, dtype=float)[-window:]
    origin = dates[0]
    x = _calendar_features(dates, origin)
    xt = _calendar_features(future_dates, origin)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, np.log1p(y))
    return np.expm1(model.predict(xt))


def candidate_matrix(
    train_dates: Sequence[pd.Timestamp],
    train_y: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    """生成所有候选模型的预测矩阵。"""
    horizon = len(future_dates)
    return np.vstack(
        [
            calendar_seasonal_base(train_dates, train_y, future_dates),
            weighted_same_weekday(train_dates, train_y, future_dates, 4, 0.78),
            weighted_same_weekday(train_dates, train_y, future_dates, 8, 0.82),
            weighted_same_weekday(train_dates, train_y, future_dates, 12, 0.86),
            holt_winters_log(train_y, horizon),
        ]
    ).T


def stable_ensemble(
    train_dates: Sequence[pd.Timestamp],
    train_y: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    target: str,
) -> np.ndarray:
    """候选模型加权融合。"""
    matrix = candidate_matrix(train_dates, train_y, future_dates)
    if target == "purchase":
        weights = PURCHASE_WEIGHTS
    elif target == "redeem":
        weights = REDEEM_WEIGHTS
    else:
        raise ValueError("target 必须是 purchase 或 redeem")
    return matrix @ weights


def apply_mid_autumn_adjustment(
    future_dates: Sequence[pd.Timestamp], purchase: np.ndarray, redeem: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """对 2014 年 9 月中秋假期做保守修正。"""
    purchase = np.asarray(purchase, dtype=float).copy()
    redeem = np.asarray(redeem, dtype=float).copy()
    multipliers = {
        pd.Timestamp("2014-09-06"): (0.82, 1.05),
        pd.Timestamp("2014-09-07"): (0.76, 0.79),
        pd.Timestamp("2014-09-08"): (0.50, 0.58),
    }
    for i, d in enumerate(pd.to_datetime(future_dates)):
        if d in multipliers:
            p_rate, r_rate = multipliers[d]
            purchase[i] *= p_rate
            redeem[i] *= r_rate
    return purchase, redeem


# 第四版保守修正：
# 第二版和第三版线上分数都低于第一版，说明大幅改动容易过拟合。
# 第四版只在第一版稳健结果上做很小的星期几乘法修正，幅度控制在 2.5% 左右。
PURCHASE_V4_WEEKDAY_FACTOR = {
    0: 1.025,
    1: 1.025,
    2: 0.9915,
    3: 0.9855,
    4: 1.0155,
    5: 1.002,
    6: 1.012,
}

REDEEM_V4_WEEKDAY_FACTOR = {
    0: 1.025,
    1: 0.975,
    2: 1.0075,
    3: 0.991,
    4: 0.9975,
    5: 0.983,
    6: 1.025,
}


def apply_v4_weekday_adjustment(
    future_dates: Sequence[pd.Timestamp], purchase: np.ndarray, redeem: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """第四版低风险星期几修正。"""
    purchase = np.asarray(purchase, dtype=float).copy()
    redeem = np.asarray(redeem, dtype=float).copy()
    for i, d in enumerate(pd.to_datetime(future_dates)):
        dow = int(d.weekday())
        purchase[i] *= PURCHASE_V4_WEEKDAY_FACTOR[dow]
        redeem[i] *= REDEEM_V4_WEEKDAY_FACTOR[dow]
    return purchase, redeem
