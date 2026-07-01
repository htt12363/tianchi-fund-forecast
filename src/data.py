from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import zipfile

import pandas as pd


# 根据文件名关键词自动寻找比赛数据文件，避免文件名中带 (1)、(2) 时读取失败。
FILE_PATTERNS = {
    "balance": ("user_balance_table",),
    "profile": ("user_profile_table",),
    "interest": ("mfd_day_share_interest",),
    "shibor": ("mfd_bank_shibor",),
    "sample": ("comp_predict_table", "predict_table"),
}


def _find_one(data_dir: Path, keywords: Iterable[str], required: bool = True) -> Path | None:
    candidates = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(keyword.lower() in name for keyword in keywords):
            candidates.append(path)

    if candidates:
        # 主交易表优先读取 zip，可以减少磁盘占用；同类文件取文件名较短者。
        candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".zip" else 1, len(p.name)))
        return candidates[0]

    if required:
        raise FileNotFoundError(f"在 {data_dir} 中没有找到包含 {tuple(keywords)} 的文件")
    return None


def discover_files(data_dir: str | Path) -> Dict[str, Path | None]:
    """在 data 目录中自动查找比赛所需文件。"""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    return {
        key: _find_one(root, keywords, required=(key == "balance"))
        for key, keywords in FILE_PATTERNS.items()
    }


def _validate_zip(path: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    with zipfile.ZipFile(path) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"压缩包中没有 CSV 文件：{path}")


def aggregate_daily(balance_path: str | Path, chunksize: int = 300_000) -> pd.DataFrame:
    """分块读取用户交易明细，并汇总为每日申购、赎回总额。

    原始 user_balance_table 体积较大，直接一次性读入容易占内存。
    这里每次读取一部分数据，先在块内按日期汇总，最后再合并所有块。
    """
    path = Path(balance_path)
    _validate_zip(path)

    usecols = ["report_date", "total_purchase_amt", "total_redeem_amt"]
    compression = "zip" if path.suffix.lower() == ".zip" else "infer"
    parts = []

    for chunk in pd.read_csv(
        path,
        compression=compression,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"report_date": "int64"},
    ):
        chunk["total_purchase_amt"] = pd.to_numeric(
            chunk["total_purchase_amt"], errors="coerce"
        ).fillna(0.0)
        chunk["total_redeem_amt"] = pd.to_numeric(
            chunk["total_redeem_amt"], errors="coerce"
        ).fillna(0.0)
        parts.append(
            chunk.groupby("report_date", as_index=True)[
                ["total_purchase_amt", "total_redeem_amt"]
            ].sum()
        )

    if not parts:
        raise ValueError("用户交易表为空")

    daily = pd.concat(parts).groupby(level=0).sum().sort_index().reset_index()
    daily["date"] = pd.to_datetime(daily["report_date"].astype(str), format="%Y%m%d")

    # 检查日期是否连续，避免后续周期模型出错。
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    missing = full_dates.difference(pd.DatetimeIndex(daily["date"]))
    if len(missing) > 0:
        raise ValueError(f"日期不连续，缺少 {len(missing)} 天，例如：{list(missing[:5])}")

    return daily[["report_date", "date", "total_purchase_amt", "total_redeem_amt"]]


def read_auxiliary(files: Dict[str, Path | None]) -> Dict[str, pd.DataFrame]:
    """读取辅助表用于数据完整性检查。

    当前最终模型以交易日序列和日历规律为主，辅助表保留为后续改进方向。
    """
    result: Dict[str, pd.DataFrame] = {}
    for key in ("profile", "interest", "shibor"):
        path = files.get(key)
        if path is not None:
            result[key] = pd.read_csv(path)
    return result
