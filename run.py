from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import aggregate_daily, discover_files, read_auxiliary
from src.models import apply_mid_autumn_adjustment, apply_v4_weekday_adjustment, stable_ensemble
from src.validation import rolling_month_validation, save_validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="天池资金流入流出预测竞赛代码")
    parser.add_argument("--data-dir", default="data", help="比赛数据文件所在目录")
    parser.add_argument("--output-dir", default="output", help="结果输出目录")
    parser.add_argument(
        "--mode",
        choices=("all", "validate", "predict"),
        default="all",
        help="all=验证并预测；validate=只验证；predict=只生成提交文件",
    )
    return parser.parse_args()


def generate_submission(daily: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """训练最终模型，并生成 2014 年 9 月提交文件。"""
    future_dates = pd.date_range("2014-09-01", "2014-09-30", freq="D")

    purchase = stable_ensemble(daily["date"], daily["total_purchase_amt"], future_dates, "purchase")
    redeem = stable_ensemble(daily["date"], daily["total_redeem_amt"], future_dates, "redeem")

    purchase, redeem = apply_mid_autumn_adjustment(future_dates, purchase, redeem)
    purchase, redeem = apply_v4_weekday_adjustment(future_dates, purchase, redeem)

    submission = pd.DataFrame(
        {
            "report_date": future_dates.strftime("%Y%m%d").astype(int),
            "total_purchase_amt": np.maximum(0, np.rint(purchase)).astype("int64"),
            "total_redeem_amt": np.maximum(0, np.rint(redeem)).astype("int64"),
        }
    )

    # 天池提交要求无表头。
    submission.to_csv(output_dir / "submission.csv", index=False, header=False)
    return submission


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.data_dir)
    print("发现的数据文件：")
    for key, path in files.items():
        print(f"  {key:8s}: {path}")

    daily = aggregate_daily(files["balance"])
    print(f"每日汇总：{len(daily)} 天，{daily['date'].min().date()} 至 {daily['date'].max().date()}")

    # 辅助表当前主要用于检查数据是否存在，后续可继续扩展特征。
    for name, frame in read_auxiliary(files).items():
        print(f"辅助表 {name}: {frame.shape}")

    if args.mode in ("all", "validate"):
        metrics, _ = rolling_month_validation(daily)
        summary = save_validation_summary(metrics, output_dir / "validation_summary.csv")
        print("\n滚动验证汇总：")
        print(summary.to_string(index=False))

    if args.mode in ("all", "predict"):
        submission = generate_submission(daily, output_dir)
        print("\n已生成提交文件：", output_dir / "submission.csv")
        print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
