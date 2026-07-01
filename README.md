# 天池资金流入流出预测竞赛代码

## 1. 项目简介

这是我的机器学习课程设计竞赛代码，赛题为天池“资金流入流出预测”。任务是根据 2013-07-01 至 2014-08-31 的历史数据，预测 2014 年 9 月每天的申购总额和赎回总额。

原始用户交易表约 284 万条记录，涉及 28041 名用户。由于比赛最终评价的是每日总金额，所以本项目先将用户级交易数据按日期汇总，再基于日历周期和时间序列规律进行预测。

当前最终版本为第四版，天池线上最好成绩为 **131.0938**。

## 2. 方法说明

本项目没有使用深度学习，主要使用以下方法：

- 按日期汇总每日申购总额和赎回总额；
- 构造星期几、月内日期等时间规律；
- 使用最近 4 周、8 周、12 周同星期几的加权平均；
- 使用 Holt-Winters 七日周期模型；
- 使用 Ridge 日历回归作为机器学习对照模型；
- 对多个稳定模型进行加权融合；
- 对 2014 年 9 月中秋假期和星期几偏差做小幅修正。

线上提交过程中，第二版和第三版由于修正幅度较大，分数没有超过第一版。第四版改为在第一版基础上做更保守的小幅修正，最终分数提升到 **131.0938**。

## 3. 项目结构

```text
.
├── README.md
├── requirements.txt
├── run.py
├── src
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── validation.py
├── data
│   └── README.md
└── output
    ├── submission.csv
    └── validation_summary.csv
```

文件说明：

- `run.py`：主程序入口；
- `src/data.py`：数据文件查找、分块读取、每日汇总；
- `src/models.py`：候选模型、模型融合和第四版修正规则；
- `src/validation.py`：滚动月份验证和评价指标；
- `output/submission.csv`：第四版天池提交文件；
- `output/validation_summary.csv`：本地滚动验证结果汇总。

## 4. 运行环境

建议使用 Python 3.10 或 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

## 5. 数据准备

请从天池比赛页面下载原始数据，并放入 `data/` 文件夹。需要的文件包括：

```text
user_balance_table.zip
user_profile_table.csv
mfd_day_share_interest.csv
mfd_bank_shibor.csv
comp_predict_table.csv
```

文件名后面带 `(1)`、`(2)` 等编号也可以，程序会根据关键词自动识别。原始比赛数据体积较大，本仓库不上传数据文件。

## 6. 运行方式

完成验证并生成提交文件：

```bash
python run.py --data-dir data --output-dir output --mode all
```

只生成提交文件：

```bash
python run.py --data-dir data --output-dir output --mode predict
```

只进行滚动验证：

```bash
python run.py --data-dir data --output-dir output --mode validate
```

## 7. 实验结果

我的几次线上提交结果如下：

| 版本 | 线上分数 | 说明 |
| --- | ---: | --- |
| v1 | 128.6225 | 稳健基线版本 |
| v2 | 110.8297 | 节假日和赎回修正过大，效果下降 |
| v3 | 122.9599 | 小幅星期校准，但仍低于 v1 |
| v4 | **131.0938** | 保守反向修正，当前最好成绩 |

最终采用第四版作为课程项目结果。

## 8. 总结

本次实验说明，时间序列比赛不能只看本地平均误差，还要控制单日预测的波动。申购和赎回都存在明显的星期周期，但对节假日和特殊日期的修正不能过于激进。第四版采用保守融合和小幅校准后，线上成绩相对第一版有所提升。
