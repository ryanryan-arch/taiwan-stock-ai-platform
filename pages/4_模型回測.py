from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results"


st.set_page_config(
    page_title="模型回測",
    page_icon="🧪",
    layout="wide",
)


# --------------------------------------------------
# 資料載入函式
# --------------------------------------------------

@st.cache_data(ttl=300)
def load_csv_file(file_name):

    file_path = RESULT_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


@st.cache_data(ttl=300)
def load_backtest_data():

    top5_df = load_csv_file(
        "top5_portfolio_backtest.csv"
    )

    benchmark_df = load_csv_file(
        "equal_weight_benchmark.csv"
    )

    kpi_df = load_csv_file(
        "portfolio_kpis.csv"
    )

    fold_df = load_csv_file(
        "fold_backtest_results.csv"
    )

    cv_df = load_csv_file(
        "timeseries_cv_results.csv"
    )

    for dataframe in [
        top5_df,
        benchmark_df,
        fold_df,
        cv_df,
    ]:
        for date_column in [
            "date",
            "Start_Date",
            "End_Date",
            "ValidStart",
            "ValidEnd",
            "TrainStart",
            "TrainEnd",
        ]:
            if (
                not dataframe.empty
                and date_column in dataframe.columns
            ):
                dataframe[date_column] = pd.to_datetime(
                    dataframe[date_column],
                    errors="coerce",
                )

    return (
        top5_df,
        benchmark_df,
        kpi_df,
        fold_df,
        cv_df,
    )


(
    top5_df,
    benchmark_df,
    kpi_df,
    fold_df,
    cv_df,
) = load_backtest_data()


# --------------------------------------------------
# 頁面標題
# --------------------------------------------------

st.title("AI Top 5 策略回測")

st.caption(
    "使用 TimeSeriesSplit 產生樣本外預測，"
    "每 5 個交易日重新選取 AI 分數最高的 5 檔股票，"
    "並與同期可用股票等權基準進行比較。"
)


# --------------------------------------------------
# 檔案完整性檢查
# --------------------------------------------------

missing_files = []

if top5_df.empty:
    missing_files.append(
        "top5_portfolio_backtest.csv"
    )

if benchmark_df.empty:
    missing_files.append(
        "equal_weight_benchmark.csv"
    )

if kpi_df.empty:
    missing_files.append(
        "portfolio_kpis.csv"
    )


if missing_files:

    st.error(
        "缺少以下回測檔案："
        + "、".join(missing_files)
    )

    st.info(
        "請確認 Colab 的 results 資料夾已完整複製到 "
        "VS Code 專案的 results 資料夾。"
    )

    st.stop()


# --------------------------------------------------
# 數值格式整理
# --------------------------------------------------

numeric_kpi_columns = [
    "累積報酬",
    "年化報酬",
    "年化波動率",
    "最大回撤",
    "Sharpe_Ratio",
    "每期勝率",
    "每期虧損率",
    "平均每期成本後報酬",
    "每期報酬中位數",
    "最佳一期",
    "最差一期",
]


for column in numeric_kpi_columns:

    if column in kpi_df.columns:
        kpi_df[column] = pd.to_numeric(
            kpi_df[column],
            errors="coerce",
        )


# 取得策略與基準資料
top5_kpi = kpi_df[
    kpi_df["策略"] == "AI Top 5"
]

benchmark_kpi = kpi_df[
    kpi_df["策略"] != "AI Top 5"
]


if top5_kpi.empty:
    st.error(
        "portfolio_kpis.csv 中找不到 AI Top 5。"
    )
    st.stop()


top5_kpi = top5_kpi.iloc[0]


if not benchmark_kpi.empty:
    benchmark_kpi = benchmark_kpi.iloc[0]
else:
    benchmark_kpi = None


# --------------------------------------------------
# 回測設定
# --------------------------------------------------

st.subheader("回測設定")


setting_col1, setting_col2, setting_col3, setting_col4 = (
    st.columns(4)
)


setting_col1.metric(
    "選股範圍",
    "85 檔",
)

setting_col2.metric(
    "持股數量",
    "Top 5",
)

setting_col3.metric(
    "重新平衡週期",
    "每 5 個交易日",
)

setting_col4.metric(
    "完整交易成本",
    "0.60%",
)


st.caption(
    "五檔股票採等權配置，回測使用 TimeSeriesSplit "
    "產生的 OOF 樣本外預測，降低資料洩漏風險。"
)


st.divider()


# --------------------------------------------------
# 重要 KPI
# --------------------------------------------------

st.subheader("AI Top 5 績效摘要")


kpi_col1, kpi_col2, kpi_col3, kpi_col4 = (
    st.columns(4)
)


kpi_col1.metric(
    "累積報酬",
    f'{top5_kpi["累積報酬"]:.2%}',
    (
        f'{top5_kpi["累積報酬"] - benchmark_kpi["累積報酬"]:+.2%}'
        if benchmark_kpi is not None
        else None
    ),
)


kpi_col2.metric(
    "年化報酬",
    f'{top5_kpi["年化報酬"]:.2%}',
    (
        f'{top5_kpi["年化報酬"] - benchmark_kpi["年化報酬"]:+.2%}'
        if benchmark_kpi is not None
        else None
    ),
)


kpi_col3.metric(
    "Sharpe Ratio",
    f'{top5_kpi["Sharpe_Ratio"]:.3f}',
    (
        f'{top5_kpi["Sharpe_Ratio"] - benchmark_kpi["Sharpe_Ratio"]:+.3f}'
        if benchmark_kpi is not None
        else None
    ),
)


kpi_col4.metric(
    "最大回撤",
    f'{top5_kpi["最大回撤"]:.2%}',
    (
        f'{top5_kpi["最大回撤"] - benchmark_kpi["最大回撤"]:+.2%}'
        if benchmark_kpi is not None
        else None
    ),
    delta_color="inverse",
)


second_col1, second_col2, second_col3, second_col4 = (
    st.columns(4)
)


second_col1.metric(
    "每期勝率",
    f'{top5_kpi["每期勝率"]:.2%}',
)


second_col2.metric(
    "平均每期淨報酬",
    f'{top5_kpi["平均每期成本後報酬"]:.2%}',
)


second_col3.metric(
    "報酬中位數",
    f'{top5_kpi["每期報酬中位數"]:.2%}',
)


second_col4.metric(
    "回測期數",
    int(top5_kpi["回測期數"]),
)


st.warning(
    "AI Top 5 的歷史報酬雖高於等權基準，"
    "但最大回撤超過 50%，代表策略仍具有高度波動與集中風險。"
)


st.divider()


# --------------------------------------------------
# 資產曲線
# --------------------------------------------------

st.subheader("累積資產曲線")


if "date" in top5_df.columns:
    top5_df["date"] = pd.to_datetime(
        top5_df["date"],
        errors="coerce",
    )

if "date" in benchmark_df.columns:
    benchmark_df["date"] = pd.to_datetime(
        benchmark_df["date"],
        errors="coerce",
    )


equity_figure = go.Figure()


if (
    "date" in top5_df.columns
    and "Net_Equity" in top5_df.columns
):

    equity_figure.add_trace(
        go.Scatter(
            x=top5_df["date"],
            y=top5_df["Net_Equity"],
            mode="lines",
            name="AI Top 5",
            line=dict(
                width=3,
                color="#E74C3C",
            ),
        )
    )


if (
    "date" in benchmark_df.columns
    and "Net_Equity" in benchmark_df.columns
):

    equity_figure.add_trace(
        go.Scatter(
            x=benchmark_df["date"],
            y=benchmark_df["Net_Equity"],
            mode="lines",
            name="等權基準",
            line=dict(
                width=3,
                color="#3498DB",
            ),
        )
    )


equity_figure.update_layout(
    height=600,
    hovermode="x unified",
    xaxis_title="日期",
    yaxis_title="累積資產倍數",
    legend_title="策略",
)


st.plotly_chart(
    equity_figure,
    use_container_width=True,
)


st.caption(
    "資產曲線初始值為 1。"
    "例如資產值 3.60 代表累積報酬約為 260%。"
)


st.divider()


# --------------------------------------------------
# 最大回撤曲線
# --------------------------------------------------

st.subheader("策略回撤")


def add_drawdown_column(dataframe):

    result = dataframe.copy()

    if "Net_Equity" not in result.columns:
        return result

    result["Running_Max"] = (
        result["Net_Equity"].cummax()
    )

    result["Drawdown"] = (
        result["Net_Equity"]
        / result["Running_Max"]
        - 1
    )

    return result


top5_drawdown_df = add_drawdown_column(
    top5_df
)

benchmark_drawdown_df = add_drawdown_column(
    benchmark_df
)


drawdown_figure = go.Figure()


if (
    "date" in top5_drawdown_df.columns
    and "Drawdown" in top5_drawdown_df.columns
):

    drawdown_figure.add_trace(
        go.Scatter(
            x=top5_drawdown_df["date"],
            y=top5_drawdown_df["Drawdown"],
            mode="lines",
            name="AI Top 5",
            fill="tozeroy",
            line=dict(
                color="#E74C3C",
            ),
        )
    )


if (
    "date" in benchmark_drawdown_df.columns
    and "Drawdown" in benchmark_drawdown_df.columns
):

    drawdown_figure.add_trace(
        go.Scatter(
            x=benchmark_drawdown_df["date"],
            y=benchmark_drawdown_df["Drawdown"],
            mode="lines",
            name="等權基準",
            line=dict(
                color="#3498DB",
            ),
        )
    )


drawdown_figure.update_layout(
    height=500,
    hovermode="x unified",
    xaxis_title="日期",
    yaxis_title="回撤幅度",
    yaxis_tickformat=".0%",
    legend_title="策略",
)


st.plotly_chart(
    drawdown_figure,
    use_container_width=True,
)


st.divider()


# --------------------------------------------------
# 策略與基準 KPI 比較
# --------------------------------------------------

st.subheader("策略與基準完整比較")


kpi_display_columns = [
    column
    for column in [
        "策略",
        "回測期數",
        "累積報酬",
        "年化報酬",
        "年化波動率",
        "最大回撤",
        "Sharpe_Ratio",
        "每期勝率",
        "平均每期成本後報酬",
        "每期報酬中位數",
        "最佳一期",
        "最差一期",
    ]
    if column in kpi_df.columns
]


kpi_column_config = {}


for percentage_column in [
    "累積報酬",
    "年化報酬",
    "年化波動率",
    "最大回撤",
    "每期勝率",
    "平均每期成本後報酬",
    "每期報酬中位數",
    "最佳一期",
    "最差一期",
]:

    if percentage_column in kpi_display_columns:
        kpi_column_config[
            percentage_column
        ] = st.column_config.NumberColumn(
            percentage_column,
            format="%.2f%%",
        )


if "Sharpe_Ratio" in kpi_display_columns:
    kpi_column_config[
        "Sharpe_Ratio"
    ] = st.column_config.NumberColumn(
        "Sharpe Ratio",
        format="%.3f",
    )


st.dataframe(
    kpi_df[kpi_display_columns],
    use_container_width=True,
    hide_index=True,
    column_config=kpi_column_config,
)


st.divider()


# --------------------------------------------------
# TimeSeriesSplit 分折績效
# --------------------------------------------------

st.subheader("TimeSeriesSplit 分期回測")


if fold_df.empty:

    st.info(
        "目前沒有 fold_backtest_results.csv，"
        "無法顯示分期策略績效。"
    )

else:

    fold_display_df = fold_df.copy()

    fold_display_columns = [
        column
        for column in [
            "Fold",
            "Start_Date",
            "End_Date",
            "Periods",
            "Top5_Total_Return",
            "Benchmark_Total_Return",
            "Average_Excess_Return",
            "Top5_Max_Drawdown",
            "Benchmark_Max_Drawdown",
            "Top5_Sharpe",
            "Benchmark_Sharpe",
        ]
        if column in fold_display_df.columns
    ]


    fold_column_config = {}


    for percentage_column in [
        "Top5_Total_Return",
        "Benchmark_Total_Return",
        "Average_Excess_Return",
        "Top5_Max_Drawdown",
        "Benchmark_Max_Drawdown",
    ]:

        if percentage_column in fold_display_columns:

            fold_column_config[
                percentage_column
            ] = st.column_config.NumberColumn(
                percentage_column,
                format="%.2f%%",
            )


    if "Start_Date" in fold_display_columns:
        fold_column_config[
            "Start_Date"
        ] = st.column_config.DateColumn(
            "開始日期",
            format="YYYY-MM-DD",
        )


    if "End_Date" in fold_display_columns:
        fold_column_config[
            "End_Date"
        ] = st.column_config.DateColumn(
            "結束日期",
            format="YYYY-MM-DD",
        )


    st.dataframe(
        fold_display_df[
            fold_display_columns
        ],
        use_container_width=True,
        hide_index=True,
        column_config=fold_column_config,
    )


    if "Average_Excess_Return" in fold_df.columns:

        positive_fold_count = int(
            (
                fold_df[
                    "Average_Excess_Return"
                ] > 0
            ).sum()
        )

        total_fold_count = len(fold_df)

        st.success(
            f"{positive_fold_count} / "
            f"{total_fold_count} 個 Fold "
            "的平均超額報酬為正。"
        )


st.divider()


# --------------------------------------------------
# 模型分類績效
# --------------------------------------------------

st.subheader("XGBoost 五折分類績效")


if cv_df.empty:

    st.info(
        "目前沒有 timeseries_cv_results.csv，"
        "無法顯示模型五折分類績效。"
    )

else:

    cv_display_columns = [
        column
        for column in [
            "Fold",
            "ValidStart",
            "ValidEnd",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ]
        if column in cv_df.columns
    ]


    st.dataframe(
        cv_df[cv_display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(
                "Accuracy",
                format="%.4f",
            ),
            "Precision": st.column_config.NumberColumn(
                "Precision",
                format="%.4f",
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                format="%.4f",
            ),
            "F1": st.column_config.NumberColumn(
                "F1",
                format="%.4f",
            ),
            "ROC_AUC": st.column_config.NumberColumn(
                "ROC-AUC",
                format="%.4f",
            ),
        },
    )


    if "ROC_AUC" in cv_df.columns:

        average_auc = cv_df["ROC_AUC"].mean()
        auc_std = cv_df["ROC_AUC"].std()

        auc_col1, auc_col2 = st.columns(2)

        auc_col1.metric(
            "五折平均 ROC-AUC",
            f"{average_auc:.4f}",
        )

        auc_col2.metric(
            "ROC-AUC 標準差",
            f"{auc_std:.4f}",
        )


st.divider()


# --------------------------------------------------
# 結果說明
# --------------------------------------------------

st.subheader("回測結果解讀")


st.markdown(
    """
- AI Top 5 的累積報酬與 Sharpe Ratio 高於等權基準，代表模型具有初步選股排序價值。
- 五個 TimeSeriesSplit 驗證期間的平均超額報酬皆為正，但部分期間的策略絕對報酬仍為負。
- 策略最大回撤超過 50%，顯示集中持有五檔股票仍有明顯下行風險。
- 回測已扣除每次完整買進與賣出的簡化交易成本 0.6%。
- 歷史回測結果不代表未來績效，模型分數也不代表保證上漲機率。
"""
)


st.warning(
    "本頁僅供課程研究、模型驗證與歷史策略分析，"
    "不構成任何投資建議。"
)