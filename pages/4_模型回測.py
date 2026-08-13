import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ui import (  # noqa: E402
    load_global_css,
    render_footer,
    render_sidebar_info,
)

RESULT_DIR = PROJECT_ROOT / "results"


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="模型回測",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_global_css()
render_sidebar_info()


# ==================================================
# 資料載入函式
# ==================================================

@st.cache_data(ttl=300)
def load_csv_file(file_name):
    """載入 results 資料夾內的 CSV。"""

    file_path = RESULT_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_first_available_csv(candidate_names):
    """從候選檔名中載入第一個存在且有資料的 CSV。"""

    for file_name in candidate_names:
        dataframe = load_csv_file(file_name)
        if not dataframe.empty:
            return dataframe, file_name

    return pd.DataFrame(), None


@st.cache_data(ttl=300)
def load_backtest_data():
    """載入策略、基準、KPI、Fold 與分類績效。"""

    top5_df = load_csv_file("top5_portfolio_backtest.csv")
    benchmark_df = load_csv_file("equal_weight_benchmark.csv")
    kpi_df = load_csv_file("portfolio_kpis.csv")

    fold_df, fold_file_name = load_first_available_csv(
        [
            "fold_backtest_performance.csv",
            "fold_performance.csv",
            "fold_backtest_results.csv",
            "timeseries_fold_results.csv",
            "backtest_fold_summary.csv",
        ]
    )

    cv_df, cv_file_name = load_first_available_csv(
        [
            "cv_metrics.csv",
            "cv_results.csv",
            "xgb_cv_metrics.csv",
            "model_cv_metrics.csv",
            "timeseries_cv_results.csv",
        ]
    )

    return (
        top5_df,
        benchmark_df,
        kpi_df,
        fold_df,
        cv_df,
        fold_file_name,
        cv_file_name,
    )


# ==================================================
# 資料整理與格式化函式
# ==================================================

def to_numeric_columns(dataframe, columns):
    output_df = dataframe.copy()

    for column in columns:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(
                output_df[column],
                errors="coerce",
            )

    return output_df


def prepare_date_column(dataframe):
    output_df = dataframe.copy()

    possible_date_columns = [
        "date",
        "Date",
        "rebalance_date",
        "PredictionDate",
        "預測日期",
        "日期",
    ]

    date_column = next(
        (
            candidate
            for candidate in possible_date_columns
            if candidate in output_df.columns
        ),
        None,
    )

    if date_column is None:
        return output_df

    if date_column != "date":
        output_df = output_df.rename(columns={date_column: "date"})

    output_df["date"] = pd.to_datetime(
        output_df["date"],
        errors="coerce",
    )

    return (
        output_df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_kpi_value(kpi_row, column_name, default_value=np.nan):
    if kpi_row is None or column_name not in kpi_row.index:
        return default_value

    value = pd.to_numeric(kpi_row[column_name], errors="coerce")
    return default_value if pd.isna(value) else float(value)


def find_column(dataframe, candidates):
    return next(
        (candidate for candidate in candidates if candidate in dataframe.columns),
        None,
    )


def format_percentage(value, decimal_places=2):
    if pd.isna(value):
        return "資料不足"
    return f"{value:.{decimal_places}%}"


def format_number(value, decimal_places=3):
    if pd.isna(value):
        return "資料不足"
    return f"{value:.{decimal_places}f}"


def format_integer(value):
    if pd.isna(value):
        return "資料不足"
    return f"{int(value):,}"


def add_drawdown_column(dataframe):
    output_df = dataframe.copy()

    if "Net_Equity" not in output_df.columns:
        output_df["Drawdown"] = np.nan
        return output_df

    output_df["Net_Equity"] = pd.to_numeric(
        output_df["Net_Equity"],
        errors="coerce",
    )

    running_max = output_df["Net_Equity"].cummax()
    output_df["Drawdown"] = output_df["Net_Equity"] / running_max - 1
    return output_df


# ==================================================
# HTML 卡片函式
# ==================================================

def create_setting_card(label, value, accent_color):
    return f"""
    <div style="
        width:100%; height:145px; min-height:145px;
        box-sizing:border-box; display:flex; flex-direction:column;
        justify-content:space-between; padding:20px; background:#ffffff;
        border:1px solid #dce5ef; border-left:5px solid {accent_color};
        border-radius:14px;
        box-shadow:0 4px 14px rgba(30,64,175,0.07);
    ">
        <div style="color:#5f6f85;font-size:0.92rem;font-weight:700;">
            {label}
        </div>
        <div style="color:#172033;font-size:1.55rem;font-weight:850;line-height:1.25;">
            {value}
        </div>
    </div>
    """


def create_kpi_card(
    label,
    value,
    note,
    accent_color,
    value_color="#172033",
):
    return f"""
    <div style="
        width:100%; height:195px; min-height:195px;
        box-sizing:border-box; display:flex; flex-direction:column;
        justify-content:space-between; padding:22px;
        background:linear-gradient(145deg,#ffffff 0%,#f8fbff 100%);
        border:1px solid #dce5ef; border-top:4px solid {accent_color};
        border-radius:16px;
        box-shadow:0 5px 18px rgba(30,64,175,0.09);
    ">
        <div style="min-height:28px;color:#5f6f85;font-size:0.97rem;font-weight:700;">
            {label}
        </div>
        <div style="
            min-height:64px;display:flex;align-items:center;color:{value_color};
            font-size:2rem;font-weight:850;line-height:1.15;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        ">
            {value}
        </div>
        <div style="min-height:32px;color:#66768b;font-size:0.84rem;font-weight:600;line-height:1.45;">
            {note}
        </div>
    </div>
    """


# ==================================================
# 載入並檢查回測資料
# ==================================================

(
    top5_df,
    benchmark_df,
    kpi_df,
    fold_df,
    cv_df,
    fold_file_name,
    cv_file_name,
) = load_backtest_data()

missing_files = []

if top5_df.empty:
    missing_files.append("top5_portfolio_backtest.csv")
if benchmark_df.empty:
    missing_files.append("equal_weight_benchmark.csv")
if kpi_df.empty:
    missing_files.append("portfolio_kpis.csv")

if missing_files:
    st.error(f"缺少必要回測結果檔案：{missing_files}")
    st.info("請確認回測程式已產生必要 CSV，並放在 results 資料夾。")
    st.stop()

numeric_kpi_columns = [
    "回測期數",
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

kpi_df = to_numeric_columns(kpi_df, numeric_kpi_columns)
top5_df = prepare_date_column(top5_df)
benchmark_df = prepare_date_column(benchmark_df)

for dataframe in (top5_df, benchmark_df):
    if "Net_Equity" in dataframe.columns:
        dataframe["Net_Equity"] = pd.to_numeric(
            dataframe["Net_Equity"],
            errors="coerce",
        )

if "策略" not in kpi_df.columns:
    st.error("portfolio_kpis.csv 缺少『策略』欄位。")
    st.stop()

top5_kpi_df = kpi_df[kpi_df["策略"] == "AI Top 5"].copy()
benchmark_kpi_df = kpi_df[kpi_df["策略"] != "AI Top 5"].copy()

if top5_kpi_df.empty:
    st.error("portfolio_kpis.csv 中找不到 AI Top 5 策略。")
    st.stop()

top5_kpi = top5_kpi_df.iloc[0]
benchmark_kpi = benchmark_kpi_df.iloc[0] if not benchmark_kpi_df.empty else None

top5_cumulative_return = get_kpi_value(top5_kpi, "累積報酬")
top5_annual_return = get_kpi_value(top5_kpi, "年化報酬")
top5_sharpe = get_kpi_value(top5_kpi, "Sharpe_Ratio")
top5_max_drawdown = get_kpi_value(top5_kpi, "最大回撤")
top5_win_rate = get_kpi_value(top5_kpi, "每期勝率")
top5_average_return = get_kpi_value(top5_kpi, "平均每期成本後報酬")
top5_median_return = get_kpi_value(top5_kpi, "每期報酬中位數")
top5_period_count = get_kpi_value(top5_kpi, "回測期數")

benchmark_cumulative_return = get_kpi_value(benchmark_kpi, "累積報酬")
benchmark_annual_return = get_kpi_value(benchmark_kpi, "年化報酬")
benchmark_sharpe = get_kpi_value(benchmark_kpi, "Sharpe_Ratio")
benchmark_max_drawdown = get_kpi_value(benchmark_kpi, "最大回撤")


# ==================================================
# 頁面 Hero
# ==================================================

st.html(
    """
    <div class="ai-hero">
        <div class="ai-hero-title">AI Top 5 量化回測</div>
        <div class="ai-hero-subtitle">
            使用 TimeSeriesSplit 產生 OOF 樣本外預測，
            每 5 個交易日選取 AI 分數最高的 5 檔股票，
            採等權配置並扣除完整交易成本，
            與同期可用股票的等權基準比較。
        </div>
        <div style="margin-top:18px;">
            <span class="ai-badge ai-badge-blue">TimeSeriesSplit</span>
            <span class="ai-badge ai-badge-green">OOF 樣本外預測</span>
            <span class="ai-badge ai-badge-purple">AI Top 5</span>
            <span class="ai-badge ai-badge-orange">完整成本 0.60%</span>
        </div>
    </div>
    """
)


# ==================================================
# 回測設定
# ==================================================

st.subheader("回測設定")

setting_col1, setting_col2, setting_col3, setting_col4 = st.columns(
    4,
    gap="medium",
)

with setting_col1:
    st.html(create_setting_card("選股範圍", "85 檔股票", "#2563eb"))

with setting_col2:
    st.html(create_setting_card("持股數量", "AI Top 5", "#6d28d9"))

with setting_col3:
    st.html(create_setting_card("重新平衡週期", "每 5 個交易日", "#0891b2"))

with setting_col4:
    st.html(create_setting_card("完整交易成本", "0.60%", "#d97706"))

st.caption(
    "五檔股票採等權配置。回測使用 TimeSeriesSplit 產生的 "
    "OOF 樣本外預測，以降低資料洩漏與樣本內績效高估風險。"
)

st.divider()


# ==================================================
# 重要 KPI
# ==================================================

st.subheader("AI Top 5 績效摘要")

cumulative_excess = (
    top5_cumulative_return - benchmark_cumulative_return
    if pd.notna(top5_cumulative_return)
    and pd.notna(benchmark_cumulative_return)
    else np.nan
)

annual_excess = (
    top5_annual_return - benchmark_annual_return
    if pd.notna(top5_annual_return)
    and pd.notna(benchmark_annual_return)
    else np.nan
)

sharpe_excess = (
    top5_sharpe - benchmark_sharpe
    if pd.notna(top5_sharpe) and pd.notna(benchmark_sharpe)
    else np.nan
)

drawdown_difference = (
    top5_max_drawdown - benchmark_max_drawdown
    if pd.notna(top5_max_drawdown)
    and pd.notna(benchmark_max_drawdown)
    else np.nan
)

first_col1, first_col2, first_col3, first_col4 = st.columns(
    4,
    gap="medium",
)

with first_col1:
    cumulative_note = (
        f"較等權基準 {cumulative_excess:+.2%}"
        if pd.notna(cumulative_excess)
        else "等權基準資料不足"
    )
    st.html(
        create_kpi_card(
            "累積報酬",
            format_percentage(top5_cumulative_return),
            cumulative_note,
            "#10b981",
            "#047857",
        )
    )

with first_col2:
    annual_note = (
        f"較等權基準 {annual_excess:+.2%}"
        if pd.notna(annual_excess)
        else "等權基準資料不足"
    )
    st.html(
        create_kpi_card(
            "年化報酬",
            format_percentage(top5_annual_return),
            annual_note,
            "#2563eb",
            "#1d4ed8",
        )
    )

with first_col3:
    sharpe_note = (
        f"較等權基準 {sharpe_excess:+.3f}"
        if pd.notna(sharpe_excess)
        else "等權基準資料不足"
    )
    st.html(
        create_kpi_card(
            "Sharpe Ratio",
            format_number(top5_sharpe, 3),
            sharpe_note,
            "#0891b2",
        )
    )

with first_col4:
    drawdown_note = (
        f"與等權基準差異 {drawdown_difference:+.2%}"
        if pd.notna(drawdown_difference)
        else "等權基準資料不足"
    )
    st.html(
        create_kpi_card(
            "最大回撤",
            format_percentage(top5_max_drawdown),
            drawdown_note,
            "#dc2626",
            "#b91c1c",
        )
    )

second_col1, second_col2, second_col3, second_col4 = st.columns(
    4,
    gap="medium",
)

with second_col1:
    st.html(
        create_kpi_card(
            "每期勝率",
            format_percentage(top5_win_rate),
            "成本後淨報酬大於 0 的回測期數比例",
            "#6d28d9",
        )
    )

with second_col2:
    st.html(
        create_kpi_card(
            "平均每期淨報酬",
            format_percentage(top5_average_return),
            "每個五交易日週期扣除成本後的平均",
            "#10b981",
        )
    )

with second_col3:
    st.html(
        create_kpi_card(
            "每期報酬中位數",
            format_percentage(top5_median_return),
            "降低少數極端報酬對平均值的影響",
            "#f59e0b",
        )
    )

with second_col4:
    st.html(
        create_kpi_card(
            "回測期數",
            format_integer(top5_period_count),
            "每期為 5 個交易日",
            "#64748b",
        )
    )

st.warning(
    "AI Top 5 的歷史累積報酬雖高於等權基準，但最大回撤超過 50%，"
    "代表集中持有五檔股票仍具有顯著波動與下行風險。"
)

st.divider()


# ==================================================
# 累積資產曲線
# ==================================================

st.subheader("累積資產曲線")

equity_figure = go.Figure()

if "date" in top5_df.columns and "Net_Equity" in top5_df.columns:
    equity_figure.add_trace(
        go.Scatter(
            x=top5_df["date"],
            y=top5_df["Net_Equity"],
            name="AI Top 5",
            mode="lines",
            line={"color": "#dc2626", "width": 3},
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "AI Top 5 資產：%{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

if "date" in benchmark_df.columns and "Net_Equity" in benchmark_df.columns:
    equity_figure.add_trace(
        go.Scatter(
            x=benchmark_df["date"],
            y=benchmark_df["Net_Equity"],
            name="等權基準",
            mode="lines",
            line={"color": "#2563eb", "width": 2.5},
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "等權基準資產：%{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

equity_figure.update_layout(
    height=650,
    margin={"l": 45, "r": 35, "t": 75, "b": 90},
    title={
        "text": "AI Top 5 與等權基準累積資產",
        "x": 0.02,
        "font": {"size": 20, "color": "#172033"},
    },
    hovermode="x unified",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font={
        "family": "Microsoft JhengHei, Noto Sans TC, Arial",
        "color": "#172033",
        "size": 14,
    },
    xaxis={
        "automargin": True,
        "title": {"text": "日期", "standoff": 20},
        "showgrid": False,
    },
    yaxis={
        "automargin": True,
        "title": {"text": "累積資產倍數", "standoff": 18},
        "showgrid": True,
        "gridcolor": "#edf2f7",
        "zeroline": False,
    },
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
)

st.plotly_chart(equity_figure, use_container_width=True)

st.caption(
    "資產曲線初始值為 1。例如資產值為 3.60，代表累積報酬約為 260%。"
)

st.divider()


# ==================================================
# 回撤曲線
# ==================================================

st.subheader("策略回撤")

top5_drawdown_df = add_drawdown_column(top5_df)
benchmark_drawdown_df = add_drawdown_column(benchmark_df)

drawdown_figure = go.Figure()

if "date" in top5_drawdown_df.columns and "Drawdown" in top5_drawdown_df.columns:
    drawdown_figure.add_trace(
        go.Scatter(
            x=top5_drawdown_df["date"],
            y=top5_drawdown_df["Drawdown"],
            name="AI Top 5",
            mode="lines",
            fill="tozeroy",
            line={"color": "#dc2626", "width": 2.5},
            fillcolor="rgba(220,38,38,0.12)",
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "AI Top 5 回撤：%{y:.2%}"
                "<extra></extra>"
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
            name="等權基準",
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "等權基準回撤：%{y:.2%}"
                "<extra></extra>"
            ),
        )
    )

drawdown_figure.update_layout(
    height=570,
    margin={"l": 45, "r": 35, "t": 75, "b": 90},
    title={
        "text": "AI Top 5 與等權基準回撤",
        "x": 0.02,
        "font": {"size": 20, "color": "#172033"},
    },
    hovermode="x unified",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font={
        "family": "Microsoft JhengHei, Noto Sans TC, Arial",
        "color": "#172033",
        "size": 14,
    },
    xaxis={
        "automargin": True,
        "title": {"text": "日期", "standoff": 20},
        "showgrid": False,
    },
    yaxis={
        "automargin": True,
        "title": {"text": "回撤幅度", "standoff": 18},
        "tickformat": ".0%",
        "showgrid": True,
        "gridcolor": "#edf2f7",
        "zeroline": True,
        "zerolinecolor": "#94a3b8",
    },
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
)

st.plotly_chart(drawdown_figure, use_container_width=True)

st.divider()


# ==================================================
# 策略與基準完整比較
# ==================================================

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

kpi_column_config = {
    "策略": st.column_config.TextColumn("策略"),
    "回測期數": st.column_config.NumberColumn("回測期數", format="%d"),
}

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
        kpi_column_config[percentage_column] = st.column_config.NumberColumn(
            percentage_column,
            format="%.2f%%",
        )

if "Sharpe_Ratio" in kpi_display_columns:
    kpi_column_config["Sharpe_Ratio"] = st.column_config.NumberColumn(
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


# ==================================================
# TimeSeriesSplit 分期回測
# ==================================================

st.subheader("TimeSeriesSplit 分期回測")

if fold_df.empty:
    st.info(
        "目前未找到 Fold 分期回測檔案。其他主要回測結果仍可正常顯示。"
    )
else:
    fold_column = find_column(fold_df, ["Fold", "fold", "折數", "驗證折數"])
    strategy_return_column = find_column(
        fold_df,
        ["策略報酬", "AI_Top5_Return", "Top5_Return", "Portfolio_Return"],
    )
    benchmark_return_column = find_column(
        fold_df,
        ["基準報酬", "Benchmark_Return", "Equal_Weight_Return"],
    )

    for column in [strategy_return_column, benchmark_return_column]:
        if column is not None:
            fold_df[column] = pd.to_numeric(fold_df[column], errors="coerce")

    if fold_column is not None:
        fold_df[fold_column] = fold_df[fold_column].astype(str)

    if fold_column is not None and strategy_return_column is not None:
        fold_figure = go.Figure()

        fold_figure.add_trace(
            go.Bar(
                x=fold_df[fold_column],
                y=fold_df[strategy_return_column],
                name="AI Top 5",
                marker_color="#dc2626",
                hovertemplate=(
                    "Fold：%{x}<br>AI Top 5 報酬：%{y:.2%}<extra></extra>"
                ),
            )
        )

        if benchmark_return_column is not None:
            fold_figure.add_trace(
                go.Bar(
                    x=fold_df[fold_column],
                    y=fold_df[benchmark_return_column],
                    name="等權基準",
                    marker_color="#2563eb",
                    hovertemplate=(
                        "Fold：%{x}<br>等權基準報酬：%{y:.2%}<extra></extra>"
                    ),
                )
            )

        fold_figure.update_layout(
            height=520,
            barmode="group",
            margin={"l": 45, "r": 35, "t": 65, "b": 85},
            title={"text": "各 Fold 策略與基準報酬", "x": 0.02},
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font={
                "family": "Microsoft JhengHei, Noto Sans TC, Arial",
                "color": "#172033",
                "size": 14,
            },
            xaxis={"title": {"text": "Fold", "standoff": 18}},
            yaxis={
                "title": {"text": "報酬率", "standoff": 18},
                "tickformat": ".0%",
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": True,
                "zerolinecolor": "#94a3b8",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )

        st.plotly_chart(fold_figure, use_container_width=True)

    st.caption(f"Fold 分期資料來源：{fold_file_name}")
    st.dataframe(fold_df, use_container_width=True, hide_index=True)

st.divider()


# ==================================================
# XGBoost 五折分類績效
# ==================================================

st.subheader("XGBoost 五折分類績效")

if cv_df.empty:
    st.info(
        "目前未找到五折分類績效檔案。不影響 AI Top 5 回測主要結果。"
    )
else:
    auc_column = find_column(
        cv_df,
        ["ROC_AUC", "ROC-AUC", "AUC", "roc_auc", "Valid_ROC_AUC"],
    )
    cv_fold_column = find_column(cv_df, ["Fold", "fold", "折數"])

    if auc_column is not None:
        cv_df[auc_column] = pd.to_numeric(cv_df[auc_column], errors="coerce")

    if cv_fold_column is not None:
        cv_df[cv_fold_column] = cv_df[cv_fold_column].astype(str)

    if auc_column is not None and cv_fold_column is not None:
        cv_figure = go.Figure()

        cv_figure.add_trace(
            go.Bar(
                x=cv_df[cv_fold_column],
                y=cv_df[auc_column],
                name="ROC-AUC",
                marker_color="#6d28d9",
                text=cv_df[auc_column],
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Fold：%{x}<br>ROC-AUC：%{y:.4f}<extra></extra>"
                ),
            )
        )

        cv_figure.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="隨機分類基準 0.5",
            annotation_position="bottom right",
        )

        valid_auc_values = cv_df[auc_column].dropna()
        auc_upper_limit = 0.60

        if not valid_auc_values.empty:
            auc_upper_limit = max(0.60, float(valid_auc_values.max()) + 0.03)

        cv_figure.update_layout(
            height=500,
            margin={"l": 45, "r": 45, "t": 65, "b": 85},
            title={"text": "各 Fold ROC-AUC", "x": 0.02},
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font={
                "family": "Microsoft JhengHei, Noto Sans TC, Arial",
                "color": "#172033",
                "size": 14,
            },
            xaxis={"title": {"text": "Fold", "standoff": 18}},
            yaxis={
                "title": {"text": "ROC-AUC", "standoff": 18},
                "range": [0.48, auc_upper_limit],
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
            },
            showlegend=False,
        )

        st.plotly_chart(cv_figure, use_container_width=True)

    st.caption(f"五折分類績效資料來源：{cv_file_name}")
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

st.divider()


# ==================================================
# 回測結果解讀
# ==================================================

st.html(
    """
    <div class="ai-card">
        <div class="ai-card-title">回測結果如何解讀？</div>
        <div class="ai-card-text">
            <strong>OOF 樣本外回測：</strong>
            每一筆回測預測，都由未使用該驗證期間資料訓練的模型產生。
            <br><br>
            <strong>AI Top 5：</strong>
            每 5 個交易日依 AI 分數選取前 5 名股票，五檔採等權配置。
            <br><br>
            <strong>平均每期淨報酬：</strong>
            每個五交易日回測期，在扣除完整交易成本後的算術平均報酬。
            <br><br>
            <strong>累積報酬：</strong>
            將各期淨報酬依時間連乘，反映複利累積資產。
            <br><br>
            <strong>最大回撤：</strong>
            資產曲線從歷史高點到後續低點的最大跌幅，數值越負代表下行風險越高。
        </div>
    </div>
    """
)

st.warning(
    "歷史 OOF 回測結果不代表未來實際績效。"
    "AI Top 5 策略具有集中持股與高度回撤風險，"
    "本頁僅供課程研究、模型驗證與歷史策略分析，"
    "不構成任何投資建議。"
)

render_footer()
