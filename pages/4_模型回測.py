import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ==================================================
# 專案路徑與頁面設定
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ui import load_global_css, render_footer, render_sidebar_info  # noqa: E402

RESULT_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="模型回測",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_global_css()
render_sidebar_info()


# ==================================================
# 常數與顯示名稱
# ==================================================

MODEL_NAME_MAP = {
    "Logistic Regression": "Logistic Regression（邏輯斯迴歸）",
    "Random Forest": "Random Forest（隨機森林）",
    "XGBoost": "XGBoost",
}

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "Random Forest": "#16a34a",
    "XGBoost": "#dc2626",
}

PLOT_FONT = "Microsoft JhengHei, Noto Sans TC, Arial"


# ==================================================
# 資料載入
# ==================================================

@st.cache_data(ttl=300)
def load_csv(file_name):
    path = RESULT_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_first_available(file_names):
    for file_name in file_names:
        dataframe = load_csv(file_name)
        if not dataframe.empty:
            return dataframe, file_name
    return pd.DataFrame(), None


def prepare_date(dataframe):
    output = dataframe.copy()
    date_column = next(
        (column for column in ["date", "Date", "日期", "rebalance_date"] if column in output.columns),
        None,
    )
    if date_column is None:
        return output
    if date_column != "date":
        output = output.rename(columns={date_column: "date"})
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    return output.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def numeric(dataframe, columns):
    output = dataframe.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_drawdown(dataframe):
    output = dataframe.copy()
    if "Net_Equity" not in output.columns:
        output["Drawdown"] = np.nan
        return output
    output["Net_Equity"] = pd.to_numeric(output["Net_Equity"], errors="coerce")
    running_max = output["Net_Equity"].cummax()
    output["Drawdown"] = output["Net_Equity"] / running_max - 1
    return output


def row_value(row, column, default=np.nan):
    if row is None or column not in row.index:
        return default
    value = pd.to_numeric(row[column], errors="coerce")
    return default if pd.isna(value) else float(value)


@st.cache_data(ttl=900, show_spinner=False)
def load_0050_buy_and_hold(start_date, end_date):
    """下載 0050 調整後價格並建立同期買進持有資產曲線。"""

    empty_result = pd.DataFrame()

    if pd.isna(start_date) or pd.isna(end_date):
        return empty_result, "回測起訖日期無效"

    start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_text = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    error_messages = []
    data = pd.DataFrame()

    try:
        data = yf.download(
            "0050.TW",
            start=start_text,
            end=end_text,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=30,
        )
    except Exception as error:
        error_messages.append(f"download: {error}")

    if data is None or data.empty:
        try:
            data = yf.Ticker("0050.TW").history(
                start=start_text,
                end=end_text,
                auto_adjust=True,
                timeout=30,
            )
        except Exception as error:
            error_messages.append(f"history: {error}")

    if data is None or data.empty:
        detail = "；".join(error_messages) if error_messages else "Yahoo Finance 回傳空資料"
        return empty_result, detail

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        return empty_result, "0050 歷史資料缺少 Close 欄位"

    output = data.reset_index()
    date_col = "Date" if "Date" in output.columns else output.columns[0]
    output = output.rename(
        columns={date_col: "date", "Close": "Adjusted_Close"}
    )
    output = output[["date", "Adjusted_Close"]].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    if output["date"].dt.tz is not None:
        output["date"] = output["date"].dt.tz_localize(None)
    output["Adjusted_Close"] = pd.to_numeric(
        output["Adjusted_Close"], errors="coerce"
    )
    output = (
        output
        .dropna(subset=["date", "Adjusted_Close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if output.empty:
        return empty_result, "0050 整理後沒有有效價格"

    initial_price = output.iloc[0]["Adjusted_Close"]
    if initial_price <= 0:
        return empty_result, "0050 起始調整後價格無效"

    output["Net_Equity"] = output["Adjusted_Close"] / initial_price
    return output, ""


# 原有 XGBoost 策略回測資料
top5_df = prepare_date(load_csv("top5_portfolio_backtest.csv"))
benchmark_df = prepare_date(load_csv("equal_weight_benchmark.csv"))
if "date" in top5_df.columns and not top5_df.empty:
    benchmark_0050_df, benchmark_0050_error = load_0050_buy_and_hold(
        top5_df["date"].min(),
        top5_df["date"].max(),
    )
else:
    benchmark_0050_df = pd.DataFrame()
    benchmark_0050_error = "AI Top 5 回測日期無效"
kpi_df = load_csv("portfolio_kpis.csv")
fold_df, fold_file_name = load_first_available([
    "fold_backtest_results.csv",
    "fold_backtest_performance.csv",
    "fold_performance.csv",
])

# 新增的多模型比較資料
comparison_df = load_csv("model_comparison_metrics.csv")
model_fold_df = load_csv("model_fold_metrics.csv")
portfolio_df = load_csv("model_portfolio_comparison.csv")
equity_df = prepare_date(load_csv("model_equity_curves.csv"))

kpi_numeric_columns = [
    "回測期數", "累積報酬", "年化報酬", "年化波動率", "最大回撤",
    "Sharpe_Ratio", "每期勝率", "平均每期成本後報酬", "每期報酬中位數",
    "最佳一期", "最差一期",
]
kpi_df = numeric(kpi_df, kpi_numeric_columns)

comparison_df = numeric(comparison_df, [
    "Mean_ROC_AUC", "Std_ROC_AUC", "Overall_OOF_ROC_AUC", "Accuracy",
    "Precision", "Recall", "F1", "Mean_Training_Seconds", "Sample_Count",
])
model_fold_df = numeric(model_fold_df, ["Fold", "ROC_AUC", "Accuracy", "Precision", "Recall", "F1", "Seconds"])
portfolio_df = numeric(portfolio_df, [
    "Backtest_Periods", "Cumulative_Return", "Annual_Return", "Annual_Volatility",
    "Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Mean_Net_Return",
    "Median_Net_Return", "Best_Period", "Worst_Period",
])
equity_df = numeric(equity_df, ["Net_Equity", "Drawdown", "Net_Return"])


# ==================================================
# 顯示工具
# ==================================================

def fmt_pct(value, digits=2):
    return "資料不足" if pd.isna(value) else f"{value:.{digits}%}"


def fmt_num(value, digits=3):
    return "資料不足" if pd.isna(value) else f"{value:.{digits}f}"


def fmt_int(value):
    return "資料不足" if pd.isna(value) else f"{int(value):,}"


def card(label, value, note, color="#2563eb", value_color="#172033"):
    return f"""
    <div style="height:180px;box-sizing:border-box;padding:21px;display:flex;
         flex-direction:column;justify-content:space-between;background:linear-gradient(145deg,#fff,#f8fbff);
         border:1px solid #dce5ef;border-top:4px solid {color};border-radius:16px;
         box-shadow:0 5px 18px rgba(30,64,175,.09);">
      <div style="color:#5f6f85;font-size:.96rem;font-weight:700;">{label}</div>
      <div style="color:{value_color};font-size:1.9rem;font-weight:850;line-height:1.15;">{value}</div>
      <div style="color:#66768b;font-size:.84rem;font-weight:600;line-height:1.45;">{note}</div>
    </div>
    """


def setting_card(label, value, color):
    return f"""
    <div style="height:135px;box-sizing:border-box;padding:20px;display:flex;flex-direction:column;
         justify-content:space-between;background:#fff;border:1px solid #dce5ef;
         border-left:5px solid {color};border-radius:14px;box-shadow:0 4px 14px rgba(30,64,175,.07);">
      <div style="color:#5f6f85;font-size:.92rem;font-weight:700;">{label}</div>
      <div style="color:#172033;font-size:1.5rem;font-weight:850;">{value}</div>
    </div>
    """


def style_figure(figure, title, y_title, height=560, percent_axis=False):
    figure.update_layout(
        height=height,
        margin={"l": 45, "r": 35, "t": 75, "b": 85},
        title={"text": title, "x": 0.02, "font": {"size": 20, "color": "#172033"}},
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"family": PLOT_FONT, "color": "#172033", "size": 14},
        xaxis={"automargin": True, "showgrid": False},
        yaxis={
            "automargin": True,
            "title": {"text": y_title, "standoff": 18},
            "showgrid": True,
            "gridcolor": "#edf2f7",
            "zeroline": True,
            "zerolinecolor": "#94a3b8",
            **({"tickformat": ".0%"} if percent_axis else {}),
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return figure


# ==================================================
# 頁面 Hero 與分頁
# ==================================================

st.html("""
<div class="ai-hero">
  <div class="ai-hero-title">AI Top 5 量化回測與多模型比較</div>
  <div class="ai-hero-subtitle">
    使用相同的 85 檔股票、32 個特徵、五日 Target、TimeSeriesSplit 五折、
    gap = 5、OOF 預測與完整交易成本，比較正式 XGBoost 策略及其他模型。
  </div>
  <div style="margin-top:18px;">
    <span class="ai-badge ai-badge-blue">TimeSeriesSplit</span>
    <span class="ai-badge ai-badge-green">OOF 樣本外預測</span>
    <span class="ai-badge ai-badge-purple">三模型公平比較</span>
    <span class="ai-badge ai-badge-orange">完整成本 0.60%</span>
  </div>
</div>
""")

xgb_tab, comparison_tab = st.tabs([
    "📈 XGBoost 策略回測",
    "⚖️ 多模型公平比較",
])


# ==================================================
# 分頁一：XGBoost 策略回測
# ==================================================

with xgb_tab:
    required_original = {
        "top5_portfolio_backtest.csv": top5_df,
        "equal_weight_benchmark.csv": benchmark_df,
        "portfolio_kpis.csv": kpi_df,
    }
    missing_original = [name for name, dataframe in required_original.items() if dataframe.empty]

    if missing_original:
        st.error(f"缺少原有回測檔案：{missing_original}")
    elif "策略" not in kpi_df.columns:
        st.error("portfolio_kpis.csv 缺少『策略』欄位。")
    else:
        top5_rows = kpi_df[kpi_df["策略"] == "AI Top 5"]
        benchmark_rows = kpi_df[kpi_df["策略"] != "AI Top 5"]

        if top5_rows.empty:
            st.error("portfolio_kpis.csv 中找不到 AI Top 5。")
        else:
            top5_kpi = top5_rows.iloc[0]
            benchmark_kpi = benchmark_rows.iloc[0] if not benchmark_rows.empty else None

            st.subheader("回測設定")
            cols = st.columns(4, gap="medium")
            settings = [
                ("選股範圍", "85 檔股票", "#2563eb"),
                ("持股數量", "AI Top 5", "#6d28d9"),
                ("重新平衡週期", "每 5 個交易日", "#0891b2"),
                ("完整交易成本", "0.60%", "#d97706"),
            ]
            for column, (label, value, color) in zip(cols, settings):
                with column:
                    st.html(setting_card(label, value, color))

            st.caption("回測使用 TimeSeriesSplit 產生的 OOF 樣本外預測，五檔等權配置。")
            st.divider()

            values = {
                "累積報酬": row_value(top5_kpi, "累積報酬"),
                "年化報酬": row_value(top5_kpi, "年化報酬"),
                "Sharpe Ratio": row_value(top5_kpi, "Sharpe_Ratio"),
                "最大回撤": row_value(top5_kpi, "最大回撤"),
                "每期勝率": row_value(top5_kpi, "每期勝率"),
                "平均每期淨報酬": row_value(top5_kpi, "平均每期成本後報酬"),
                "每期報酬中位數": row_value(top5_kpi, "每期報酬中位數"),
                "回測期數": row_value(top5_kpi, "回測期數"),
            }

            st.subheader("AI Top 5 績效摘要")
            first = st.columns(4, gap="medium")
            second = st.columns(4, gap="medium")
            first_cards = [
                ("累積報酬", fmt_pct(values["累積報酬"]), "複利累積成果", "#10b981", "#047857"),
                ("年化報酬", fmt_pct(values["年化報酬"]), "依回測期間年化", "#2563eb", "#1d4ed8"),
                ("Sharpe Ratio", fmt_num(values["Sharpe Ratio"]), "風險調整後報酬", "#0891b2", "#172033"),
                ("最大回撤", fmt_pct(values["最大回撤"]), "相對歷史高點最大跌幅", "#dc2626", "#b91c1c"),
            ]
            second_cards = [
                ("每期勝率", fmt_pct(values["每期勝率"]), "淨報酬大於 0 的比例", "#6d28d9", "#172033"),
                ("平均每期淨報酬", fmt_pct(values["平均每期淨報酬"]), "每個五交易日週期", "#10b981", "#172033"),
                ("每期報酬中位數", fmt_pct(values["每期報酬中位數"]), "降低極端值影響", "#f59e0b", "#172033"),
                ("回測期數", fmt_int(values["回測期數"]), "每期為 5 個交易日", "#64748b", "#172033"),
            ]
            for column, item in zip(first, first_cards):
                with column:
                    st.html(card(*item))
            for column, item in zip(second, second_cards):
                with column:
                    st.html(card(*item))

            st.warning("歷史績效不代表未來結果，集中持有五檔股票仍具有顯著下行風險。")
            st.divider()

            st.subheader("累積資產曲線")
            figure = go.Figure()
            if {"date", "Net_Equity"}.issubset(top5_df.columns):
                figure.add_trace(go.Scatter(
                    x=top5_df["date"], y=top5_df["Net_Equity"], name="AI Top 5",
                    mode="lines", line={"color": "#dc2626", "width": 3},
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>累積資產：%{y:.4f}<extra></extra>",
                ))
            if {"date", "Net_Equity"}.issubset(benchmark_df.columns):
                figure.add_trace(go.Scatter(
                    x=benchmark_df["date"], y=benchmark_df["Net_Equity"], name="股票池等權基準",
                    mode="lines", line={"color": "#2563eb", "width": 2.5},
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>累積資產：%{y:.4f}<extra></extra>",
                ))
            if {"date", "Net_Equity"}.issubset(benchmark_0050_df.columns):
                figure.add_trace(go.Scatter(
                    x=benchmark_0050_df["date"], y=benchmark_0050_df["Net_Equity"],
                    name="0050 買進持有", mode="lines",
                    line={"color": "#64748b", "width": 2.5, "dash": "dot"},
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>0050 累積資產：%{y:.4f}<extra></extra>",
                ))
            st.plotly_chart(style_figure(figure, "AI Top 5、股票池等權基準與 0050 累積資產", "累積資產倍數", 630), use_container_width=True)
            st.info(
                "股票池等權基準將同期資料有效的股票平均配置，用於檢驗 AI 選股效果；"
                "0050 使用同期調整後價格建立買進持有曲線，作為大型權值股市場參考。"
                "0050 與 AI Top 5 的配置方式及交易頻率不同，因此不取代股票池等權基準。"
            )
            if benchmark_0050_df.empty:
                st.warning(
                    "目前無法取得 0050.TW 歷史價格，因此暫時不顯示 0050 曲線。"
                    f"詳細原因：{benchmark_0050_error}"
                )
            st.divider()

            st.subheader("策略回撤與下行風險")
            figure = go.Figure()
            top5_drawdown = add_drawdown(top5_df)
            benchmark_drawdown = add_drawdown(benchmark_df)
            if {"date", "Drawdown"}.issubset(top5_drawdown.columns):
                figure.add_trace(go.Scatter(
                    x=top5_drawdown["date"], y=top5_drawdown["Drawdown"], name="AI Top 5",
                    mode="lines", fill="tozeroy", line={"color": "#dc2626", "width": 2.5},
                    fillcolor="rgba(220,38,38,.12)",
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>回撤：%{y:.2%}<extra></extra>",
                ))
            if {"date", "Drawdown"}.issubset(benchmark_drawdown.columns):
                figure.add_trace(go.Scatter(
                    x=benchmark_drawdown["date"], y=benchmark_drawdown["Drawdown"], name="股票池等權基準",
                    mode="lines", line={"color": "#2563eb", "width": 2},
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>回撤：%{y:.2%}<extra></extra>",
                ))
            benchmark_0050_drawdown = add_drawdown(benchmark_0050_df)
            if {"date", "Drawdown"}.issubset(benchmark_0050_drawdown.columns):
                figure.add_trace(go.Scatter(
                    x=benchmark_0050_drawdown["date"], y=benchmark_0050_drawdown["Drawdown"],
                    name="0050 買進持有", mode="lines",
                    line={"color": "#64748b", "width": 2, "dash": "dot"},
                    hovertemplate="日期：%{x|%Y-%m-%d}<br>0050 回撤：%{y:.2%}<extra></extra>",
                ))
            st.plotly_chart(style_figure(figure, "AI Top 5、股票池等權基準與 0050 回撤曲線", "相對歷史高點跌幅", 560, True), use_container_width=True)
            st.divider()

            st.subheader("策略與基準完整比較")
            display_columns = [column for column in kpi_numeric_columns if column in kpi_df.columns]
            if "策略" in kpi_df.columns:
                display_columns.insert(0, "策略")
            st.dataframe(kpi_df[display_columns], use_container_width=True, hide_index=True)

            if not fold_df.empty:
                st.divider()
                st.subheader("TimeSeriesSplit 分期回測")
                st.caption(f"資料來源：{fold_file_name}")
                st.dataframe(fold_df, use_container_width=True, hide_index=True)


# ==================================================
# 分頁二：多模型公平比較
# ==================================================

with comparison_tab:
    required_comparison = {
        "model_comparison_metrics.csv": comparison_df,
        "model_fold_metrics.csv": model_fold_df,
        "model_portfolio_comparison.csv": portfolio_df,
        "model_equity_curves.csv": equity_df,
    }
    missing_comparison = [name for name, dataframe in required_comparison.items() if dataframe.empty]

    if missing_comparison:
        st.error(f"缺少多模型比較檔案：{missing_comparison}")
    else:
        st.info(
            "本比較使用相同的 85 檔股票、32 個特徵、五日 Target、TimeSeriesSplit 五折、"
            "gap = 5、OOF 預測、Top 5 等權配置與完整交易成本 0.60%。"
            "目前每日排行榜與 SHAP 解釋仍使用正式 XGBoost 模型。"
        )

        for dataframe in [comparison_df, model_fold_df, portfolio_df, equity_df]:
            if "Model" in dataframe.columns:
                dataframe["模型"] = dataframe["Model"].map(MODEL_NAME_MAP).fillna(dataframe["Model"])

        st.subheader("五折平均 ROC-AUC 與穩定性")
        roc_figure = go.Figure()
        for model_name in MODEL_NAME_MAP:
            data = model_fold_df[model_fold_df["Model"] == model_name].sort_values("Fold")
            roc_figure.add_trace(go.Scatter(
                x=data["Fold"], y=data["ROC_AUC"],
                name=MODEL_NAME_MAP[model_name], mode="lines+markers",
                line={"color": MODEL_COLORS[model_name], "width": 2.5}, marker={"size": 9},
                hovertemplate="Fold %{x}<br>ROC-AUC：%{y:.4f}<extra></extra>",
            ))
        roc_figure.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="隨機分類基準 0.5")
        roc_figure.update_xaxes(title="Fold", dtick=1)
        st.plotly_chart(style_figure(roc_figure, "三模型各 Fold ROC-AUC", "ROC-AUC", 520), use_container_width=True)

        classification_table = comparison_df[[
            "模型", "Mean_ROC_AUC", "Std_ROC_AUC", "Overall_OOF_ROC_AUC",
            "Accuracy", "Precision", "Recall", "F1", "Mean_Training_Seconds",
        ]].rename(columns={
            "Mean_ROC_AUC": "五折平均 ROC-AUC",
            "Std_ROC_AUC": "ROC-AUC 標準差",
            "Overall_OOF_ROC_AUC": "整體 OOF ROC-AUC",
            "Accuracy": "準確率",
            "Precision": "精確率",
            "Recall": "召回率",
            "F1": "F1 分數",
            "Mean_Training_Seconds": "平均訓練秒數",
        })
        st.dataframe(classification_table, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("三模型 Top 5 投資組合績效")
        best_portfolio = portfolio_df.loc[portfolio_df["Sharpe_Ratio"].idxmax()]
        cols = st.columns(4, gap="medium")
        portfolio_cards = [
            ("最佳 Sharpe Ratio", MODEL_NAME_MAP.get(best_portfolio["Model"], best_portfolio["Model"]), fmt_num(best_portfolio["Sharpe_Ratio"], 3), "#6d28d9"),
            ("累積報酬", fmt_pct(best_portfolio["Cumulative_Return"]), "最佳風險調整後模型", "#10b981"),
            ("最大回撤", fmt_pct(best_portfolio["Max_Drawdown"]), "相對歷史高點最大跌幅", "#dc2626"),
            ("平均每期淨報酬", fmt_pct(best_portfolio["Mean_Net_Return"]), "已扣除完整成本 0.60%", "#2563eb"),
        ]
        for column, item in zip(cols, portfolio_cards):
            with column:
                st.html(card(*item))

        portfolio_table = portfolio_df[[
            "模型", "Backtest_Periods", "Cumulative_Return", "Annual_Return",
            "Annual_Volatility", "Sharpe_Ratio", "Max_Drawdown", "Win_Rate",
            "Mean_Net_Return", "Median_Net_Return",
        ]].rename(columns={
            "Backtest_Periods": "回測期數",
            "Cumulative_Return": "累積報酬",
            "Annual_Return": "年化報酬",
            "Annual_Volatility": "年化波動率",
            "Sharpe_Ratio": "Sharpe Ratio",
            "Max_Drawdown": "最大回撤",
            "Win_Rate": "每期勝率",
            "Mean_Net_Return": "平均每期淨報酬",
            "Median_Net_Return": "每期淨報酬中位數",
        })
        st.dataframe(portfolio_table, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("三模型累積資產曲線")
        equity_figure = go.Figure()
        for model_name in MODEL_NAME_MAP:
            data = equity_df[equity_df["Model"] == model_name].sort_values("date")
            equity_figure.add_trace(go.Scatter(
                x=data["date"], y=data["Net_Equity"],
                name=MODEL_NAME_MAP[model_name], mode="lines",
                line={"color": MODEL_COLORS[model_name], "width": 2.5},
                hovertemplate="日期：%{x|%Y-%m-%d}<br>累積資產：%{y:.4f}<extra></extra>",
            ))
        st.plotly_chart(style_figure(equity_figure, "三模型 Top 5 累積資產曲線", "累積資產倍數", 630), use_container_width=True)

        st.divider()
        st.subheader("三模型回撤曲線與下行風險")
        drawdown_figure = go.Figure()
        for model_name in MODEL_NAME_MAP:
            data = equity_df[equity_df["Model"] == model_name].sort_values("date")
            drawdown_figure.add_trace(go.Scatter(
                x=data["date"], y=data["Drawdown"],
                name=MODEL_NAME_MAP[model_name], mode="lines",
                line={"color": MODEL_COLORS[model_name], "width": 2.2},
                hovertemplate="日期：%{x|%Y-%m-%d}<br>回撤：%{y:.2%}<extra></extra>",
            ))
        st.plotly_chart(style_figure(drawdown_figure, "三模型 Top 5 回撤曲線", "相對歷史高點跌幅", 560, True), use_container_width=True)

        st.divider()
        st.subheader("模型選擇結論")
        xgb_class = comparison_df[comparison_df["Model"] == "XGBoost"].iloc[0]
        rf_class = comparison_df[comparison_df["Model"] == "Random Forest"].iloc[0]
        lr_class = comparison_df[comparison_df["Model"] == "Logistic Regression"].iloc[0]
        xgb_port = portfolio_df[portfolio_df["Model"] == "XGBoost"].iloc[0]

        st.html(f"""
        <div class="ai-card">
          <div class="ai-card-title">綜合判斷：正式排行榜繼續使用 XGBoost</div>
          <div class="ai-card-text">
            <strong>分類能力：</strong>隨機森林的五折平均 ROC-AUC 為 {rf_class['Mean_ROC_AUC']:.4f}，
            略高於邏輯斯迴歸的 {lr_class['Mean_ROC_AUC']:.4f} 與 XGBoost 的 {xgb_class['Mean_ROC_AUC']:.4f}，
            但三者差距很小。<br><br>
            <strong>穩定性：</strong>XGBoost 的 ROC-AUC 標準差為 {xgb_class['Std_ROC_AUC']:.4f}，
            是三個模型中最低，代表跨 Fold 表現相對穩定。<br><br>
            <strong>投資組合績效：</strong>XGBoost Top 5 的累積報酬為 {xgb_port['Cumulative_Return']:.2%}、
            Sharpe Ratio 為 {xgb_port['Sharpe_Ratio']:.3f}、最大回撤為 {xgb_port['Max_Drawdown']:.2%}，
            在三模型 Top 5 比較中具有最佳風險調整後報酬。<br><br>
            <strong>部署考量：</strong>XGBoost 已整合每日排行榜、SHAP 解釋與 GitHub Actions，
            因此目前不更換正式模型。多模型結果作為公平比較與模型選擇依據。
          </div>
        </div>
        """)

        st.warning("多模型結果屬於歷史 OOF 模擬，不代表未來績效，也不構成投資建議。")


render_footer()
