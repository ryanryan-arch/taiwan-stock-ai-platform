from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"
RAW_PRICE_DIR = PROJECT_ROOT / "data" / "raw_price"

RANKING_PATH = (
    RESULT_DIR / "latest_rankings.csv"
)

LATEST_FEATURE_PATH = (
    RESULT_DIR / "latest_features.parquet"
)

SHAP_PATH = (
    RESULT_DIR / "latest_shap_values.parquet"
)


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="個股分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# SHAP 特徵中文名稱
# ==================================================

FEATURE_NAME_MAP = {
    "Return_1D": "1 日價格報酬",
    "Return_3D": "3 日價格報酬",
    "Return_5D": "5 日價格報酬",
    "Return_10D": "10 日價格報酬",
    "Return_20D": "20 日價格報酬",

    "Close_MA5_Ratio": "股價相對 MA5",
    "Close_MA10_Ratio": "股價相對 MA10",
    "Close_MA20_Ratio": "股價相對 MA20",
    "Close_MA60_Ratio": "股價相對 MA60",
    "MA5_MA20_Ratio": "MA5 相對 MA20",
    "MA20_MA60_Ratio": "MA20 相對 MA60",

    "RSI": "RSI",
    "MACD_Diff_Ratio": "MACD 柱狀體比例",
    "ATR_Ratio": "ATR 波動比例",
    "Volatility_5D": "5 日歷史波動率",
    "Volatility_20D": "20 日歷史波動率",

    "Volume_Ratio": "成交量相對 20 日均量",
    "Volume_5D_20D_Ratio": (
        "5 日均量相對 20 日均量"
    ),

    "Foreign_NetBuy_Ratio": (
        "外資單日買賣超比例"
    ),
    "InvestmentTrust_NetBuy_Ratio": (
        "投信單日買賣超比例"
    ),
    "Dealer_NetBuy_Ratio": (
        "自營商單日買賣超比例"
    ),
    "Institutional_NetBuy_Ratio": (
        "三大法人單日買賣超比例"
    ),

    "Foreign_NetBuy_5D_Ratio": (
        "外資 5 日買賣超比例"
    ),
    "Foreign_NetBuy_20D_Ratio": (
        "外資 20 日買賣超比例"
    ),
    "InvestmentTrust_NetBuy_5D_Ratio": (
        "投信 5 日買賣超比例"
    ),
    "InvestmentTrust_NetBuy_20D_Ratio": (
        "投信 20 日買賣超比例"
    ),
    "Dealer_NetBuy_5D_Ratio": (
        "自營商 5 日買賣超比例"
    ),
    "Institutional_NetBuy_5D_Ratio": (
        "三大法人 5 日買賣超比例"
    ),
    "Institutional_NetBuy_20D_Ratio": (
        "三大法人 20 日買賣超比例"
    ),

    "Foreign_Buy_Streak": (
        "外資連續買超天數"
    ),
    "InvestmentTrust_Buy_Streak": (
        "投信連續買超天數"
    ),
    "Dealer_Buy_Streak": (
        "自營商連續買超天數"
    ),
}


# ==================================================
# 資料載入函式
# ==================================================

@st.cache_data(ttl=300)
def load_rankings():
    """
    載入最新 85 檔 AI 排行榜。
    """

    if not RANKING_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        RANKING_PATH,
        dtype={"StockID": str},
    )

    dataframe["StockID"] = (
        dataframe["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

    return dataframe


@st.cache_data(ttl=300)
def load_latest_features():
    """
    載入最新 85 檔完整特徵資料。
    """

    if not LATEST_FEATURE_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_parquet(
        LATEST_FEATURE_PATH
    )

    dataframe["StockID"] = (
        dataframe["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

    return dataframe


@st.cache_data(ttl=300)
def load_shap_values():
    """
    載入最新 85 檔 SHAP 模型解釋。
    """

    if not SHAP_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_parquet(
        SHAP_PATH
    )

    required_columns = [
        "StockID",
        "Feature",
        "Feature_Value",
        "SHAP_Value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    dataframe["StockID"] = (
        dataframe["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    dataframe["Feature_Value"] = (
        pd.to_numeric(
            dataframe["Feature_Value"],
            errors="coerce",
        )
    )

    dataframe["SHAP_Value"] = (
        pd.to_numeric(
            dataframe["SHAP_Value"],
            errors="coerce",
        )
    )

    dataframe["Abs_SHAP_Value"] = (
        dataframe["SHAP_Value"].abs()
    )

    return dataframe


@st.cache_data(ttl=300)
def load_price_history(stock_id):
    """
    載入指定股票的歷史股價。
    """

    possible_paths = [
        RAW_PRICE_DIR
        / f"{stock_id}_price.parquet",

        RAW_PRICE_DIR
        / f"{stock_id}.parquet",
    ]

    price_path = None

    for path in possible_paths:
        if path.exists():
            price_path = path
            break

    if price_path is None:
        return pd.DataFrame()

    dataframe = pd.read_parquet(
        price_path
    )

    if "Date" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"Date": "date"}
        )

    if "date" not in dataframe.columns:
        return pd.DataFrame()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    dataframe = (
        dataframe
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataframe


# ==================================================
# 顯示格式函式
# ==================================================

def safe_number(
    row,
    column_name,
    number_format="{:.2f}",
):
    """
    安全格式化數值。
    """

    value = row.get(
        column_name,
        float("nan"),
    )

    if pd.isna(value):
        return "資料不足"

    try:
        return number_format.format(value)

    except (TypeError, ValueError):
        return str(value)


def format_feature_value(
    feature_name,
    feature_value,
):
    """
    依 SHAP 特徵類型格式化實際數值。
    """

    if pd.isna(feature_value):
        return "資料不足"

    percentage_features = {
        "Return_1D",
        "Return_3D",
        "Return_5D",
        "Return_10D",
        "Return_20D",
        "Close_MA5_Ratio",
        "Close_MA10_Ratio",
        "Close_MA20_Ratio",
        "Close_MA60_Ratio",
        "MA5_MA20_Ratio",
        "MA20_MA60_Ratio",
        "MACD_Diff_Ratio",
        "ATR_Ratio",
        "Volatility_5D",
        "Volatility_20D",
        "Foreign_NetBuy_Ratio",
        "InvestmentTrust_NetBuy_Ratio",
        "Dealer_NetBuy_Ratio",
        "Institutional_NetBuy_Ratio",
        "Foreign_NetBuy_5D_Ratio",
        "Foreign_NetBuy_20D_Ratio",
        "InvestmentTrust_NetBuy_5D_Ratio",
        "InvestmentTrust_NetBuy_20D_Ratio",
        "Dealer_NetBuy_5D_Ratio",
        "Institutional_NetBuy_5D_Ratio",
        "Institutional_NetBuy_20D_Ratio",
    }

    streak_features = {
        "Foreign_Buy_Streak",
        "InvestmentTrust_Buy_Streak",
        "Dealer_Buy_Streak",
    }

    if feature_name in percentage_features:
        return f"{feature_value:.2%}"

    if feature_name in streak_features:
        return f"{feature_value:.0f} 日"

    if feature_name == "RSI":
        return f"{feature_value:.2f}"

    if feature_name in {
        "Volume_Ratio",
        "Volume_5D_20D_Ratio",
    }:
        return f"{feature_value:.2f} 倍"

    return f"{feature_value:.6f}"


# ==================================================
# 載入資料
# ==================================================

ranking_df = load_rankings()
latest_features_df = load_latest_features()
shap_df = load_shap_values()


# ==================================================
# 核心資料檢查
# ==================================================

if ranking_df.empty:
    st.error(
        "找不到 results/latest_rankings.csv，"
        "無法載入個股排行榜。"
    )
    st.stop()


if latest_features_df.empty:
    st.error(
        "找不到 results/latest_features.parquet，"
        "無法載入個股特徵資料。"
    )
    st.stop()


required_ranking_columns = [
    "StockID",
    "StockName",
    "Rank",
    "AI_Score",
    "Close",
    "Signal",
    "Risk_Level",
]


missing_ranking_columns = [
    column
    for column in required_ranking_columns
    if column not in ranking_df.columns
]


if missing_ranking_columns:
    st.error(
        "排行榜資料缺少必要欄位："
        f"{missing_ranking_columns}"
    )
    st.stop()


# ==================================================
# 頁面標題
# ==================================================

st.title("個股 AI 分析")

st.caption(
    "查看個股 AI 排名、最新技術指標、"
    "歷史走勢、法人籌碼摘要及 SHAP 模型解釋。"
)


# ==================================================
# 股票選擇
# ==================================================

ranking_df["DisplayName"] = (
    ranking_df["StockID"]
    + " "
    + ranking_df["StockName"]
)


stock_options = (
    ranking_df
    .sort_values("Rank")["DisplayName"]
    .tolist()
)


selected_display = st.selectbox(
    "選擇股票",
    options=stock_options,
)


selected_stock_id = (
    selected_display.split(" ")[0]
)


selected_ranking_df = ranking_df[
    ranking_df["StockID"]
    == selected_stock_id
].copy()


selected_feature_df = latest_features_df[
    latest_features_df["StockID"]
    == selected_stock_id
].copy()


if selected_ranking_df.empty:
    st.error(
        "排行榜中找不到所選股票。"
    )
    st.stop()


if selected_feature_df.empty:
    st.error(
        "最新特徵中找不到所選股票。"
    )
    st.stop()


selected_ranking = (
    selected_ranking_df.iloc[0]
)

selected_features = (
    selected_feature_df.iloc[0]
)


# ==================================================
# 個股 AI 摘要
# ==================================================

st.subheader(
    f'{selected_ranking["StockID"]} '
    f'{selected_ranking["StockName"]}'
)


if "Industry" in selected_ranking.index:
    st.caption(
        "產業族群："
        f'{selected_ranking["Industry"]}'
    )


summary_col1, summary_col2, summary_col3, \
summary_col4, summary_col5 = st.columns(5)


summary_col1.metric(
    "AI 排名",
    f'第 {int(selected_ranking["Rank"])} 名',
)


summary_col2.metric(
    "AI 分數",
    safe_number(
        selected_ranking,
        "AI_Score",
        "{:.2f}",
    ),
)


summary_col3.metric(
    "最新收盤價",
    safe_number(
        selected_ranking,
        "Close",
        "{:.2f}",
    ),
)


summary_col4.metric(
    "模型訊號",
    str(
        selected_ranking.get(
            "Signal",
            "資料不足",
        )
    ),
)


summary_col5.metric(
    "風險等級",
    str(
        selected_ranking.get(
            "Risk_Level",
            "資料不足",
        )
    ),
)


data_date = None

if "date" in selected_ranking.index:
    data_date = pd.to_datetime(
        selected_ranking.get("date"),
        errors="coerce",
    )


if pd.notna(data_date):
    st.caption(
        "排行榜資料日期："
        f"{data_date.strftime('%Y-%m-%d')}"
    )


st.divider()


# ==================================================
# 股價與均線走勢
# ==================================================

st.subheader("股價與均線走勢")


period_options = {
    "近 3 個月": 65,
    "近 6 個月": 130,
    "近 1 年": 252,
    "近 2 年": 504,
}


selected_period = st.radio(
    "顯示期間",
    options=list(period_options.keys()),
    horizontal=True,
)


price_df = load_price_history(
    selected_stock_id
)


required_price_columns = [
    "date",
    "Open",
    "High",
    "Low",
    "Close",
]


if price_df.empty:

    st.warning(
        "找不到這檔股票的歷史股價資料。"
    )

elif any(
    column not in price_df.columns
    for column in required_price_columns
):

    st.warning(
        "歷史股價資料缺少 K 線必要欄位。"
    )

else:

    chart_df = (
        price_df
        .tail(
            period_options[
                selected_period
            ]
        )
        .copy()
    )

    chart_df["MA5"] = (
        chart_df["Close"]
        .rolling(5)
        .mean()
    )

    chart_df["MA20"] = (
        chart_df["Close"]
        .rolling(20)
        .mean()
    )

    chart_df["MA60"] = (
        chart_df["Close"]
        .rolling(60)
        .mean()
    )

    price_figure = go.Figure()

    price_figure.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="K 線",
            increasing_line_color="#D62728",
            decreasing_line_color="#2CA02C",
        )
    )

    price_figure.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["MA5"],
            name="MA5",
            mode="lines",
            line={
                "color": "#F5A623",
                "width": 1.5,
            },
        )
    )

    price_figure.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["MA20"],
            name="MA20",
            mode="lines",
            line={
                "color": "#1F77B4",
                "width": 1.8,
            },
        )
    )

    price_figure.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["MA60"],
            name="MA60",
            mode="lines",
            line={
                "color": "#9467BD",
                "width": 1.8,
            },
        )
    )

    price_figure.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        title=(
            f'{selected_ranking["StockID"]} '
            f'{selected_ranking["StockName"]} '
            f'{selected_period}走勢'
        ),
        xaxis_title="日期",
        yaxis_title="股價",
        legend_title="價格指標",
    )

    st.plotly_chart(
        price_figure,
        use_container_width=True,
    )


st.divider()


# ==================================================
# 最新技術指標
# ==================================================

st.subheader("最新技術與量價指標")


technical_metrics = [
    (
        "RSI",
        "RSI",
        "{:.2f}",
    ),
    (
        "成交量比率",
        "Volume_Ratio",
        "{:.2f} 倍",
    ),
    (
        "ATR 風險比例",
        "ATR_Ratio",
        "{:.2%}",
    ),
    (
        "1 日報酬",
        "Return_1D",
        "{:.2%}",
    ),
    (
        "5 日報酬",
        "Return_5D",
        "{:.2%}",
    ),
    (
        "20 日報酬",
        "Return_20D",
        "{:.2%}",
    ),
]


technical_columns = st.columns(3)


for index, (
    label,
    column_name,
    value_format,
) in enumerate(technical_metrics):

    value = selected_features.get(
        column_name,
        float("nan"),
    )

    if pd.isna(value):
        display_value = "資料不足"

    else:
        display_value = (
            value_format.format(value)
        )

    technical_columns[
        index % 3
    ].metric(
        label,
        display_value,
    )


st.divider()


# ==================================================
# 技術面與籌碼面摘要
# ==================================================

st.subheader("技術面與籌碼面摘要")

st.caption(
    "以下內容依最新特徵數值產生，"
    "屬於規則式狀態摘要，不是模型內部判斷原因。"
)


summary_messages = []


close_ma20_ratio = selected_features.get(
    "Close_MA20_Ratio",
    float("nan"),
)


if pd.notna(close_ma20_ratio):

    if close_ma20_ratio > 0:
        summary_messages.append(
            "股價位於 20 日均線上方，"
            "目前中短期價格位置相對偏強。"
        )

    else:
        summary_messages.append(
            "股價位於 20 日均線下方，"
            "目前中短期價格位置相對偏弱。"
        )


ma5_ma20_ratio = selected_features.get(
    "MA5_MA20_Ratio",
    float("nan"),
)


if pd.notna(ma5_ma20_ratio):

    if ma5_ma20_ratio > 0:
        summary_messages.append(
            "MA5 高於 MA20，"
            "短期均線結構相對偏多。"
        )

    else:
        summary_messages.append(
            "MA5 低於 MA20，"
            "短期均線結構相對偏弱。"
        )


volume_ratio = selected_features.get(
    "Volume_Ratio",
    float("nan"),
)


if pd.notna(volume_ratio):

    if volume_ratio > 1:
        summary_messages.append(
            "當日成交量高於 20 日平均成交量。"
        )

    else:
        summary_messages.append(
            "當日成交量低於 20 日平均成交量。"
        )


foreign_5d = selected_features.get(
    "Foreign_NetBuy_5D_Ratio",
    float("nan"),
)


if pd.notna(foreign_5d):

    if foreign_5d > 0:
        summary_messages.append(
            "外資近 5 日累計買賣超比例為正。"
        )

    elif foreign_5d < 0:
        summary_messages.append(
            "外資近 5 日累計買賣超比例為負。"
        )

    else:
        summary_messages.append(
            "外資近 5 日累計買賣超接近中性。"
        )


trust_5d = selected_features.get(
    "InvestmentTrust_NetBuy_5D_Ratio",
    float("nan"),
)


if pd.notna(trust_5d):

    if trust_5d > 0:
        summary_messages.append(
            "投信近 5 日累計買賣超比例為正。"
        )

    elif trust_5d < 0:
        summary_messages.append(
            "投信近 5 日累計買賣超比例為負。"
        )

    else:
        summary_messages.append(
            "投信近 5 日累計買賣超接近中性。"
        )


if summary_messages:

    for message in summary_messages:
        st.write(f"• {message}")

else:

    st.info(
        "目前沒有足夠資料產生特徵摘要。"
    )


st.divider()


# ==================================================
# SHAP 模型解釋
# ==================================================

st.subheader("SHAP 模型解釋")

st.caption(
    "SHAP 解釋每個特徵如何影響 XGBoost "
    "對所選股票的模型輸出。"
    "正值代表推升 Target = 1 的模型輸出，"
    "負值代表壓低模型輸出。"
)


if shap_df.empty:

    st.info(
        "目前尚未產生 SHAP 模型解釋資料。"
        "請先執行 scripts/calculate_shap.py。"
    )

else:

    selected_shap_df = shap_df[
        shap_df["StockID"]
        == selected_stock_id
    ].copy()

    selected_shap_df = (
        selected_shap_df
        .dropna(
            subset=[
                "Feature",
                "SHAP_Value",
            ]
        )
        .copy()
    )

    if selected_shap_df.empty:

        st.info(
            "目前沒有這檔股票的 SHAP 解釋資料。"
        )

    else:

        selected_shap_df[
            "特徵名稱"
        ] = (
            selected_shap_df["Feature"]
            .map(FEATURE_NAME_MAP)
            .fillna(
                selected_shap_df[
                    "Feature"
                ]
            )
        )

        selected_shap_df[
            "特徵實際值"
        ] = selected_shap_df.apply(
            lambda row: format_feature_value(
                row["Feature"],
                row["Feature_Value"],
            ),
            axis=1,
        )

        positive_shap_df = (
            selected_shap_df[
                selected_shap_df[
                    "SHAP_Value"
                ] > 0
            ]
            .nlargest(
                5,
                "SHAP_Value",
            )
            .copy()
        )

        negative_shap_df = (
            selected_shap_df[
                selected_shap_df[
                    "SHAP_Value"
                ] < 0
            ]
            .nsmallest(
                5,
                "SHAP_Value",
            )
            .copy()
        )

        positive_col, negative_col = (
            st.columns(2)
        )

        with positive_col:

            st.markdown(
                "### 推升 AI 分數的主要因素"
            )

            if positive_shap_df.empty:

                st.write(
                    "目前沒有明顯正向因素。"
                )

            else:

                for _, row in (
                    positive_shap_df.iterrows()
                ):

                    st.markdown(
                        f'**{row["特徵名稱"]}**  \n'
                        f'實際值：'
                        f'{row["特徵實際值"]}  \n'
                        f'SHAP 貢獻：'
                        f'`{row["SHAP_Value"]:+.4f}`'
                    )

        with negative_col:

            st.markdown(
                "### 壓低 AI 分數的主要因素"
            )

            if negative_shap_df.empty:

                st.write(
                    "目前沒有明顯負向因素。"
                )

            else:

                for _, row in (
                    negative_shap_df.iterrows()
                ):

                    st.markdown(
                        f'**{row["特徵名稱"]}**  \n'
                        f'實際值：'
                        f'{row["特徵實際值"]}  \n'
                        f'SHAP 貢獻：'
                        f'`{row["SHAP_Value"]:+.4f}`'
                    )

        st.markdown(
            "### 模型影響程度前 10 名"
        )

        chart_df = (
            selected_shap_df
            .nlargest(
                10,
                "Abs_SHAP_Value",
            )
            .sort_values(
                "SHAP_Value",
                ascending=True,
            )
            .copy()
        )

        chart_df["影響方向"] = (
            chart_df["SHAP_Value"]
            .apply(
                lambda value: (
                    "推升模型分數"
                    if value > 0
                    else "壓低模型分數"
                )
            )
        )

        shap_figure = px.bar(
            chart_df,
            x="SHAP_Value",
            y="特徵名稱",
            orientation="h",
            color="影響方向",
            color_discrete_map={
                "推升模型分數": "#D62728",
                "壓低模型分數": "#2CA02C",
            },
            custom_data=[
                "Feature",
                "特徵實際值",
            ],
            labels={
                "SHAP_Value": "SHAP 貢獻值",
                "特徵名稱": "模型特徵",
            },
        )

        # 修正版滑鼠提示：
        # x 軸本身就是 SHAP_Value，
        # 因此直接使用 %{x:+.4f} 顯示。
        shap_figure.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "原始特徵："
                "%{customdata[0]}<br>"
                "特徵值："
                "%{customdata[1]}<br>"
                "SHAP 貢獻："
                "%{x:+.4f}"
                "<extra></extra>"
            )
        )

        shap_figure.update_layout(
            height=560,
            xaxis_title=(
                "對 XGBoost 模型原始輸出的貢獻"
            ),
            yaxis_title="",
            legend_title="影響方向",
            hovermode="closest",
        )

        shap_figure.add_vline(
            x=0,
            line_width=1,
            line_color="#666666",
        )

        st.plotly_chart(
            shap_figure,
            use_container_width=True,
        )

        top_impact_feature = (
            selected_shap_df
            .nlargest(
                1,
                "Abs_SHAP_Value",
            )
            .iloc[0]
        )

        top_direction = (
            "推升"
            if top_impact_feature[
                "SHAP_Value"
            ] > 0
            else "壓低"
        )

        st.success(
            "目前對這檔股票影響最大的特徵是："
            f'「{top_impact_feature["特徵名稱"]}」，'
            f'其作用方向為「{top_direction}模型分數」。'
        )

        st.info(
            "SHAP 值反映特徵對模型輸出的貢獻，"
            "不是報酬率，也不是實際上漲機率。"
            "不同特徵可能存在交互作用，"
            "不應將單一 SHAP 因素視為買賣依據。"
        )


# ==================================================
# 免責聲明
# ==================================================

st.divider()

st.info(
    "AI 分數是模型用於 85 檔股票相對排序的分數，"
    "尚未經過機率校準，不應直接視為真實上漲機率。"
)

st.warning(
    "本平台僅供課程專題、資料分析與模型研究，"
    "不構成投資建議、買賣推薦或獲利保證。"
)