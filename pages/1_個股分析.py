from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results"
RAW_PRICE_DIR = PROJECT_ROOT / "data" / "raw_price"


st.set_page_config(
    page_title="個股分析",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_rankings():
    df = pd.read_csv(
        RESULT_DIR / "latest_rankings.csv",
        dtype={"StockID": str},
    )

    df["StockID"] = (
        df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return df


@st.cache_data(ttl=300)
def load_latest_features():
    df = pd.read_parquet(
        RESULT_DIR / "latest_features.parquet"
    )

    df["StockID"] = (
        df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return df


@st.cache_data(ttl=300)
def load_price_history(stock_id):
    price_path = (
        RAW_PRICE_DIR
        / f"{stock_id}_price.parquet"
    )

    if not price_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(price_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


ranking_df = load_rankings()
latest_features_df = load_latest_features()


st.title("個股 AI 分析")

st.caption(
    "查看個股 AI 排名、技術指標、風險狀態與歷史走勢。"
)


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


selected_ranking = ranking_df[
    ranking_df["StockID"] == selected_stock_id
].iloc[0]


selected_features = latest_features_df[
    latest_features_df["StockID"]
    == selected_stock_id
].iloc[0]


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "AI 排名",
    f'第 {int(selected_ranking["Rank"])} 名',
)

col2.metric(
    "AI 分數",
    f'{selected_ranking["AI_Score"]:.2f}',
)

col3.metric(
    "最新收盤價",
    f'{selected_ranking["Close"]:.2f}',
)

col4.metric(
    "模型訊號",
    selected_ranking["Signal"],
)

col5.metric(
    "風險等級",
    selected_ranking["Risk_Level"],
)


st.divider()


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


if price_df.empty:
    st.warning("找不到這檔股票的歷史股價資料。")

else:
    price_df = price_df.tail(
        period_options[selected_period]
    ).copy()

    price_df["MA5"] = (
        price_df["Close"]
        .rolling(5)
        .mean()
    )

    price_df["MA20"] = (
        price_df["Close"]
        .rolling(20)
        .mean()
    )

    price_df["MA60"] = (
        price_df["Close"]
        .rolling(60)
        .mean()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=price_df["date"],
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="K 線",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["MA5"],
            name="MA5",
            mode="lines",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["MA20"],
            name="MA20",
            mode="lines",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["MA60"],
            name="MA60",
            mode="lines",
        )
    )

    figure.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        title=(
            f'{selected_ranking["StockID"]} '
            f'{selected_ranking["StockName"]}'
        ),
        xaxis_title="日期",
        yaxis_title="股價",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


st.divider()

st.subheader("最新技術指標")


technical_metrics = [
    ("RSI", "RSI", "{:.2f}"),
    ("成交量比率", "Volume_Ratio", "{:.2f}"),
    ("ATR 風險比例", "ATR_Ratio", "{:.2%}"),
    ("1 日報酬", "Return_1D", "{:.2%}"),
    ("5 日報酬", "Return_5D", "{:.2%}"),
    ("20 日報酬", "Return_20D", "{:.2%}"),
]


metric_columns = st.columns(3)


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
        display_value = value_format.format(value)

    metric_columns[index % 3].metric(
        label,
        display_value,
    )


st.subheader("AI 訊號解讀")


signal_messages = []


if selected_features["Close_MA20_Ratio"] > 0:
    signal_messages.append(
        "股價位於 20 日均線上方。"
    )
else:
    signal_messages.append(
        "股價位於 20 日均線下方。"
    )


if selected_features["MA5_MA20_Ratio"] > 0:
    signal_messages.append(
        "短期均線 MA5 高於 MA20。"
    )
else:
    signal_messages.append(
        "短期均線 MA5 低於 MA20。"
    )


if selected_features["Volume_Ratio"] > 1:
    signal_messages.append(
        "目前成交量高於 20 日平均量。"
    )
else:
    signal_messages.append(
        "目前成交量低於 20 日平均量。"
    )


if selected_features["Foreign_NetBuy_5D_Ratio"] > 0:
    signal_messages.append(
        "外資近 5 日累計買賣超為正。"
    )
else:
    signal_messages.append(
        "外資近 5 日累計買賣超為負。"
    )


if selected_features["InvestmentTrust_NetBuy_5D_Ratio"] > 0:
    signal_messages.append(
        "投信近 5 日累計買賣超為正。"
    )
else:
    signal_messages.append(
        "投信近 5 日累計買賣超為負。"
    )


for message in signal_messages:
    st.write(f"• {message}")


st.info(
    "AI 分數為模型相對排序結果，"
    "不代表保證上漲機率。"
)

st.warning(
    "本頁僅供課程研究與模型驗證，"
    "不構成投資建議。"
)