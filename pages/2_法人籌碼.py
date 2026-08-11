from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results"
MERGED_DIR = PROJECT_ROOT / "data" / "merged"


st.set_page_config(
    page_title="法人籌碼",
    page_icon="🏦",
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
def load_merged_history(stock_id):

    file_path = (
        MERGED_DIR
        / f"{stock_id}_merged.parquet"
    )

    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(file_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["StockID"] = (
        df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


ranking_df = load_rankings()
latest_features_df = load_latest_features()


st.title("三大法人籌碼分析")

st.caption(
    "整合外資、投信與自營商買賣超，"
    "觀察法人資金流向及連續買超狀態。"
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


selected_stock_id = selected_display.split(" ")[0]


selected_ranking = ranking_df[
    ranking_df["StockID"] == selected_stock_id
].iloc[0]


selected_features = latest_features_df[
    latest_features_df["StockID"] == selected_stock_id
].iloc[0]


history_df = load_merged_history(
    selected_stock_id
)


latest_price_date = selected_features["date"]


st.write(
    f'目前分析：**{selected_ranking["StockID"]} '
    f'{selected_ranking["StockName"]}**'
)

st.caption(
    f"最新股價資料日期："
    f"{latest_price_date.strftime('%Y-%m-%d')}"
)


st.divider()

st.subheader("最新法人籌碼摘要")


summary_columns = st.columns(4)


summary_items = [
    (
        "外資 5 日買賣超比例",
        "Foreign_NetBuy_5D_Ratio",
        "{:.2%}",
    ),
    (
        "投信 5 日買賣超比例",
        "InvestmentTrust_NetBuy_5D_Ratio",
        "{:.2%}",
    ),
    (
        "外資連續買超",
        "Foreign_Buy_Streak",
        "{:.0f} 日",
    ),
    (
        "投信連續買超",
        "InvestmentTrust_Buy_Streak",
        "{:.0f} 日",
    ),
]


for index, (
    label,
    column_name,
    value_format,
) in enumerate(summary_items):

    value = selected_features.get(
        column_name,
        float("nan"),
    )

    if pd.isna(value):
        display_value = "資料不足"
    else:
        display_value = value_format.format(value)

    summary_columns[index].metric(
        label,
        display_value,
    )


st.divider()

st.subheader("近期三大法人每日買賣超")


period_options = {
    "近 20 日": 20,
    "近 60 日": 60,
    "近 120 日": 120,
}


selected_period = st.radio(
    "顯示期間",
    options=list(period_options.keys()),
    horizontal=True,
)


if history_df.empty:

    st.warning(
        "找不到該股票的法人歷史資料。"
    )

else:

    chart_df = history_df.tail(
        period_options[selected_period]
    ).copy()

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["Foreign_NetBuy"],
            name="外資",
        )
    )

    figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["InvestmentTrust_NetBuy"],
            name="投信",
        )
    )

    figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["Dealer_NetBuy"],
            name="自營商",
        )
    )

    figure.update_layout(
        barmode="group",
        height=550,
        hovermode="x unified",
        xaxis_title="日期",
        yaxis_title="買賣超股數",
        title=(
            f'{selected_ranking["StockID"]} '
            f'{selected_ranking["StockName"]} '
            f'三大法人買賣超'
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


st.divider()

st.subheader("法人籌碼解讀")


messages = []


foreign_5d = selected_features.get(
    "Foreign_NetBuy_5D_Ratio",
    0,
)

trust_5d = selected_features.get(
    "InvestmentTrust_NetBuy_5D_Ratio",
    0,
)

foreign_streak = selected_features.get(
    "Foreign_Buy_Streak",
    0,
)

trust_streak = selected_features.get(
    "InvestmentTrust_Buy_Streak",
    0,
)


if foreign_5d > 0:
    messages.append(
        "外資近 5 日累計買賣超為正，籌碼偏多。"
    )
else:
    messages.append(
        "外資近 5 日累計買賣超為負，籌碼偏弱。"
    )


if trust_5d > 0:
    messages.append(
        "投信近 5 日累計買賣超為正。"
    )
else:
    messages.append(
        "投信近 5 日累計買賣超為負。"
    )


if foreign_streak >= 3:
    messages.append(
        f"外資已連續買超 {int(foreign_streak)} 日。"
    )


if trust_streak >= 3:
    messages.append(
        f"投信已連續買超 {int(trust_streak)} 日。"
    )


if foreign_5d > 0 and trust_5d > 0:
    messages.append(
        "外資與投信近 5 日方向一致，法人籌碼相對正向。"
    )


for message in messages:
    st.write(f"• {message}")


st.info(
    "法人買賣超資料可能晚於股價資料發布，"
    "網站後續自動更新時會分別檢查資料日期。"
)

st.warning(
    "法人買賣超不代表未來股價必然上漲，"
    "本頁僅供課程研究，不構成投資建議。"
)