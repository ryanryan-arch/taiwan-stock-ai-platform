from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "results"


st.set_page_config(
    page_title="台股 AI 智慧選股平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def load_rankings():
    ranking_path = RESULT_DIR / "latest_rankings.csv"

    df = pd.read_csv(
        ranking_path,
        dtype={"StockID": str},
    )

    df["StockID"] = (
        df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    return df


@st.cache_data(ttl=300)
def load_industry_rankings():
    return pd.read_csv(
        RESULT_DIR / "industry_rankings.csv"
    )


ranking_df = load_rankings()
industry_df = load_industry_rankings()


st.title("台股 AI 智慧選股與五日趨勢預測平台")

st.caption(
    "整合 Yahoo Finance、三大法人籌碼、"
    "32 個技術與籌碼特徵、XGBoost 與 "
    "TimeSeriesSplit 的 85 檔台股選股系統。"
)


data_date = "未知"

if "date" in ranking_df.columns:
    latest_date = ranking_df["date"].max()

    if pd.notna(latest_date):
        data_date = latest_date.strftime("%Y-%m-%d")


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "資料日期",
    data_date,
)

col2.metric(
    "股票數量",
    ranking_df["StockID"].nunique(),
)

col3.metric(
    "產業族群",
    ranking_df["Industry"].nunique(),
)

col4.metric(
    "AI Top 5 回測平均淨報酬",
    "0.66%",
)


st.divider()

st.subheader("今日 AI Top 10")


display_columns = [
    column
    for column in [
        "Rank",
        "StockID",
        "StockName",
        "Industry",
        "Close",
        "AI_Score",
        "Signal",
        "Risk_Level",
    ]
    if column in ranking_df.columns
]


top10_df = (
    ranking_df
    .sort_values("Rank")
    .head(10)
    .copy()
)


column_config = {}

if "Rank" in top10_df.columns:
    column_config["Rank"] = st.column_config.NumberColumn(
        "排名",
        format="%d",
    )

if "StockID" in top10_df.columns:
    column_config["StockID"] = st.column_config.TextColumn(
        "股票代碼",
    )

if "StockName" in top10_df.columns:
    column_config["StockName"] = st.column_config.TextColumn(
        "股票名稱",
    )

if "Industry" in top10_df.columns:
    column_config["Industry"] = st.column_config.TextColumn(
        "族群",
    )

if "Close" in top10_df.columns:
    column_config["Close"] = st.column_config.NumberColumn(
        "最新收盤價",
        format="%.2f",
    )

if "AI_Score" in top10_df.columns:
    column_config["AI_Score"] = (
        st.column_config.ProgressColumn(
            "AI 分數",
            min_value=0,
            max_value=100,
            format="%.2f",
        )
    )

if "Signal" in top10_df.columns:
    column_config["Signal"] = st.column_config.TextColumn(
        "模型訊號",
    )

if "Risk_Level" in top10_df.columns:
    column_config["Risk_Level"] = (
        st.column_config.TextColumn(
            "風險等級",
        )
    )


st.dataframe(
    top10_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)


st.subheader("完整 85 檔排行榜")


industry_options = [
    "全部族群"
] + sorted(
    ranking_df["Industry"]
    .dropna()
    .unique()
    .tolist()
)


selected_industry = st.selectbox(
    "依族群篩選",
    options=industry_options,
)


search_text = st.text_input(
    "搜尋股票代碼或名稱",
    placeholder="例如：2330、台積電",
)


filtered_df = ranking_df.copy()

if selected_industry != "全部族群":
    filtered_df = filtered_df[
        filtered_df["Industry"] == selected_industry
    ].copy()


if search_text.strip():
    keyword = search_text.strip()

    filtered_df = filtered_df[
        filtered_df["StockID"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
        )
        |
        filtered_df["StockName"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
        )
    ].copy()


st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)


st.divider()

st.subheader("17 族群 AI 排行")


industry_display_columns = [
    column
    for column in [
        "Industry_Rank",
        "Industry",
        "Industry_AI_Score",
        "Top_StockID",
        "Top_StockName",
        "Top_Stock_AI_Score",
    ]
    if column in industry_df.columns
]


st.dataframe(
    industry_df[industry_display_columns],
    use_container_width=True,
    hide_index=True,
)


st.info(
    "AI 分數代表模型的相對排序分數，"
    "尚未經過機率校準，不應視為保證上漲機率。"
)

st.warning(
    "本平台僅供課程研究與模型驗證，"
    "不構成任何投資建議。"
)