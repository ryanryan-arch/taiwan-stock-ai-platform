from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results"


st.set_page_config(
    page_title="族群排行",
    page_icon="🏭",
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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    return df


@st.cache_data(ttl=300)
def load_industry_rankings():

    df = pd.read_csv(
        RESULT_DIR / "industry_rankings.csv",
        dtype={"Top_StockID": str},
    )

    if "Top_StockID" in df.columns:
        df["Top_StockID"] = (
            df["Top_StockID"]
            .astype(str)
            .str.zfill(4)
        )

    return df


ranking_df = load_rankings()
industry_df = load_industry_rankings()


st.title("17 族群 AI 強度排行")

st.caption(
    "比較 17 個產業族群的平均 AI 分數、"
    "法人籌碼、技術面狀態及族群內個股排名。"
)


data_date = "未知"

if "date" in ranking_df.columns:
    latest_date = ranking_df["date"].max()

    if pd.notna(latest_date):
        data_date = latest_date.strftime(
            "%Y-%m-%d"
        )


top_industry = industry_df.sort_values(
    "Industry_Rank"
).iloc[0]


top_industry_name = top_industry["Industry"]

top_industry_score = top_industry[
    "Industry_AI_Score"
]

top_stock_name = top_industry[
    "Top_StockName"
]

top_stock_score = top_industry[
    "Top_Stock_AI_Score"
]


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "資料日期",
    data_date,
)

col2.metric(
    "分析族群數",
    ranking_df["Industry"].nunique(),
)

col3.metric(
    "目前最強族群",
    top_industry_name,
    f"{top_industry_score:.2f} 分",
)

col4.metric(
    "最強族群代表股",
    top_stock_name,
    f"{top_stock_score:.2f} 分",
)


st.divider()

st.subheader("族群 AI 分數比較")


chart_df = (
    industry_df
    .sort_values(
        "Industry_AI_Score",
        ascending=True,
    )
    .copy()
)


industry_chart = px.bar(
    chart_df,
    x="Industry_AI_Score",
    y="Industry",
    orientation="h",
    color="Industry_AI_Score",
    color_continuous_scale="RdYlGn",
    labels={
        "Industry_AI_Score": "族群平均 AI 分數",
        "Industry": "產業族群",
    },
    hover_data={
        "Industry_AI_Score": ":.2f",
        "Highest_AI_Score": ":.2f",
        "Average_RSI": ":.2f",
        "Stock_Count": True,
    },
)


industry_chart.update_layout(
    height=700,
    coloraxis_showscale=False,
    xaxis_title="族群平均 AI 分數",
    yaxis_title="產業族群",
    hovermode="closest",
)


st.plotly_chart(
    industry_chart,
    use_container_width=True,
)


st.divider()

st.subheader("完整族群排行榜")


industry_display_df = (
    industry_df
    .sort_values("Industry_Rank")
    .copy()
)


industry_display_columns = [
    column
    for column in [
        "Industry_Rank",
        "Industry",
        "Industry_AI_Score",
        "Highest_AI_Score",
        "Average_RSI",
        "Top_StockID",
        "Top_StockName",
        "Top_Stock_AI_Score",
        "Stock_Count",
    ]
    if column in industry_display_df.columns
]


industry_column_config = {}

if "Industry_Rank" in industry_display_columns:
    industry_column_config[
        "Industry_Rank"
    ] = st.column_config.NumberColumn(
        "族群排名",
        format="%d",
    )

if "Industry" in industry_display_columns:
    industry_column_config[
        "Industry"
    ] = st.column_config.TextColumn(
        "產業族群",
    )

if "Industry_AI_Score" in industry_display_columns:
    industry_column_config[
        "Industry_AI_Score"
    ] = st.column_config.ProgressColumn(
        "族群平均 AI 分數",
        min_value=0,
        max_value=100,
        format="%.2f",
    )

if "Highest_AI_Score" in industry_display_columns:
    industry_column_config[
        "Highest_AI_Score"
    ] = st.column_config.NumberColumn(
        "族群最高 AI 分數",
        format="%.2f",
    )

if "Average_RSI" in industry_display_columns:
    industry_column_config[
        "Average_RSI"
    ] = st.column_config.NumberColumn(
        "族群平均 RSI",
        format="%.2f",
    )

if "Top_StockID" in industry_display_columns:
    industry_column_config[
        "Top_StockID"
    ] = st.column_config.TextColumn(
        "代表股代碼",
    )

if "Top_StockName" in industry_display_columns:
    industry_column_config[
        "Top_StockName"
    ] = st.column_config.TextColumn(
        "代表股",
    )

if "Top_Stock_AI_Score" in industry_display_columns:
    industry_column_config[
        "Top_Stock_AI_Score"
    ] = st.column_config.NumberColumn(
        "代表股 AI 分數",
        format="%.2f",
    )

if "Stock_Count" in industry_display_columns:
    industry_column_config[
        "Stock_Count"
    ] = st.column_config.NumberColumn(
        "股票數量",
        format="%d",
    )


st.dataframe(
    industry_display_df[
        industry_display_columns
    ],
    use_container_width=True,
    hide_index=True,
    column_config=industry_column_config,
)


st.divider()

st.subheader("查看族群內個股排行")


industry_options = (
    industry_df
    .sort_values("Industry_Rank")["Industry"]
    .tolist()
)


selected_industry = st.selectbox(
    "選擇產業族群",
    options=industry_options,
)


selected_industry_df = (
    ranking_df[
        ranking_df["Industry"] == selected_industry
    ]
    .sort_values(
        "AI_Score",
        ascending=False,
    )
    .copy()
)


selected_industry_df[
    "Industry_Rank"
] = range(
    1,
    len(selected_industry_df) + 1,
)


selected_industry_info = industry_df[
    industry_df["Industry"] == selected_industry
].iloc[0]


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


summary_col1.metric(
    "族群總排名",
    f'第 {int(selected_industry_info["Industry_Rank"])} 名',
)


summary_col2.metric(
    "族群平均 AI 分數",
    f'{selected_industry_info["Industry_AI_Score"]:.2f}',
)


summary_col3.metric(
    "族群代表股",
    selected_industry_info["Top_StockName"],
    f'{selected_industry_info["Top_Stock_AI_Score"]:.2f} 分',
)


stock_display_columns = [
    column
    for column in [
        "Industry_Rank",
        "StockID",
        "StockName",
        "Close",
        "AI_Score",
        "Signal",
        "Risk_Level",
        "RSI",
        "Foreign_NetBuy_5D_Ratio",
        "InvestmentTrust_NetBuy_5D_Ratio",
    ]
    if column in selected_industry_df.columns
]


stock_column_config = {}

if "Industry_Rank" in stock_display_columns:
    stock_column_config[
        "Industry_Rank"
    ] = st.column_config.NumberColumn(
        "族群內排名",
        format="%d",
    )

if "StockID" in stock_display_columns:
    stock_column_config[
        "StockID"
    ] = st.column_config.TextColumn(
        "股票代碼",
    )

if "StockName" in stock_display_columns:
    stock_column_config[
        "StockName"
    ] = st.column_config.TextColumn(
        "股票名稱",
    )

if "Close" in stock_display_columns:
    stock_column_config[
        "Close"
    ] = st.column_config.NumberColumn(
        "最新收盤價",
        format="%.2f",
    )

if "AI_Score" in stock_display_columns:
    stock_column_config[
        "AI_Score"
    ] = st.column_config.ProgressColumn(
        "AI 分數",
        min_value=0,
        max_value=100,
        format="%.2f",
    )

if "Signal" in stock_display_columns:
    stock_column_config[
        "Signal"
    ] = st.column_config.TextColumn(
        "模型訊號",
    )

if "Risk_Level" in stock_display_columns:
    stock_column_config[
        "Risk_Level"
    ] = st.column_config.TextColumn(
        "風險等級",
    )

if "RSI" in stock_display_columns:
    stock_column_config[
        "RSI"
    ] = st.column_config.NumberColumn(
        "RSI",
        format="%.2f",
    )

if "Foreign_NetBuy_5D_Ratio" in stock_display_columns:
    stock_column_config[
        "Foreign_NetBuy_5D_Ratio"
    ] = st.column_config.NumberColumn(
        "外資 5 日買賣超比例",
        format="%.2f%%",
    )

if (
    "InvestmentTrust_NetBuy_5D_Ratio"
    in stock_display_columns
):
    stock_column_config[
        "InvestmentTrust_NetBuy_5D_Ratio"
    ] = st.column_config.NumberColumn(
        "投信 5 日買賣超比例",
        format="%.2f%%",
    )


st.dataframe(
    selected_industry_df[
        stock_display_columns
    ],
    use_container_width=True,
    hide_index=True,
    column_config=stock_column_config,
)


st.subheader("族群內 AI 分數比較")


stock_chart = px.bar(
    selected_industry_df.sort_values(
        "AI_Score",
        ascending=True,
    ),
    x="AI_Score",
    y="StockName",
    orientation="h",
    color="AI_Score",
    color_continuous_scale="RdYlGn",
    text="AI_Score",
    labels={
        "AI_Score": "AI 分數",
        "StockName": "股票名稱",
    },
)


stock_chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)


stock_chart.update_layout(
    height=420,
    coloraxis_showscale=False,
    xaxis_title="AI 分數",
    yaxis_title="股票名稱",
)


st.plotly_chart(
    stock_chart,
    use_container_width=True,
)


st.info(
    "族群 AI 分數是該族群 5 檔股票 AI 分數的平均值，"
    "適合用於相對強弱比較，不代表整個產業必然上漲。"
)

st.warning(
    "本頁僅供課程研究及模型驗證，"
    "不構成任何投資建議。"
)