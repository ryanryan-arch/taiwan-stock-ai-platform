import json
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import (
    load_global_css,
    render_footer,
    render_sidebar_info,
)


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "results"

RANKING_PATH = (
    RESULT_DIR / "latest_rankings.csv"
)

INDUSTRY_RANKING_PATH = (
    RESULT_DIR / "industry_rankings.csv"
)

UPDATE_STATUS_PATH = (
    RESULT_DIR / "update_status.json"
)
TOP10_SNAPSHOT_PATH = (
    RESULT_DIR
    / "daily_top10_snapshots.csv"
)

TOP10_PERFORMANCE_PATH = (
    RESULT_DIR
    / "daily_top10_performance.csv"
)

# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="台股 AI 智慧選股平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 載入全站 CSS 與側邊欄資訊
# ==================================================

load_global_css()
render_sidebar_info()


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

    if "StockID" in dataframe.columns:
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

    numeric_columns = [
        "Rank",
        "Close",
        "AI_Score",
        "Predicted_Probability",
        "RSI",
        "Volume_Ratio",
        "ATR_Ratio",
        "Foreign_NetBuy_5D_Ratio",
        "InvestmentTrust_NetBuy_5D_Ratio",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=300)
def load_industry_rankings():
    """
    載入最新 17 個產業族群排行榜。
    """

    if not INDUSTRY_RANKING_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        INDUSTRY_RANKING_PATH,
        dtype={"Top_StockID": str},
    )

    if "Top_StockID" in dataframe.columns:
        dataframe["Top_StockID"] = (
            dataframe["Top_StockID"]
            .astype(str)
            .str.zfill(4)
        )

    numeric_columns = [
        "Industry_Rank",
        "Industry_AI_Score",
        "Highest_AI_Score",
        "Average_RSI",
        "Average_Foreign_5D_Ratio",
        "Top_Stock_AI_Score",
        "Stock_Count",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=300)
def load_update_status():
    """
    載入 GitHub Actions 最近一次更新狀態。
    """

    default_status = {
        "status": "unknown",
        "updated_at": "尚無紀錄",
        "price_data_date": "尚無紀錄",
        "institution_data_date": "尚無紀錄",
        "ranking_data_date_min": "尚無紀錄",
        "ranking_data_date_max": "尚無紀錄",
        "stock_count": 0,
        "industry_count": 0,
        "message": "目前無法讀取自動更新狀態",
    }

    if not UPDATE_STATUS_PATH.exists():
        return default_status

    try:
        with open(
            UPDATE_STATUS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            status_data = json.load(file)

        return {
            **default_status,
            **status_data,
        }

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return default_status

@st.cache_data(ttl=300)
def load_top10_snapshots():
    """
    載入每日 AI Top 10 推薦快照。
    """

    if not TOP10_SNAPSHOT_PATH.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            TOP10_SNAPSHOT_PATH,
            dtype={
                "StockID": str,
            },
        )

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()

    if "StockID" in dataframe.columns:
        dataframe["StockID"] = (
            dataframe["StockID"]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.zfill(4)
        )

    date_columns = [
        "Recommendation_Date",
        "Target_Date",
        "Completed_Date",
    ]

    for column in date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

    numeric_columns = [
        "Rank",
        "AI_Score",
        "Entry_Close",
        "Elapsed_Trading_Days",
        "Exit_Close",
        "Actual_Return_5D",
        "Weight",
        "Contribution_Return",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=300)
def load_top10_performance():
    """
    載入每日 AI Top 10 組合實績。
    """

    if not TOP10_PERFORMANCE_PATH.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            TOP10_PERFORMANCE_PATH,
        )

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()

    if "Recommendation_Date" in dataframe.columns:
        dataframe["Recommendation_Date"] = pd.to_datetime(
            dataframe["Recommendation_Date"],
            errors="coerce",
        )

    numeric_columns = [
        "Completed_Stocks",
        "Total_Stocks",
        "Gross_Return",
        "Trading_Cost",
        "Net_Return",
        "Win_Count",
        "Loss_Count",
        "Win_Rate",
        "Net_Equity",
        "Running_Max_Equity",
        "Drawdown",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe
# ==================================================
# 載入資料
# ==================================================

ranking_df = load_rankings()
industry_df = load_industry_rankings()
update_status = load_update_status()

top10_snapshot_df = load_top10_snapshots()

top10_performance_df = load_top10_performance()
# ==================================================
# 必要資料檢查
# ==================================================

if ranking_df.empty:
    st.error(
        "找不到 results/latest_rankings.csv，"
        "目前無法顯示 AI 排行榜。"
    )
    st.stop()


required_columns = [
    "StockID",
    "StockName",
    "Rank",
    "Close",
    "AI_Score",
    "Signal",
    "Risk_Level",
    "Industry",
]


missing_columns = [
    column
    for column in required_columns
    if column not in ranking_df.columns
]


if missing_columns:
    st.error(
        "AI 排行榜缺少必要欄位："
        f"{missing_columns}"
    )
    st.stop()


# ==================================================
# 首頁品牌 Hero
# ==================================================

hero_html = dedent(
    """
    <div class="ai-hero">
        <div class="ai-hero-title">
            台股 AI 智慧選股與五日趨勢預測平台
        </div>

        <div class="ai-hero-subtitle">
            整合 Yahoo Finance 股價、技術指標與
            FinMind 三大法人籌碼，運用 XGBoost
            分析個股未來 5 個交易日的相對趨勢，
            每日提供 85 檔台股 AI 排名、17 個族群
            強度與 SHAP 模型解釋。
        </div>

        <div style="margin-top: 18px;">
            <span class="ai-badge ai-badge-blue">
                85 檔股票
            </span>

            <span class="ai-badge ai-badge-green">
                17 個族群
            </span>

            <span class="ai-badge ai-badge-purple">
                32 個模型特徵
            </span>

            <span class="ai-badge ai-badge-orange">
                未來 5 個交易日
            </span>
        </div>
    </div>
    """
).strip()


st.html(hero_html)


# ==================================================
# 自動更新狀態
# ==================================================

status_value = str(
    update_status.get(
        "status",
        "unknown",
    )
).lower()


if status_value == "success":

    status_badge = (
        '<span class="ai-badge ai-badge-green">'
        "更新成功"
        "</span>"
    )

elif status_value == "initial":

    status_badge = (
        '<span class="ai-badge ai-badge-orange">'
        "等待更新"
        "</span>"
    )

else:

    status_badge = (
        '<span class="ai-badge ai-badge-red">'
        "狀態未知"
        "</span>"
    )


updated_at_raw = update_status.get(
    "updated_at",
    "尚無紀錄",
)


def format_taiwan_update_time(value):
    """
    將 GitHub Actions 儲存的 UTC 執行時間轉為台灣時間。
    """

    if value in [None, "", "尚無紀錄"]:
        return "尚無紀錄"

    parsed_time = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed_time):
        return str(value)

    taiwan_time = parsed_time.tz_convert(
        "Asia/Taipei"
    )

    return taiwan_time.strftime(
        "%Y-%m-%d %H:%M:%S 台灣時間"
    )


updated_at = format_taiwan_update_time(
    updated_at_raw
)

price_data_date = update_status.get(
    "price_data_date",
    "尚無紀錄",
)

institution_data_date = update_status.get(
    "institution_data_date",
    "尚無紀錄",
)

ranking_data_date = update_status.get(
    "ranking_data_date_max",
    institution_data_date,
)

update_message = update_status.get(
    "message",
    "尚無更新訊息",
)


status_html = dedent(
    f"""
    <div class="ai-card">
        <div class="ai-card-title">
            自動更新狀態
            {status_badge}
        </div>

        <div class="ai-card-text">
            最後執行時間：
            <strong>{updated_at}</strong>
            <br><br>

            股價資料日期：
            <strong>{price_data_date}</strong>

            &nbsp;&nbsp;｜&nbsp;&nbsp;

            法人資料日期：
            <strong>{institution_data_date}</strong>

            &nbsp;&nbsp;｜&nbsp;&nbsp;

            排行榜資料日期：
            <strong>{ranking_data_date}</strong>
            <br><br>

            更新訊息：
            <strong>{update_message}</strong>
        </div>
    </div>
    """
).strip()


st.html(status_html)


# ==================================================
# 首頁 KPI 卡片
# ==================================================

# ==================================================
# AI 排名第一
# ==================================================

top_stock = (
    ranking_df
    .sort_values(
        "Rank",
        ascending=True,
    )
    .iloc[0]
)


top_stock_id = str(
    top_stock["StockID"]
)


top_stock_name = str(
    top_stock["StockName"]
)


top_stock_score = pd.to_numeric(
    top_stock["AI_Score"],
    errors="coerce",
)


top_stock_industry = str(
    top_stock.get(
        "Industry",
        "未分類",
    )
)


top_stock_signal = str(
    top_stock.get(
        "Signal",
        "資料不足",
    )
)


top_stock_risk = str(
    top_stock.get(
        "Risk_Level",
        "資料不足",
    )
)


if pd.isna(top_stock_score):
    top_stock_score_text = "資料不足"

else:
    top_stock_score_text = (
        f"{top_stock_score:.2f}"
    )


top_stock_html = f"""
<div style="
    width: 100%;
    box-sizing: border-box;

    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 28px;
    flex-wrap: wrap;

    padding: 24px 28px;
    margin-top: 4px;
    margin-bottom: 24px;

    background:
        linear-gradient(
            145deg,
            #ffffff 0%,
            #f5f9ff 100%
        );

    border: 1px solid #d7e3f0;
    border-left: 6px solid #2563eb;
    border-radius: 18px;

    box-shadow:
        0 6px 20px
        rgba(30, 64, 175, 0.10);
">

    <div style="
        min-width: 240px;
    ">

        <div style="
            color: #5f6f85;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 8px;
        ">
            AI 排名第一
        </div>

        <div style="
            color: #172033;
            font-size: 2.1rem;
            font-weight: 850;
            line-height: 1.25;
        ">
            {top_stock_name}
        </div>

        <div style="
            color: #2563eb;
            font-size: 1.05rem;
            font-weight: 750;
            margin-top: 6px;
        ">
            {top_stock_id}
            ｜{top_stock_industry}
        </div>

    </div>

    <div style="
        min-width: 180px;
        text-align: center;
        padding-left: 24px;
        padding-right: 24px;
        border-left: 1px solid #dce5ef;
        border-right: 1px solid #dce5ef;
    ">

        <div style="
            color: #5f6f85;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 6px;
        ">
            AI 分數
        </div>

        <div style="
            color: #1d4ed8;
            font-size: 2.4rem;
            font-weight: 900;
            line-height: 1.15;
        ">
            {top_stock_score_text}
        </div>

        <div style="
            display: inline-block;
            margin-top: 8px;
            padding: 5px 12px;

            color: #166534;
            background: #dcfce7;

            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 750;
        ">
            {top_stock_signal}
        </div>

    </div>

    <div style="
        min-width: 220px;
        color: #5f6f85;
        font-size: 0.93rem;
        line-height: 1.8;
    ">

        <div>
            <strong style="color: #172033;">
                全市場排名：
            </strong>
            第 1 名
        </div>

        <div>
            <strong style="color: #172033;">
                產業族群：
            </strong>
            {top_stock_industry}
        </div>

        <div>
            <strong style="color: #172033;">
                風險等級：
            </strong>
            {top_stock_risk}
        </div>

    </div>

</div>
"""


st.html(
    top_stock_html
)


st.divider()


# ==================================================
# 今日 AI Top 10
# ==================================================

st.subheader("今日 AI Top 10")

st.caption(
    "依 XGBoost AI 分數由高至低排序。"
    "AI 分數主要用於 85 檔股票的相對比較，"
    "不等於真實上漲機率。"
)


top_n = min(
    10,
    len(ranking_df),
)


top10_df = (
    ranking_df
    .sort_values("Rank")
    .head(top_n)
    .copy()
)





# ==================================================
# Top 10 表格
# ==================================================

top10_display_columns = [
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
        "RSI",
        "Volume_Ratio",
    ]
    if column in top10_df.columns
]


top10_column_config = {
    "Rank": st.column_config.NumberColumn(
        "排名",
        format="%d",
    ),

    "StockID": st.column_config.TextColumn(
        "股票代碼",
    ),

    "StockName": st.column_config.TextColumn(
        "股票名稱",
    ),

    "Industry": st.column_config.TextColumn(
        "產業族群",
    ),

    "Close": st.column_config.NumberColumn(
        "收盤價",
        format="%.2f",
    ),

    "AI_Score": st.column_config.ProgressColumn(
        "AI 分數",
        min_value=0,
        max_value=100,
        format="%.2f",
    ),

    "Signal": st.column_config.TextColumn(
        "模型訊號",
    ),

    "Risk_Level": st.column_config.TextColumn(
        "風險等級",
    ),

    "RSI": st.column_config.NumberColumn(
        "RSI",
        format="%.2f",
    ),

    "Volume_Ratio": (
        st.column_config.NumberColumn(
            "成交量比率",
            format="%.2f",
        )
    ),
}


st.dataframe(
    top10_df[
        top10_display_columns
    ],
    use_container_width=True,
    hide_index=True,
    column_config=top10_column_config,
)


st.divider()


# ==================================================
# 85 檔完整排行榜
# ==================================================

st.subheader("85 檔完整 AI 排行榜")


filter_col1, filter_col2 = st.columns(
    [1, 2]
)


industry_options = [
    "全部族群"
] + sorted(
    ranking_df["Industry"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


selected_industry = filter_col1.selectbox(
    "產業族群",
    options=industry_options,
)


stock_search = filter_col2.text_input(
    "搜尋股票",
    placeholder="輸入股票代碼或股票名稱",
)


filtered_df = ranking_df.copy()


if selected_industry != "全部族群":

    filtered_df = filtered_df[
        filtered_df["Industry"]
        == selected_industry
    ].copy()


if stock_search.strip():

    search_value = (
        stock_search
        .strip()
        .lower()
    )

    search_condition = (
        filtered_df["StockID"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_value,
            na=False,
            regex=False,
        )
        |
        filtered_df["StockName"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_value,
            na=False,
            regex=False,
        )
    )

    filtered_df = filtered_df[
        search_condition
    ].copy()


filtered_df = (
    filtered_df
    .sort_values("Rank")
    .reset_index(drop=True)
)


st.caption(
    f"目前顯示 {len(filtered_df)} 檔股票"
)


ranking_display_columns = [
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
        "RSI",
        "Volume_Ratio",
        "ATR_Ratio",
    ]
    if column in filtered_df.columns
]


ranking_column_config = {
    "Rank": st.column_config.NumberColumn(
        "排名",
        format="%d",
    ),

    "StockID": st.column_config.TextColumn(
        "股票代碼",
    ),

    "StockName": st.column_config.TextColumn(
        "股票名稱",
    ),

    "Industry": st.column_config.TextColumn(
        "產業族群",
    ),

    "Close": st.column_config.NumberColumn(
        "收盤價",
        format="%.2f",
    ),

    "AI_Score": st.column_config.ProgressColumn(
        "AI 分數",
        min_value=0,
        max_value=100,
        format="%.2f",
    ),

    "Signal": st.column_config.TextColumn(
        "模型訊號",
    ),

    "Risk_Level": st.column_config.TextColumn(
        "風險等級",
    ),

    "RSI": st.column_config.NumberColumn(
        "RSI",
        format="%.2f",
    ),

    "Volume_Ratio": (
        st.column_config.NumberColumn(
            "成交量比率",
            format="%.2f",
        )
    ),

    "ATR_Ratio": (
        st.column_config.NumberColumn(
            "ATR 風險比例",
            format="%.2f%%",
        )
    ),
}


st.dataframe(
    filtered_df[
        ranking_display_columns
    ],
    use_container_width=True,
    hide_index=True,
    column_config=ranking_column_config,
)


st.divider()


# ==================================================
# 族群 AI 強度前五強
# ==================================================

st.subheader("族群 AI 強度前五強")

st.caption(
    "依各族群成分股的平均 AI 分數排序，"
    "顯示目前相對強度最高的五個族群。"
)


if industry_df.empty:

    st.info(
        "目前無法載入產業族群排行榜。"
    )

elif (
    "Industry" not in industry_df.columns
    or "Industry_AI_Score"
    not in industry_df.columns
):

    st.info(
        "族群排行榜缺少必要欄位。"
    )

else:

    top5_industry_df = (
        industry_df
        .dropna(
            subset=[
                "Industry",
                "Industry_AI_Score",
            ]
        )
        .sort_values(
            "Industry_AI_Score",
            ascending=False,
        )
        .head(5)
        .reset_index(drop=True)
        .copy()
    )

    top5_industry_df["族群排名"] = range(
        1,
        len(top5_industry_df) + 1,
    )

    top5_industry_df["族群名稱"] = (
        top5_industry_df["Industry"]
        .astype(str)
    )

    top5_industry_df["平均 AI 分數"] = (
        pd.to_numeric(
            top5_industry_df[
                "Industry_AI_Score"
            ],
            errors="coerce",
        )
    )

    if "Top_StockID" in top5_industry_df.columns:

        top5_industry_df["代表股代碼"] = (
            top5_industry_df["Top_StockID"]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.zfill(4)
        )

    else:

        top5_industry_df["代表股代碼"] = (
            "資料不足"
        )

    if "Top_StockName" in top5_industry_df.columns:

        top5_industry_df["代表股票"] = (
            top5_industry_df[
                "Top_StockName"
            ]
            .fillna("資料不足")
            .astype(str)
        )

    else:

        top5_industry_df["代表股票"] = (
            "資料不足"
        )

    if (
        "Top_Stock_AI_Score"
        in top5_industry_df.columns
    ):

        top5_industry_df[
            "代表股 AI 分數"
        ] = pd.to_numeric(
            top5_industry_df[
                "Top_Stock_AI_Score"
            ],
            errors="coerce",
        )

    elif (
        "Highest_AI_Score"
        in top5_industry_df.columns
    ):

        top5_industry_df[
            "代表股 AI 分數"
        ] = pd.to_numeric(
            top5_industry_df[
                "Highest_AI_Score"
            ],
            errors="coerce",
        )

    else:

        top5_industry_df[
            "代表股 AI 分數"
        ] = float("nan")

    display_top5_industry_df = (
        top5_industry_df[
            [
                "族群排名",
                "族群名稱",
                "平均 AI 分數",
                "代表股代碼",
                "代表股票",
                "代表股 AI 分數",
            ]
        ]
        .copy()
    )

    top5_industry_column_config = {
        "族群排名": (
            st.column_config.NumberColumn(
                "排名",
                format="%d",
                width="small",
            )
        ),

        "族群名稱": (
            st.column_config.TextColumn(
                "產業族群",
                width="medium",
            )
        ),

        "平均 AI 分數": (
            st.column_config.ProgressColumn(
                "族群平均 AI 分數",
                min_value=0,
                max_value=100,
                format="%.2f",
                width="large",
            )
        ),

        "代表股代碼": (
            st.column_config.TextColumn(
                "代表股代碼",
                width="small",
            )
        ),

        "代表股票": (
            st.column_config.TextColumn(
                "代表股票",
                width="medium",
            )
        ),

        "代表股 AI 分數": (
            st.column_config.NumberColumn(
                "代表股 AI 分數",
                format="%.2f",
                width="medium",
            )
        ),
    }

    st.dataframe(
        display_top5_industry_df,
        width="stretch",
        hide_index=True,
        column_config=(
            top5_industry_column_config
        ),
    )

    if not top5_industry_df.empty:

        top_industry = (
            top5_industry_df.iloc[0]
        )

        top_industry_name = (
            top_industry["族群名稱"]
        )

        top_industry_score = (
            top_industry["平均 AI 分數"]
        )

        st.success(
            "目前族群 AI 強度第一名為"
            f"「{top_industry_name}」，"
            "族群平均 AI 分數為 "
            f"{top_industry_score:.2f}。"
        )


st.divider()


  
# ==================================================
# AI Top 10 上線實績追蹤
# ==================================================

st.subheader(
    "AI Top 10 上線實績追蹤"
)

st.caption(
    "系統每日保存 AI 排名前 10 名股票，"
    "等待後續第 5 個實際交易日完成後，"
    "自動回填真實五日報酬率。"
)


if (
    top10_snapshot_df.empty
    or top10_performance_df.empty
):

    st.info(
        "目前尚無 AI Top 10 實績追蹤資料。"
        "每日更新流程執行後，"
        "系統將開始累積推薦紀錄。"
    )

else:

    valid_performance_df = (
        top10_performance_df
        .dropna(
            subset=[
                "Recommendation_Date",
            ]
        )
        .sort_values(
            "Recommendation_Date",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    if valid_performance_df.empty:

        st.info(
            "目前尚無有效的 Top 10 實績資料。"
        )

    else:

        latest_performance = (
            valid_performance_df.iloc[0]
        )

        latest_recommendation_date = (
            latest_performance[
                "Recommendation_Date"
            ]
        )

        latest_status = str(
            latest_performance.get(
                "Status",
                "觀察中",
            )
        )

        completed_stocks = pd.to_numeric(
            latest_performance.get(
                "Completed_Stocks",
                0,
            ),
            errors="coerce",
        )

        total_stocks = pd.to_numeric(
            latest_performance.get(
                "Total_Stocks",
                10,
            ),
            errors="coerce",
        )

        gross_return = pd.to_numeric(
            latest_performance.get(
                "Gross_Return",
                float("nan"),
            ),
            errors="coerce",
        )

        net_return = pd.to_numeric(
            latest_performance.get(
                "Net_Return",
                float("nan"),
            ),
            errors="coerce",
        )

        latest_snapshot_df = (
            top10_snapshot_df[
                top10_snapshot_df[
                    "Recommendation_Date"
                ]
                == latest_recommendation_date
            ]
            .sort_values("Rank")
            .copy()
        )

        if (
            not latest_snapshot_df.empty
            and "Elapsed_Trading_Days"
            in latest_snapshot_df.columns
        ):

            elapsed_days = pd.to_numeric(
                latest_snapshot_df[
                    "Elapsed_Trading_Days"
                ],
                errors="coerce",
            ).max()

        else:

            elapsed_days = 0

        if pd.isna(elapsed_days):
            elapsed_days = 0

        if pd.isna(completed_stocks):
            completed_stocks = 0

        if pd.isna(total_stocks):
            total_stocks = 10

        recommendation_date_text = (
            latest_recommendation_date.strftime(
                "%Y-%m-%d"
            )
        )

        if latest_status == "已完成":

            status_color = "#047857"
            status_background = "#dcfce7"
            status_border = "#10b981"

        elif latest_status == "資料不足":

            status_color = "#b45309"
            status_background = "#fef3c7"
            status_border = "#f59e0b"

        else:

            status_color = "#1d4ed8"
            status_background = "#dbeafe"
            status_border = "#2563eb"

        status_html = f"""
        <div style="
            width: 100%;
            box-sizing: border-box;
            padding: 20px 24px;
            margin-bottom: 18px;
            background: {status_background};
            border: 1px solid {status_border};
            border-left: 6px solid {status_border};
            border-radius: 15px;
        ">

            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 18px;
                flex-wrap: wrap;
            ">

                <div>
                    <div style="
                        color: #5f6f85;
                        font-size: 0.90rem;
                        font-weight: 700;
                    ">
                        最新推薦日期
                    </div>

                    <div style="
                        color: #172033;
                        font-size: 1.35rem;
                        font-weight: 850;
                        margin-top: 5px;
                    ">
                        {recommendation_date_text}
                    </div>
                </div>

                <div style="
                    display: inline-block;
                    padding: 8px 16px;
                    color: {status_color};
                    background: #ffffff;
                    border: 1px solid {status_border};
                    border-radius: 999px;
                    font-size: 0.95rem;
                    font-weight: 800;
                ">
                    {latest_status}
                </div>

                <div>
                    <div style="
                        color: #5f6f85;
                        font-size: 0.90rem;
                        font-weight: 700;
                    ">
                        交易日進度
                    </div>

                    <div style="
                        color: #172033;
                        font-size: 1.35rem;
                        font-weight: 850;
                        margin-top: 5px;
                    ">
                        {int(elapsed_days)} / 5
                    </div>
                </div>

                <div>
                    <div style="
                        color: #5f6f85;
                        font-size: 0.90rem;
                        font-weight: 700;
                    ">
                        完成股票數
                    </div>

                    <div style="
                        color: #172033;
                        font-size: 1.35rem;
                        font-weight: 850;
                        margin-top: 5px;
                    ">
                        {int(completed_stocks)}
                        /
                        {int(total_stocks)}
                    </div>
                </div>

            </div>
        </div>
        """

        st.html(
            status_html
        )

        if pd.notna(gross_return):

            gross_return_text = (
                f"{gross_return:.2%}"
            )

        else:

            gross_return_text = "觀察中"

        if pd.notna(net_return):

            net_return_text = (
                f"{net_return:.2%}"
            )

        else:

            net_return_text = "觀察中"

        completed_period_count = int(
            (
                top10_performance_df[
                    "Status"
                ]
                == "已完成"
            ).sum()
        )

        metric_col1, metric_col2, metric_col3 = (
            st.columns(
                3,
                gap="medium",
            )
        )

        metric_col1.metric(
            "Top 10 平均毛報酬",
            gross_return_text,
        )

        metric_col2.metric(
            "成本後淨報酬",
            net_return_text,
            help=(
                "Top 10 平均毛報酬扣除"
                "完整交易成本 0.60%。"
            ),
        )

        metric_col3.metric(
            "已完成追蹤期數",
            completed_period_count,
        )

        if not latest_snapshot_df.empty:

            snapshot_display_df = (
                latest_snapshot_df.copy()
            )

            snapshot_display_df[
                "交易日進度"
            ] = (
                pd.to_numeric(
                    snapshot_display_df[
                        "Elapsed_Trading_Days"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .astype(str)
                + " / 5"
            )

            if (
                "Actual_Return_5D"
                in snapshot_display_df.columns
            ):

                snapshot_display_df[
                    "實際五日報酬率"
                ] = snapshot_display_df[
                    "Actual_Return_5D"
                ].apply(
                    lambda value: (
                        f"{value:.2%}"
                        if pd.notna(value)
                        else "觀察中"
                    )
                )

            else:

                snapshot_display_df[
                    "實際五日報酬率"
                ] = "觀察中"

            display_columns = [
                column
                for column in [
                    "Rank",
                    "StockID",
                    "StockName",
                    "Industry",
                    "AI_Score",
                    "Entry_Close",
                    "Status",
                    "交易日進度",
                    "Target_Date",
                    "Exit_Close",
                    "實際五日報酬率",
                ]
                if column
                in snapshot_display_df.columns
            ]

            st.dataframe(
                snapshot_display_df[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": (
                        st.column_config.NumberColumn(
                            "推薦排名",
                            format="%d",
                        )
                    ),

                    "StockID": (
                        st.column_config.TextColumn(
                            "股票代碼",
                        )
                    ),

                    "StockName": (
                        st.column_config.TextColumn(
                            "股票名稱",
                        )
                    ),

                    "Industry": (
                        st.column_config.TextColumn(
                            "產業族群",
                        )
                    ),

                    "AI_Score": (
                        st.column_config.NumberColumn(
                            "AI 分數",
                            format="%.2f",
                        )
                    ),

                    "Entry_Close": (
                        st.column_config.NumberColumn(
                            "推薦日收盤價",
                            format="%.2f",
                        )
                    ),

                    "Status": (
                        st.column_config.TextColumn(
                            "追蹤狀態",
                        )
                    ),

                    "交易日進度": (
                        st.column_config.TextColumn(
                            "交易日進度",
                        )
                    ),

                    "Target_Date": (
                        st.column_config.DateColumn(
                            "第 5 個交易日",
                            format="YYYY-MM-DD",
                        )
                    ),

                    "Exit_Close": (
                        st.column_config.NumberColumn(
                            "完成日收盤價",
                            format="%.2f",
                        )
                    ),

                    "實際五日報酬率": (
                        st.column_config.TextColumn(
                            "實際五日報酬率",
                        )
                    ),
                },
            )

        with st.expander(
            "查看歷史 AI Top 10 實績"
        ):

            history_display_df = (
                valid_performance_df.copy()
            )

            for source_column, display_column in [
                (
                    "Gross_Return",
                    "平均毛報酬",
                ),
                (
                    "Trading_Cost",
                    "交易成本",
                ),
                (
                    "Net_Return",
                    "成本後淨報酬",
                ),
                (
                    "Win_Rate",
                    "個股勝率",
                ),
                (
                    "Drawdown",
                    "回撤",
                ),
            ]:

                if (
                    source_column
                    in history_display_df.columns
                ):

                    history_display_df[
                        display_column
                    ] = history_display_df[
                        source_column
                    ].apply(
                        lambda value: (
                            f"{value:.2%}"
                            if pd.notna(value)
                            else "觀察中"
                        )
                    )

            history_columns = [
                column
                for column in [
                    "Recommendation_Date",
                    "Status",
                    "Completed_Stocks",
                    "Total_Stocks",
                    "平均毛報酬",
                    "交易成本",
                    "成本後淨報酬",
                    "Win_Count",
                    "Loss_Count",
                    "個股勝率",
                    "Net_Equity",
                    "回撤",
                ]
                if column
                in history_display_df.columns
            ]

            st.dataframe(
                history_display_df[
                    history_columns
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Recommendation_Date": (
                        st.column_config.DateColumn(
                            "推薦日期",
                            format="YYYY-MM-DD",
                        )
                    ),

                    "Status": (
                        st.column_config.TextColumn(
                            "狀態",
                        )
                    ),

                    "Completed_Stocks": (
                        st.column_config.NumberColumn(
                            "完成股票數",
                            format="%d",
                        )
                    ),

                    "Total_Stocks": (
                        st.column_config.NumberColumn(
                            "推薦股票數",
                            format="%d",
                        )
                    ),

                    "Win_Count": (
                        st.column_config.NumberColumn(
                            "上漲檔數",
                            format="%d",
                        )
                    ),

                    "Loss_Count": (
                        st.column_config.NumberColumn(
                            "下跌檔數",
                            format="%d",
                        )
                    ),

                    "Net_Equity": (
                        st.column_config.NumberColumn(
                            "累積資產",
                            format="%.4f",
                        )
                    ),
                },
            )


st.divider()

# ==================================================
# AI 分數說明
# ==================================================



st.subheader("AI 分數如何解讀？")


score_explanation_html = dedent(
    """
    <div class="ai-card">
        <div class="ai-card-title">
            AI 分數為相對排序分數
        </div>

        <div class="ai-card-text">
            AI 分數由 XGBoost 根據 32 個技術面、
            量價面與法人籌碼特徵產生，主要用於
            85 檔股票之間的相對比較。
            分數越高，代表目前特徵組合越接近
            歷史上「未來 5 個交易日報酬超過 1%」
            的樣本。
        </div>

        <div style="margin-top: 16px;">
            <span class="ai-badge ai-badge-red">
                70 分以上：高分候選
            </span>

            <span class="ai-badge ai-badge-orange">
                60 至 70 分：偏多觀察
            </span>

            <span class="ai-badge ai-badge-blue">
                50 至 60 分：中性觀察
            </span>

            <span class="ai-badge ai-badge-green">
                50 分以下：暫不列入
            </span>
        </div>
    </div>
    """
).strip()


st.html(score_explanation_html)


st.info(
    "AI 分數尚未經過機率校準，"
    "例如 AI 分數為 70，"
    "不代表未來有 70% 的上漲機率。"
)


st.warning(
    "本平台僅供課程專題、資料分析與模型研究，"
    "不構成投資建議、買賣推薦或獲利保證。"
)


# ==================================================
# 共用頁尾
# ==================================================

render_footer()