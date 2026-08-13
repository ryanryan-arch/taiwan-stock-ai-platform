import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from utils.ui import (  # noqa: E402
    load_global_css,
    render_footer,
    render_sidebar_info,
)


RESULT_DIR = PROJECT_ROOT / "results"

RANKING_PATH = (
    RESULT_DIR
    / "latest_rankings.csv"
)

INDUSTRY_RANKING_PATH = (
    RESULT_DIR
    / "industry_rankings.csv"
)


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="族群排行",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        dtype={
            "StockID": str,
        },
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
    載入最新 17 個族群排行榜。
    """

    if not INDUSTRY_RANKING_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        INDUSTRY_RANKING_PATH,
        dtype={
            "Top_StockID": str,
        },
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
        "Stock_Count",
        "Top_Stock_AI_Score",
    ]

    for column in numeric_columns:

        if column in dataframe.columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


# ==================================================
# 顯示格式函式
# ==================================================

def format_number(
    value,
    number_format="{:.2f}",
    default="資料不足",
):
    """
    安全格式化一般數值。
    """

    if pd.isna(value):
        return default

    try:
        return number_format.format(value)

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def format_percentage(
    value,
    default="資料不足",
):
    """
    安全格式化百分比。
    """

    if pd.isna(value):
        return default

    try:
        return f"{value:.2%}"

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def clean_text(
    value,
    default="資料不足",
):
    """
    清理可能為空值的文字欄位。
    """

    if pd.isna(value):
        return default

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return default

    return text


def create_industry_kpi_card(
    label,
    value,
    note,
    accent_color,
    value_font_size="1.9rem",
    allow_wrap=False,
):
    """
    建立族群頁 KPI 卡片。
    """

    if allow_wrap:

        value_overflow_style = """
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            overflow-wrap: anywhere;
            word-break: break-word;
        """

    else:

        value_overflow_style = """
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        """

    return f"""
    <div style="
        width: 100%;
        height: 200px;
        min-height: 200px;
        box-sizing: border-box;

        display: flex;
        flex-direction: column;
        justify-content: space-between;

        padding: 22px;

        background:
            linear-gradient(
                145deg,
                #ffffff 0%,
                #f8fbff 100%
            );

        border: 1px solid #dce5ef;
        border-top: 4px solid {accent_color};
        border-radius: 16px;

        box-shadow:
            0 5px 18px
            rgba(30, 64, 175, 0.09);
    ">

        <div style="
            min-height: 28px;

            color: #5f6f85;
            font-size: 0.98rem;
            font-weight: 700;

            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
            {label}
        </div>

        <div style="
            min-height: 72px;

            display: flex;
            align-items: center;

            color: #172033;
            font-size: {value_font_size};
            font-weight: 850;
            line-height: 1.18;

            {value_overflow_style}
        ">
            {value}
        </div>

        <div style="
            min-height: 32px;

            color: #66768b;
            font-size: 0.86rem;
            font-weight: 600;
            line-height: 1.5;

            overflow: hidden;
        ">
            {note}
        </div>

    </div>
    """


def create_stock_value_html(
    stock_name,
    stock_id,
):
    """
    建立股票名稱與代碼的兩行顯示。
    """

    safe_stock_name = clean_text(
        stock_name
    )

    safe_stock_id = clean_text(
        stock_id,
        default="",
    )

    stock_id_html = ""

    if safe_stock_id:

        stock_id_html = f"""
        <div style="
            margin-top: 8px;

            color: #6d28d9;
            font-size: 1rem;
            font-weight: 750;
        ">
            {safe_stock_id}
        </div>
        """

    return f"""
    <div style="
        width: 100%;
        line-height: 1.15;
    ">

        <div style="
            color: #172033;
            font-size: 1.75rem;
            font-weight: 850;
            overflow-wrap: anywhere;
        ">
            {safe_stock_name}
        </div>

        {stock_id_html}

    </div>
    """


# ==================================================
# 載入資料
# ==================================================

ranking_df = load_rankings()
industry_df = load_industry_rankings()


# ==================================================
# 核心資料檢查
# ==================================================

if ranking_df.empty:

    st.error(
        "找不到 results/latest_rankings.csv，"
        "無法載入股票排行榜。"
    )

    st.stop()


if industry_df.empty:

    st.error(
        "找不到 results/industry_rankings.csv，"
        "無法載入族群排行榜。"
    )

    st.stop()


required_ranking_columns = [
    "StockID",
    "StockName",
    "Industry",
    "Rank",
    "AI_Score",
    "Close",
]


missing_ranking_columns = [
    column
    for column in required_ranking_columns
    if column not in ranking_df.columns
]


if missing_ranking_columns:

    st.error(
        "股票排行榜缺少必要欄位："
        f"{missing_ranking_columns}"
    )

    st.stop()


required_industry_columns = [
    "Industry",
    "Industry_AI_Score",
]


missing_industry_columns = [
    column
    for column in required_industry_columns
    if column not in industry_df.columns
]


if missing_industry_columns:

    st.error(
        "族群排行榜缺少必要欄位："
        f"{missing_industry_columns}"
    )

    st.stop()


# ==================================================
# 整理族群資料
# ==================================================

industry_df = (
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
    .reset_index(drop=True)
)


if industry_df.empty:

    st.error(
        "整理後沒有可用的族群排行資料。"
    )

    st.stop()


if "Industry_Rank" not in industry_df.columns:

    industry_df["Industry_Rank"] = range(
        1,
        len(industry_df) + 1,
    )


industry_df["Industry_Rank"] = range(
    1,
    len(industry_df) + 1,
)


# ==================================================
# 頁面 Hero
# ==================================================

page_hero_html = """
<div class="ai-hero">

    <div class="ai-hero-title">
        17 產業族群 AI 強度排行
    </div>

    <div class="ai-hero-subtitle">
        計算各族群成分股的平均 AI 分數，
        比較不同產業與技術主題的相對強弱，
        並進一步查看族群內高分代表股票。
    </div>

    <div style="margin-top: 18px;">

        <span class="ai-badge ai-badge-blue">
            17 個族群
        </span>

        <span class="ai-badge ai-badge-green">
            85 檔股票
        </span>

        <span class="ai-badge ai-badge-purple">
            族群平均 AI 分數
        </span>

        <span class="ai-badge ai-badge-orange">
            Top-Down 觀察
        </span>

    </div>

</div>
"""


st.html(page_hero_html)


# ==================================================
# 資料日期
# ==================================================

latest_data_date = "日期不足"

if "date" in ranking_df.columns:

    latest_date = ranking_df["date"].max()

    if pd.notna(latest_date):

        latest_data_date = (
            latest_date.strftime(
                "%Y-%m-%d"
            )
        )


date_html = f"""
<div class="ai-card">

    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        flex-wrap: wrap;
    ">

        <div>

            <div style="
                color: #5f6f85;
                font-size: 0.95rem;
                font-weight: 700;
                margin-bottom: 6px;
            ">
                族群排行榜資料日期
            </div>

            <div style="
                color: #1e40af;
                font-size: 1.75rem;
                font-weight: 850;
                white-space: nowrap;
            ">
                {latest_data_date}
            </div>

        </div>

        <div style="
            color: #66768b;
            font-size: 0.92rem;
            line-height: 1.6;
            text-align: right;
        ">
            族群分數為該族群成分股
            最新 AI 分數的算術平均
        </div>

    </div>

</div>
"""


st.html(date_html)


# ==================================================
# 族群摘要 KPI
# ==================================================

top_industry = industry_df.iloc[0]


average_industry_score = (
    industry_df[
        "Industry_AI_Score"
    ].mean()
)


highest_industry_score = (
    industry_df[
        "Industry_AI_Score"
    ].max()
)


top_stock_name = clean_text(
    top_industry.get(
        "Top_StockName",
        "資料不足",
    )
)


top_stock_id = clean_text(
    top_industry.get(
        "Top_StockID",
        "",
    ),
    default="",
)


top_stock_score = pd.to_numeric(
    top_industry.get(
        "Top_Stock_AI_Score",
        float("nan"),
    ),
    errors="coerce",
)


top_stock_value_html = (
    create_stock_value_html(
        stock_name=top_stock_name,
        stock_id=top_stock_id,
    )
)


top_stock_note = (
    "族群內最高分股票"
)


if pd.notna(top_stock_score):

    top_stock_note = (
        "族群內最高分股票"
        f"｜{top_stock_score:.2f} 分"
    )


kpi_col1, kpi_col2, \
kpi_col3, kpi_col4 = st.columns(
    4,
    gap="medium",
)


with kpi_col1:

    st.html(
        create_industry_kpi_card(
            label="族群第一名",
            value=clean_text(
                top_industry[
                    "Industry"
                ]
            ),
            note=(
                "目前平均 AI 分數"
                "最高的族群"
            ),
            accent_color="#2563eb",
            value_font_size="1.85rem",
            allow_wrap=True,
        )
    )


with kpi_col2:

    st.html(
        create_industry_kpi_card(
            label="第一名族群分數",
            value=format_number(
                top_industry[
                    "Industry_AI_Score"
                ]
            ),
            note="族群成分股平均 AI 分數",
            accent_color="#0891b2",
        )
    )


with kpi_col3:

    st.html(
        create_industry_kpi_card(
            label="族群代表股票",
            value=top_stock_value_html,
            note=top_stock_note,
            accent_color="#6d28d9",
            value_font_size="1.7rem",
            allow_wrap=True,
        )
    )


with kpi_col4:

    st.html(
        create_industry_kpi_card(
            label="全部族群平均",
            value=format_number(
                average_industry_score
            ),
            note=(
                "最高族群分數 "
                f"{highest_industry_score:.2f}"
            ),
            accent_color="#d97706",
        )
    )


st.divider()


# ==================================================
# 17 族群平均 AI 分數圖
# ==================================================

st.subheader(
    "17 族群平均 AI 分數"
)


industry_chart_df = (
    industry_df
    .sort_values(
        "Industry_AI_Score",
        ascending=True,
    )
    .copy()
)


industry_figure = px.bar(
    industry_chart_df,
    x="Industry_AI_Score",
    y="Industry",
    orientation="h",
    color="Industry_AI_Score",
    color_continuous_scale=[
        [0.0, "#bfdbfe"],
        [0.45, "#3b82f6"],
        [1.0, "#1e40af"],
    ],
    text="Industry_AI_Score",
    labels={
        "Industry_AI_Score": (
            "族群平均 AI 分數"
        ),
        "Industry": "",
    },
)


industry_figure.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
    cliponaxis=False,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "族群平均 AI 分數："
        "%{x:.2f}"
        "<extra></extra>"
    ),
)


industry_figure.update_layout(
    height=760,
    margin={
        "l": 40,
        "r": 90,
        "t": 35,
        "b": 90,
    },
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    coloraxis_showscale=False,
    font={
        "family": (
            "Microsoft JhengHei, "
            "Noto Sans TC, Arial"
        ),
        "color": "#172033",
        "size": 14,
    },
    xaxis={
        "automargin": True,
        "title": {
            "text": (
                "族群平均 AI 分數"
            ),
            "standoff": 22,
            "font": {
                "size": 16,
                "color": "#65748b",
            },
        },
        "tickfont": {
            "size": 13,
            "color": "#65748b",
        },
        "showgrid": True,
        "gridcolor": "#edf2f7",
        "zeroline": False,
        "fixedrange": True,
    },
    yaxis={
        "automargin": True,
        "title": {
            "text": "",
        },
        "tickfont": {
            "size": 14,
            "color": "#4b5d73",
        },
        "fixedrange": True,
    },
    hoverlabel={
        "font": {
            "family": (
                "Microsoft JhengHei, "
                "Noto Sans TC, Arial"
            ),
            "size": 13,
        },
    },
)


st.plotly_chart(
    industry_figure,
    use_container_width=True,
)


st.divider()


# ==================================================
# 完整族群排行榜
# ==================================================

st.subheader(
    "完整族群排行榜"
)


industry_display_columns = [
    column
    for column in [
        "Industry_Rank",
        "Industry",
        "Industry_AI_Score",
        "Highest_AI_Score",
        "Average_RSI",
        "Average_Foreign_5D_Ratio",
        "Stock_Count",
        "Top_StockID",
        "Top_StockName",
        "Top_Stock_AI_Score",
    ]
    if column in industry_df.columns
]


industry_column_config = {
    "Industry_Rank": (
        st.column_config.NumberColumn(
            "族群排名",
            format="%d",
        )
    ),

    "Industry": (
        st.column_config.TextColumn(
            "產業族群",
        )
    ),

    "Industry_AI_Score": (
        st.column_config.ProgressColumn(
            "平均 AI 分數",
            min_value=0,
            max_value=100,
            format="%.2f",
        )
    ),

    "Highest_AI_Score": (
        st.column_config.NumberColumn(
            "族群最高分",
            format="%.2f",
        )
    ),

    "Average_RSI": (
        st.column_config.NumberColumn(
            "平均 RSI",
            format="%.2f",
        )
    ),

    "Average_Foreign_5D_Ratio": (
        st.column_config.NumberColumn(
            "平均外資 5 日比例",
            format="%.2f%%",
        )
    ),

    "Stock_Count": (
        st.column_config.NumberColumn(
            "股票數",
            format="%d",
        )
    ),

    "Top_StockID": (
        st.column_config.TextColumn(
            "代表股代碼",
        )
    ),

    "Top_StockName": (
        st.column_config.TextColumn(
            "代表股票",
        )
    ),

    "Top_Stock_AI_Score": (
        st.column_config.NumberColumn(
            "代表股 AI 分數",
            format="%.2f",
        )
    ),
}


st.dataframe(
    industry_df[
        industry_display_columns
    ],
    use_container_width=True,
    hide_index=True,
    column_config=industry_column_config,
)


st.divider()


# ==================================================
# 選擇族群
# ==================================================

st.subheader(
    "族群內股票排名"
)


industry_options = (
    industry_df["Industry"]
    .dropna()
    .astype(str)
    .tolist()
)


selected_industry = st.selectbox(
    "選擇產業族群",
    options=industry_options,
)


selected_industry_df = industry_df[
    industry_df["Industry"]
    == selected_industry
].copy()


if selected_industry_df.empty:

    st.info(
        "目前找不到所選族群的摘要資料。"
    )

else:

    selected_industry_row = (
        selected_industry_df.iloc[0]
    )


    selected_stock_df = (
        ranking_df[
            ranking_df["Industry"]
            == selected_industry
        ]
        .sort_values(
            "AI_Score",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )


    selected_stock_df[
        "族群內排名"
    ] = range(
        1,
        len(selected_stock_df) + 1,
    )


    detail_col1, detail_col2, \
    detail_col3 = st.columns(
        3,
        gap="medium",
    )


    with detail_col1:

        st.html(
            create_industry_kpi_card(
                label="所選族群",
                value=selected_industry,
                note=(
                    "族群排名第 "
                    f'{int(selected_industry_row["Industry_Rank"])} '
                    "名"
                ),
                accent_color="#2563eb",
                value_font_size="1.8rem",
                allow_wrap=True,
            )
        )


    with detail_col2:

        st.html(
            create_industry_kpi_card(
                label="族群平均 AI 分數",
                value=format_number(
                    selected_industry_row[
                        "Industry_AI_Score"
                    ]
                ),
                note=(
                    "共有 "
                    f"{len(selected_stock_df)} "
                    "檔股票"
                ),
                accent_color="#0891b2",
            )
        )


    if not selected_stock_df.empty:

        selected_top_stock = (
            selected_stock_df.iloc[0]
        )


        selected_top_stock_html = (
            create_stock_value_html(
                stock_name=(
                    selected_top_stock[
                        "StockName"
                    ]
                ),
                stock_id=(
                    selected_top_stock[
                        "StockID"
                    ]
                ),
            )
        )


        selected_top_stock_note = (
            "AI 分數 "
            f'{selected_top_stock["AI_Score"]:.2f}'
        )

    else:

        selected_top_stock_html = (
            "資料不足"
        )

        selected_top_stock_note = (
            "目前沒有族群成分股"
        )


    with detail_col3:

        st.html(
            create_industry_kpi_card(
                label="族群最高分股票",
                value=(
                    selected_top_stock_html
                ),
                note=(
                    selected_top_stock_note
                ),
                accent_color="#6d28d9",
                value_font_size="1.7rem",
                allow_wrap=True,
            )
        )


    if selected_stock_df.empty:

        st.info(
            "目前沒有這個族群的股票資料。"
        )

    else:

        st.markdown(
            "### 族群成分股 AI 分數"
        )


        stock_chart_df = (
            selected_stock_df
            .sort_values(
                "AI_Score",
                ascending=True,
            )
            .copy()
        )


        stock_chart_df["股票"] = (
            stock_chart_df["StockID"]
            + " "
            + stock_chart_df["StockName"]
        )


        stock_figure = px.bar(
            stock_chart_df,
            x="AI_Score",
            y="股票",
            orientation="h",
            color="AI_Score",
            color_continuous_scale=[
                [0.0, "#99f6e4"],
                [0.5, "#14b8a6"],
                [1.0, "#0f766e"],
            ],
            text="AI_Score",
            labels={
                "AI_Score": "AI 分數",
                "股票": "",
            },
        )


        stock_figure.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "AI 分數：%{x:.2f}"
                "<extra></extra>"
            ),
        )


        chart_height = max(
            440,
            len(stock_chart_df) * 66,
        )


        stock_figure.update_layout(
            height=chart_height,
            margin={
                "l": 35,
                "r": 90,
                "t": 35,
                "b": 85,
            },
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            coloraxis_showscale=False,
            font={
                "family": (
                    "Microsoft JhengHei, "
                    "Noto Sans TC, Arial"
                ),
                "color": "#172033",
                "size": 14,
            },
            xaxis={
                "automargin": True,
                "title": {
                    "text": "AI 分數",
                    "standoff": 20,
                },
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
            },
            yaxis={
                "automargin": True,
                "title": {
                    "text": "",
                },
            },
        )


        st.plotly_chart(
            stock_figure,
            use_container_width=True,
        )


        st.markdown(
            "### 族群成分股明細"
        )


        selected_stock_columns = [
            column
            for column in [
                "族群內排名",
                "Rank",
                "StockID",
                "StockName",
                "Close",
                "AI_Score",
                "Signal",
                "Risk_Level",
                "RSI",
                "Volume_Ratio",
            ]
            if column
            in selected_stock_df.columns
        ]


        stock_column_config = {
            "族群內排名": (
                st.column_config.NumberColumn(
                    "族群內排名",
                    format="%d",
                )
            ),

            "Rank": (
                st.column_config.NumberColumn(
                    "全市場排名",
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

            "Close": (
                st.column_config.NumberColumn(
                    "收盤價",
                    format="%.2f",
                )
            ),

            "AI_Score": (
                st.column_config.ProgressColumn(
                    "AI 分數",
                    min_value=0,
                    max_value=100,
                    format="%.2f",
                )
            ),

            "Signal": (
                st.column_config.TextColumn(
                    "模型訊號",
                )
            ),

            "Risk_Level": (
                st.column_config.TextColumn(
                    "風險等級",
                )
            ),

            "RSI": (
                st.column_config.NumberColumn(
                    "RSI",
                    format="%.2f",
                )
            ),

            "Volume_Ratio": (
                st.column_config.NumberColumn(
                    "成交量比率",
                    format="%.2f",
                )
            ),
        }


        st.dataframe(
            selected_stock_df[
                selected_stock_columns
            ],
            use_container_width=True,
            hide_index=True,
            column_config=(
                stock_column_config
            ),
        )


# ==================================================
# 族群排行說明
# ==================================================

st.divider()


explanation_html = """
<div class="ai-card">

    <div class="ai-card-title">
        族群 AI 強度如何解讀？
    </div>

    <div class="ai-card-text">

        <strong>族群平均 AI 分數：</strong>
        將同一族群內所有股票的最新 AI 分數
        取算術平均，用來比較各族群的相對強弱。
        <br><br>

        <strong>族群代表股票：</strong>
        為該族群中目前 AI 分數最高的股票，
        不代表一定會上漲，也不構成買進建議。
        <br><br>

        <strong>Top-Down 觀察：</strong>
        可先觀察相對高分族群，再查看族群內
        股票的排名、技術指標及法人籌碼狀態。
        <br><br>

        族群分數會隨每日股票特徵與模型推論結果
        改變，只適合用於相對比較。

    </div>

</div>
"""


st.html(explanation_html)


st.warning(
    "族群 AI 強度為模型輸出的相對排序結果，"
    "不代表族群未來必然上漲，"
    "也不構成投資建議或買賣推薦。"
)


# ==================================================
# 共用頁尾
# ==================================================

render_footer()