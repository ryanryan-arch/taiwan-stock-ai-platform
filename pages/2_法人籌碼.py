import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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

RAW_PRICE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_price"
)

RAW_INSTITUTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_institution"
)

RANKING_PATH = (
    RESULT_DIR
    / "latest_rankings.csv"
)

LATEST_FEATURE_PATH = (
    RESULT_DIR
    / "latest_features.parquet"
)


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="法人籌碼",
    page_icon="🏦",
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
    載入最新 AI 排行榜。
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
        "RSI",
        "Volume_Ratio",
        "ATR_Ratio",
        "Foreign_NetBuy_5D_Ratio",
        "InvestmentTrust_NetBuy_5D_Ratio",
        "Foreign_Buy_Streak",
        "InvestmentTrust_Buy_Streak",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=300)
def load_latest_features():
    """
    載入最新完整特徵資料。
    """

    if not LATEST_FEATURE_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_parquet(
        LATEST_FEATURE_PATH
    )

    if "StockID" not in dataframe.columns:
        return pd.DataFrame()

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
def load_institution_history(stock_id):
    """
    載入指定股票的法人歷史資料。
    """

    possible_paths = [
        RAW_INSTITUTION_DIR
        / f"{stock_id}_institution.parquet",

        RAW_INSTITUTION_DIR
        / f"{stock_id}.parquet",
    ]

    institution_path = None

    for path in possible_paths:
        if path.exists():
            institution_path = path
            break

    if institution_path is None:
        return pd.DataFrame()

    dataframe = pd.read_parquet(
        institution_path
    )

    rename_map = {
        "Date": "date",
        "stock_id": "StockID",
        "ForeignInvestor": "Foreign_NetBuy",
        "InvestmentTrust": (
            "InvestmentTrust_NetBuy"
        ),
        "Dealer": "Dealer_NetBuy",
    }

    dataframe = dataframe.rename(
        columns={
            old_name: new_name
            for old_name, new_name
            in rename_map.items()
            if old_name in dataframe.columns
        }
    )

    if "date" not in dataframe.columns:
        return pd.DataFrame()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    if "StockID" not in dataframe.columns:
        dataframe["StockID"] = stock_id

    dataframe["StockID"] = (
        dataframe["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    institution_columns = [
        "Foreign_NetBuy",
        "InvestmentTrust_NetBuy",
        "Dealer_NetBuy",
        "Institutional_Total_NetBuy",
    ]

    for column in institution_columns:

        if column not in dataframe.columns:
            dataframe[column] = 0.0

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).fillna(0.0)

    dataframe[
        "Institutional_Total_NetBuy"
    ] = (
        dataframe["Foreign_NetBuy"]
        + dataframe[
            "InvestmentTrust_NetBuy"
        ]
        + dataframe["Dealer_NetBuy"]
    )

    dataframe = (
        dataframe
        .dropna(subset=["date"])
        .drop_duplicates(
            subset=["date", "StockID"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return dataframe


@st.cache_data(ttl=300)
def load_price_history(stock_id):
    """
    載入指定股票歷史收盤價。
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

    if (
        "date" not in dataframe.columns
        or "Close" not in dataframe.columns
    ):
        return pd.DataFrame()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe["Close"] = pd.to_numeric(
        dataframe["Close"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "date",
                "Close",
            ]
        )
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return dataframe


# ==================================================
# 格式化函式
# ==================================================

def format_integer(value):
    """
    將法人買賣超格式化為整數與千分位。
    """

    if pd.isna(value):
        return "資料不足"

    return f"{value:,.0f}"


def format_percentage(value):
    """
    將比例格式化為百分比。
    """

    if pd.isna(value):
        return "資料不足"

    return f"{value:.2%}"


def format_streak(value):
    """
    將連續買超天數格式化。
    """

    if pd.isna(value):
        return "資料不足"

    return f"{value:.0f} 日"


def create_kpi_card(
    label,
    value,
    note,
    accent_color,
):
    """
    建立法人籌碼 KPI 卡片。
    """

    return f"""
    <div style="
        width: 100%;
        min-height: 175px;
        height: 175px;
        box-sizing: border-box;

        display: flex;
        flex-direction: column;
        justify-content: space-between;

        padding: 20px;

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
            color: #5f6f85;
            font-size: 0.95rem;
            font-weight: 700;
        ">
            {label}
        </div>

        <div style="
            color: #172033;
            font-size: 1.85rem;
            font-weight: 850;
            line-height: 1.2;

            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
            {value}
        </div>

        <div style="
            color: #66768b;
            font-size: 0.84rem;
            font-weight: 600;
            line-height: 1.45;
        ">
            {note}
        </div>
    </div>
    """


# ==================================================
# 載入核心資料
# ==================================================

ranking_df = load_rankings()
latest_feature_df = load_latest_features()


# ==================================================
# 核心資料檢查
# ==================================================

if ranking_df.empty:

    st.error(
        "找不到 results/latest_rankings.csv，"
        "無法載入股票清單。"
    )

    st.stop()


required_ranking_columns = [
    "StockID",
    "StockName",
    "Rank",
    "AI_Score",
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
# 頁面 Hero
# ==================================================

page_hero_html = """
<div class="ai-hero">

    <div class="ai-hero-title">
        三大法人籌碼分析
    </div>

    <div class="ai-hero-subtitle">
        追蹤外資、投信與自營商的每日買賣超、
        累計籌碼與連續買超天數，
        觀察法人資金流向及個股籌碼變化。
    </div>

    <div style="margin-top: 18px;">

        <span class="ai-badge ai-badge-blue">
            外資
        </span>

        <span class="ai-badge ai-badge-red">
            投信
        </span>

        <span class="ai-badge ai-badge-green">
            自營商
        </span>

        <span class="ai-badge ai-badge-purple">
            三大法人合計
        </span>

    </div>

</div>
"""


st.html(page_hero_html)


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


if selected_ranking_df.empty:

    st.error(
        "排行榜中找不到所選股票。"
    )

    st.stop()


selected_ranking = (
    selected_ranking_df.iloc[0]
)


selected_stock_name = str(
    selected_ranking["StockName"]
)


selected_industry = str(
    selected_ranking.get(
        "Industry",
        "未分類",
    )
)


ranking_date = pd.to_datetime(
    selected_ranking.get(
        "date",
        pd.NaT,
    ),
    errors="coerce",
)


ranking_date_text = (
    ranking_date.strftime("%Y-%m-%d")
    if pd.notna(ranking_date)
    else "日期不足"
)


institution_df = load_institution_history(
    selected_stock_id
)


price_df = load_price_history(
    selected_stock_id
)


selected_feature_df = (
    latest_feature_df[
        latest_feature_df["StockID"]
        == selected_stock_id
    ].copy()
    if not latest_feature_df.empty
    else pd.DataFrame()
)


selected_features = (
    selected_feature_df.iloc[0]
    if not selected_feature_df.empty
    else pd.Series(dtype="object")
)


# ==================================================
# 股票標題與資料日期
# ==================================================

st.subheader(
    f"{selected_stock_id} "
    f"{selected_stock_name}"
)


institution_date_text = "資料不足"

if (
    not institution_df.empty
    and "date" in institution_df.columns
):

    institution_latest_date = (
        institution_df["date"].max()
    )

    if pd.notna(institution_latest_date):
        institution_date_text = (
            institution_latest_date.strftime(
                "%Y-%m-%d"
            )
        )


st.caption(
    f"產業族群：{selected_industry}"
    f"｜AI 排名：第 "
    f'{int(selected_ranking["Rank"])} 名'
    f"｜排行榜日期：{ranking_date_text}"
    f"｜法人資料日期：{institution_date_text}"
)


# ==================================================
# 最新法人 KPI
# ==================================================

latest_foreign = float("nan")
latest_trust = float("nan")
latest_dealer = float("nan")
latest_total = float("nan")


if not institution_df.empty:

    latest_institution = (
        institution_df.iloc[-1]
    )

    latest_foreign = (
        latest_institution.get(
            "Foreign_NetBuy",
            float("nan"),
        )
    )

    latest_trust = (
        latest_institution.get(
            "InvestmentTrust_NetBuy",
            float("nan"),
        )
    )

    latest_dealer = (
        latest_institution.get(
            "Dealer_NetBuy",
            float("nan"),
        )
    )

    latest_total = (
        latest_institution.get(
            "Institutional_Total_NetBuy",
            float("nan"),
        )
    )


kpi_col1, kpi_col2, \
kpi_col3, kpi_col4 = st.columns(
    4,
    gap="medium",
)


with kpi_col1:

    st.html(
        create_kpi_card(
            label="外資單日買賣超",
            value=format_integer(
                latest_foreign
            ),
            note="正值為買超，負值為賣超",
            accent_color="#2563eb",
        )
    )


with kpi_col2:

    st.html(
        create_kpi_card(
            label="投信單日買賣超",
            value=format_integer(
                latest_trust
            ),
            note="追蹤投信資金動向",
            accent_color="#dc2626",
        )
    )


with kpi_col3:

    st.html(
        create_kpi_card(
            label="自營商單日買賣超",
            value=format_integer(
                latest_dealer
            ),
            note="包含自營與避險部位",
            accent_color="#16a34a",
        )
    )


with kpi_col4:

    st.html(
        create_kpi_card(
            label="三大法人合計",
            value=format_integer(
                latest_total
            ),
            note="外資、投信與自營商合計",
            accent_color="#6d28d9",
        )
    )


st.divider()


# ==================================================
# 法人籌碼特徵
# ==================================================

st.subheader("最新法人籌碼特徵")


feature_cards = [
    (
        "外資 5 日比例",
        format_percentage(
            selected_features.get(
                "Foreign_NetBuy_5D_Ratio",
                float("nan"),
            )
        ),
        "#2563eb",
    ),
    (
        "投信 5 日比例",
        format_percentage(
            selected_features.get(
                "InvestmentTrust_NetBuy_5D_Ratio",
                float("nan"),
            )
        ),
        "#dc2626",
    ),
    (
        "外資連續買超",
        format_streak(
            selected_features.get(
                "Foreign_Buy_Streak",
                float("nan"),
            )
        ),
        "#0891b2",
    ),
    (
        "投信連續買超",
        format_streak(
            selected_features.get(
                "InvestmentTrust_Buy_Streak",
                float("nan"),
            )
        ),
        "#d97706",
    ),
]


feature_columns = st.columns(
    4,
    gap="medium",
)


for column, (
    label,
    value,
    color,
) in zip(
    feature_columns,
    feature_cards,
):

    with column:

        st.html(
            create_kpi_card(
                label=label,
                value=value,
                note="模型最新籌碼特徵",
                accent_color=color,
            )
        )


st.divider()


# ==================================================
# 法人歷史圖表
# ==================================================

st.subheader("三大法人每日買賣超")


period_options = {
    "近 20 日": 20,
    "近 60 日": 60,
    "近 120 日": 120,
    "全部資料": None,
}


selected_period = st.radio(
    "顯示期間",
    options=list(
        period_options.keys()
    ),
    horizontal=True,
)


if institution_df.empty:

    st.warning(
        "找不到這檔股票的法人歷史資料。"
    )

else:

    selected_days = period_options[
        selected_period
    ]

    if selected_days is None:
        chart_df = institution_df.copy()

    else:
        chart_df = (
            institution_df
            .tail(selected_days)
            .copy()
        )

    institution_figure = go.Figure()

    institution_figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["Foreign_NetBuy"],
            name="外資",
            marker_color="#2563eb",
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "外資買賣超：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    institution_figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df[
                "InvestmentTrust_NetBuy"
            ],
            name="投信",
            marker_color="#dc2626",
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "投信買賣超：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    institution_figure.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["Dealer_NetBuy"],
            name="自營商",
            marker_color="#16a34a",
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "自營商買賣超：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    institution_figure.update_layout(
        height=680,
        barmode="group",
        margin={
            "l": 45,
            "r": 35,
            "t": 80,
            "b": 90,
        },
        title={
            "text": (
                f"{selected_stock_id} "
                f"{selected_stock_name} "
                f"{selected_period}法人買賣超"
            ),
            "x": 0.02,
            "font": {
                "size": 20,
                "color": "#172033",
            },
        },
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
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
                "text": "日期",
                "standoff": 20,
            },
            "showgrid": False,
        },
        yaxis={
            "automargin": True,
            "title": {
                "text": "買賣超股數",
                "standoff": 18,
            },
            "showgrid": True,
            "gridcolor": "#edf2f7",
            "zeroline": True,
            "zerolinecolor": "#94a3b8",
            "zerolinewidth": 1,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
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
        institution_figure,
        use_container_width=True,
    )


st.divider()


# ==================================================
# 法人累計買賣超與股價
# ==================================================

st.subheader("法人累計買賣超與股價")


if institution_df.empty:

    st.info(
        "法人歷史資料不足，"
        "無法產生累計籌碼圖。"
    )

else:

    cumulative_df = (
        institution_df
        .sort_values("date")
        .copy()
    )

    cumulative_df[
        "外資累計"
    ] = (
        cumulative_df[
            "Foreign_NetBuy"
        ].cumsum()
    )

    cumulative_df[
        "投信累計"
    ] = (
        cumulative_df[
            "InvestmentTrust_NetBuy"
        ].cumsum()
    )

    cumulative_df[
        "自營商累計"
    ] = (
        cumulative_df[
            "Dealer_NetBuy"
        ].cumsum()
    )

    if not price_df.empty:

        cumulative_df = pd.merge(
            cumulative_df,
            price_df[
                [
                    "date",
                    "Close",
                ]
            ],
            on="date",
            how="left",
        )

    cumulative_figure = (
        go.Figure()
    )

    cumulative_figure.add_trace(
        go.Scatter(
            x=cumulative_df["date"],
            y=cumulative_df["外資累計"],
            name="外資累計",
            mode="lines",
            line={
                "color": "#2563eb",
                "width": 2,
            },
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "外資累計：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    cumulative_figure.add_trace(
        go.Scatter(
            x=cumulative_df["date"],
            y=cumulative_df["投信累計"],
            name="投信累計",
            mode="lines",
            line={
                "color": "#dc2626",
                "width": 2,
            },
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "投信累計：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    cumulative_figure.add_trace(
        go.Scatter(
            x=cumulative_df["date"],
            y=cumulative_df["自營商累計"],
            name="自營商累計",
            mode="lines",
            line={
                "color": "#16a34a",
                "width": 2,
            },
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "自營商累計：%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    if "Close" in cumulative_df.columns:

        cumulative_figure.add_trace(
            go.Scatter(
                x=cumulative_df["date"],
                y=cumulative_df["Close"],
                name="收盤價",
                mode="lines",
                yaxis="y2",
                line={
                    "color": "#111827",
                    "width": 2,
                    "dash": "dot",
                },
                hovertemplate=(
                    "日期：%{x|%Y-%m-%d}<br>"
                    "收盤價：%{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    cumulative_figure.update_layout(
        height=680,
        margin={
            "l": 45,
            "r": 65,
            "t": 80,
            "b": 90,
        },
        title={
            "text": (
                f"{selected_stock_id} "
                f"{selected_stock_name} "
                "法人累計買賣超與股價"
            ),
            "x": 0.02,
            "font": {
                "size": 20,
                "color": "#172033",
            },
        },
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
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
                "text": "日期",
                "standoff": 20,
            },
            "showgrid": False,
        },
        yaxis={
            "automargin": True,
            "title": {
                "text": "法人累計買賣超",
                "standoff": 18,
            },
            "showgrid": True,
            "gridcolor": "#edf2f7",
            "zeroline": True,
            "zerolinecolor": "#94a3b8",
        },
        yaxis2={
            "title": {
                "text": "收盤價",
                "standoff": 18,
            },
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    st.plotly_chart(
        cumulative_figure,
        use_container_width=True,
    )


st.divider()


# ==================================================
# 法人歷史資料表
# ==================================================

st.subheader("法人歷史明細")


if institution_df.empty:

    st.info(
        "目前沒有法人歷史明細。"
    )

else:

    table_df = (
        institution_df
        .sort_values(
            "date",
            ascending=False,
        )
        .copy()
    )

    table_df["date"] = (
        table_df["date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_columns = [
        "date",
        "Foreign_NetBuy",
        "InvestmentTrust_NetBuy",
        "Dealer_NetBuy",
        "Institutional_Total_NetBuy",
    ]

    table_df = table_df[
        display_columns
    ].rename(
        columns={
            "date": "日期",
            "Foreign_NetBuy": "外資",
            "InvestmentTrust_NetBuy": "投信",
            "Dealer_NetBuy": "自營商",
            "Institutional_Total_NetBuy": (
                "三大法人合計"
            ),
        }
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config={
            "日期": st.column_config.TextColumn(
                "日期",
            ),
            "外資": st.column_config.NumberColumn(
                "外資",
                format="%,.0f",
            ),
            "投信": st.column_config.NumberColumn(
                "投信",
                format="%,.0f",
            ),
            "自營商": (
                st.column_config.NumberColumn(
                    "自營商",
                    format="%,.0f",
                )
            ),
            "三大法人合計": (
                st.column_config.NumberColumn(
                    "三大法人合計",
                    format="%,.0f",
                )
            ),
        },
    )


# ==================================================
# 說明與免責聲明
# ==================================================

st.divider()


explanation_html = """
<div class="ai-card">

    <div class="ai-card-title">
        法人籌碼如何解讀？
    </div>

    <div class="ai-card-text">

        <strong>外資：</strong>
        通常交易規模較大，可能受全球資金、
        匯率與市場風險偏好影響。
        <br><br>

        <strong>投信：</strong>
        常反映國內基金的選股與持股調整，
        連續買超可作為籌碼觀察資訊。
        <br><br>

        <strong>自營商：</strong>
        包含自營與避險交易，
        單日數值可能受衍生性商品避險影響。
        <br><br>

        法人買賣超只代表特定市場參與者的
        歷史交易方向，並不保證後續股價上漲或下跌。

    </div>

</div>
"""


st.html(explanation_html)


st.warning(
    "法人籌碼資料僅供課程研究與資料分析，"
    "不構成投資建議、買賣推薦或獲利保證。"
)


# ==================================================
# 共用頁尾
# ==================================================

render_footer()