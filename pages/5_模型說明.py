from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="模型說明",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 資料載入函式
# ==================================================

@st.cache_data(ttl=300)
def load_feature_list():
    """
    載入模型使用的特徵清單。
    """

    feature_path = (
        MODEL_DIR / "model_features.pkl"
    )

    if not feature_path.exists():
        return []

    return joblib.load(feature_path)


@st.cache_data(ttl=300)
def load_stock_metadata():
    """
    載入 85 檔股票清單。
    """

    stock_path = (
        CONFIG_DIR / "stocks_85.csv"
    )

    if not stock_path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        stock_path,
        dtype={"StockID": str},
    )

    dataframe["StockID"] = (
        dataframe["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    return dataframe


@st.cache_data(ttl=300)
def load_cv_results():
    """
    載入 TimeSeriesSplit 五折驗證結果。
    """

    cv_path = (
        RESULT_DIR
        / "timeseries_cv_results.csv"
    )

    if not cv_path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(cv_path)

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=300)
def load_portfolio_kpis():
    """
    載入投資組合回測 KPI。
    """

    kpi_path = (
        RESULT_DIR / "portfolio_kpis.csv"
    )

    if not kpi_path.exists():
        return pd.DataFrame()

    return pd.read_csv(kpi_path)


# ==================================================
# 載入網站資料
# ==================================================

feature_cols = load_feature_list()
stocks_df = load_stock_metadata()
cv_df = load_cv_results()
portfolio_kpi_df = load_portfolio_kpis()


# ==================================================
# 頁面標題
# ==================================================

st.title("模型與資料說明")

st.caption(
    "說明本平台的資料來源、特徵工程、預測目標、"
    "XGBoost 模型、TimeSeriesSplit、回測設計及使用限制。"
)


# ==================================================
# 專題摘要
# ==================================================

st.subheader("專題摘要")

st.markdown(
    """
本平台整合台股歷史價格、成交量與三大法人籌碼資料，
建立技術面、量價面及法人籌碼面特徵，
並利用 XGBoost 預測個股未來 5 個交易日的價格趨勢。

系統涵蓋 17 個產業族群，共計 85 檔台股。
模型每天依據最新特徵產生 AI 分數，
再將 85 檔股票由高至低排序，
提供 AI Top 10、個股分析、法人籌碼、族群排行及策略回測功能。

模型評估採用 TimeSeriesSplit，
確保訓練資料的日期早於驗證資料，
並使用 OOF 樣本外預測進行 AI Top 5 投資組合回測。
"""
)


stock_count = (
    len(stocks_df)
    if not stocks_df.empty
    else 0
)

industry_count = (
    stocks_df["Industry"].nunique()
    if (
        not stocks_df.empty
        and "Industry" in stocks_df.columns
    )
    else 0
)

feature_count = len(feature_cols)


summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

summary_col1.metric(
    "股票數量",
    stock_count,
)

summary_col2.metric(
    "產業族群",
    industry_count,
)

summary_col3.metric(
    "模型特徵",
    feature_count,
)

summary_col4.metric(
    "預測期間",
    "未來 5 日",
)


st.divider()


# ==================================================
# 系統流程
# ==================================================

st.subheader("系統流程")

st.code(
    """
Yahoo Finance 股價與成交量
              +
FinMind 三大法人籌碼
              ↓
資料清理與日期合併
              ↓
建立 32 個模型特徵
              ↓
建立未來 5 日 Target
              ↓
TimeSeriesSplit 五折驗證
              ↓
XGBoost 分類與 AI 分數
              ↓
85 檔股票排行榜
              ↓
AI Top 5 投資組合回測
              ↓
Streamlit 視覺化網站
""",
    language="text",
)


st.divider()


# ==================================================
# 資料來源
# ==================================================

st.subheader("資料來源")

source_col1, source_col2 = st.columns(2)


with source_col1:

    st.markdown("### Yahoo Finance")

    st.markdown(
        """
Yahoo Finance 提供台股日線價格與成交量資料，
本平台使用的欄位包括：

- 交易日期 Date
- 開盤價 Open
- 最高價 High
- 最低價 Low
- 收盤價 Close
- 成交量 Volume
"""
    )


with source_col2:

    st.markdown("### FinMind")

    st.markdown(
        """
FinMind 提供台股三大法人籌碼資料，
本平台使用的資料包括：

- 外資買進與賣出
- 投信買進與賣出
- 自營商自行買賣
- 自營商避險交易
- 三大法人每日買賣超
"""
    )


st.info(
    "股價與法人資料的發布時間可能不同。"
    "正式每日更新時，系統應分別檢查股價資料日期與法人資料日期。"
)


st.divider()


# ==================================================
# 股票池
# ==================================================

st.subheader("85 檔股票池")

st.markdown(
    """
股票池由 17 個產業及技術主題族群組成，
每個族群包含 5 檔股票，總計 85 檔。

每檔股票在股票池中只會出現一次，
避免模型訓練、族群分析及排行榜產生重複樣本。
"""
)


if stocks_df.empty:

    st.warning(
        "目前無法載入 stocks_85.csv。"
    )

else:

    industry_summary_df = (
        stocks_df
        .groupby("Industry")
        .agg(
            股票數量=("StockID", "nunique"),
        )
        .reset_index()
        .sort_values("Industry")
    )

    with st.expander("查看 17 個產業族群"):

        st.dataframe(
            industry_summary_df,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("查看完整 85 檔股票清單"):

        stock_display_columns = [
            column
            for column in [
                "StockID",
                "StockName",
                "Industry",
                "Market",
                "YahooID",
            ]
            if column in stocks_df.columns
        ]

        st.dataframe(
            stocks_df[stock_display_columns],
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ==================================================
# 預測目標
# ==================================================

st.subheader("預測目標 Target")

st.subheader("模型限制與風險")

model_limitations = [
    "股票市場具有高度雜訊，歷史規律不保證未來持續有效。",
    "五折平均 ROC-AUC 僅略高於 0.5，模型主要價值在相對排序，而不是精準判斷全部樣本。",
    "AI Top 5 為集中投資組合，波動與最大回撤高於分散型投資組合。",
    "回測績效會受到股票池、測試期間、交易成本及再平衡方式影響。",
    "部分新掛牌股票的歷史資料較短。",
    "法人資料可能晚於股價資料發布。",
    "模型尚未納入財務報表、總體經濟、新聞情緒與重大事件。",
    "模型輸出的 AI 分數尚未完成機率校準。",
    "公司重大事件、除權息、增資及市場制度改變可能影響模型表現。",
    "歷史回測結果不代表未來一定能取得相同績效。",
]

for number, limitation in enumerate(
    model_limitations,
    start=1,
):
    st.write(f"{number}. {limitation}")


st.error(
    "本平台僅供課程專題、資料分析與模型研究，"
    "不構成投資建議、買賣推薦或獲利保證。"
)