import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 專案路徑
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ui import (  # noqa: E402
    load_global_css,
    render_footer,
    render_sidebar_info,
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"

FEATURE_LIST_PATHS = [
    MODEL_DIR / "model_features.pkl",
    MODEL_DIR / "model_features_32.pkl",
]

STOCK_METADATA_PATHS = [
    CONFIG_DIR / "stock_metadata.csv",
    CONFIG_DIR / "stocks.csv",
    CONFIG_DIR / "stock_list.csv",
    CONFIG_DIR / "stock_universe.csv",
]

CV_RESULT_PATHS = [
    RESULT_DIR / "cv_metrics.csv",
    RESULT_DIR / "cv_results.csv",
    RESULT_DIR / "xgb_cv_metrics.csv",
    RESULT_DIR / "model_cv_metrics.csv",
    RESULT_DIR / "timeseries_cv_results.csv",
]

PORTFOLIO_KPI_PATH = RESULT_DIR / "portfolio_kpis.csv"
GLOBAL_SHAP_PATH = RESULT_DIR / "global_shap_importance.csv"
RANKING_PATH = RESULT_DIR / "latest_rankings.csv"


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="模型說明",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_global_css()
render_sidebar_info()


# ==================================================
# 模型特徵定義
# ==================================================

EXPECTED_FEATURES = [
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
    "RSI",
    "MACD_Diff_Ratio",
    "ATR_Ratio",
    "Volatility_5D",
    "Volatility_20D",
    "Volume_Ratio",
    "Volume_5D_20D_Ratio",
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
    "Foreign_Buy_Streak",
    "InvestmentTrust_Buy_Streak",
    "Dealer_Buy_Streak",
]

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
    "RSI": "RSI 相對強弱指標",
    "MACD_Diff_Ratio": "MACD 柱狀體比例",
    "ATR_Ratio": "ATR 波動比例",
    "Volatility_5D": "5 日歷史波動率",
    "Volatility_20D": "20 日歷史波動率",
    "Volume_Ratio": "成交量相對 20 日均量",
    "Volume_5D_20D_Ratio": "5 日均量相對 20 日均量",
    "Foreign_NetBuy_Ratio": "外資單日買賣超比例",
    "InvestmentTrust_NetBuy_Ratio": "投信單日買賣超比例",
    "Dealer_NetBuy_Ratio": "自營商單日買賣超比例",
    "Institutional_NetBuy_Ratio": "三大法人單日買賣超比例",
    "Foreign_NetBuy_5D_Ratio": "外資 5 日買賣超比例",
    "Foreign_NetBuy_20D_Ratio": "外資 20 日買賣超比例",
    "InvestmentTrust_NetBuy_5D_Ratio": "投信 5 日買賣超比例",
    "InvestmentTrust_NetBuy_20D_Ratio": "投信 20 日買賣超比例",
    "Dealer_NetBuy_5D_Ratio": "自營商 5 日買賣超比例",
    "Institutional_NetBuy_5D_Ratio": "三大法人 5 日買賣超比例",
    "Institutional_NetBuy_20D_Ratio": "三大法人 20 日買賣超比例",
    "Foreign_Buy_Streak": "外資連續買超天數",
    "InvestmentTrust_Buy_Streak": "投信連續買超天數",
    "Dealer_Buy_Streak": "自營商連續買超天數",
}

FEATURE_GROUP_MAP = {
    "價格動能": EXPECTED_FEATURES[0:5],
    "均線與趨勢": EXPECTED_FEATURES[5:11],
    "技術指標與波動": EXPECTED_FEATURES[11:16],
    "成交量": EXPECTED_FEATURES[16:18],
    "法人單日籌碼": EXPECTED_FEATURES[18:22],
    "法人累計籌碼": EXPECTED_FEATURES[22:29],
    "法人連續買超": EXPECTED_FEATURES[29:32],
}


# ==================================================
# 資料載入函式
# ==================================================

@st.cache_data(ttl=300)
def load_feature_list():
    """載入正式模型特徵順序。"""

    for feature_path in FEATURE_LIST_PATHS:
        if not feature_path.exists():
            continue

        try:
            loaded_features = joblib.load(feature_path)
            return [str(feature) for feature in list(loaded_features)]
        except (OSError, ValueError, TypeError):
            continue

    return EXPECTED_FEATURES.copy()


@st.cache_data(ttl=300)
def load_stock_metadata():
    """載入股票清單，若無設定檔則由排行榜建立。"""

    for metadata_path in STOCK_METADATA_PATHS:
        if not metadata_path.exists():
            continue

        try:
            dataframe = pd.read_csv(
                metadata_path,
                dtype={"StockID": str},
            )
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue

        if "StockID" in dataframe.columns:
            dataframe["StockID"] = (
                dataframe["StockID"].astype(str).str.zfill(4)
            )

        return dataframe

    if not RANKING_PATH.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            RANKING_PATH,
            dtype={"StockID": str},
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()

    if "StockID" in dataframe.columns:
        dataframe["StockID"] = dataframe["StockID"].astype(str).str.zfill(4)

    keep_columns = [
        column
        for column in ["StockID", "StockName", "Industry"]
        if column in dataframe.columns
    ]

    if not keep_columns:
        return pd.DataFrame()

    return dataframe[keep_columns].drop_duplicates().reset_index(drop=True)


@st.cache_data(ttl=300)
def load_first_available_csv(paths):
    """載入第一個存在且可讀取的 CSV。"""

    for file_path in paths:
        if not file_path.exists():
            continue

        try:
            dataframe = pd.read_csv(file_path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue

        if not dataframe.empty:
            return dataframe, file_path.name

    return pd.DataFrame(), None


@st.cache_data(ttl=300)
def load_cv_results():
    """載入 TimeSeriesSplit 五折分類績效。"""

    return load_first_available_csv(CV_RESULT_PATHS)


@st.cache_data(ttl=300)
def load_portfolio_kpis():
    """載入回測 KPI。"""

    if not PORTFOLIO_KPI_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(PORTFOLIO_KPI_PATH)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_global_shap():
    """載入全域 SHAP 特徵重要性。"""

    if not GLOBAL_SHAP_PATH.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(GLOBAL_SHAP_PATH)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()

    numeric_columns = [
        "Mean_Abs_SHAP",
        "Mean_SHAP",
        "Positive_Count",
        "Negative_Count",
        "Importance_Rank",
        "Importance_Percent",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


# ==================================================
# 顯示工具函式
# ==================================================

def create_summary_card(label, value, note, accent_color):
    """建立模型說明摘要卡片。"""

    return f"""
    <div style="
        width:100%; height:180px; min-height:180px;
        box-sizing:border-box; display:flex; flex-direction:column;
        justify-content:space-between; padding:21px;
        background:linear-gradient(145deg,#ffffff 0%,#f8fbff 100%);
        border:1px solid #dce5ef; border-top:4px solid {accent_color};
        border-radius:16px;
        box-shadow:0 5px 18px rgba(30,64,175,0.09);
    ">
        <div style="color:#5f6f85;font-size:0.96rem;font-weight:700;">
            {label}
        </div>
        <div style="color:#172033;font-size:2rem;font-weight:850;line-height:1.15;">
            {value}
        </div>
        <div style="color:#66768b;font-size:0.84rem;font-weight:600;line-height:1.45;">
            {note}
        </div>
    </div>
    """


def create_content_card(title, body, accent_color="#2563eb"):
    """建立說明內容卡片。"""

    return f"""
    <div style="
        width:100%; height:100%; box-sizing:border-box; padding:22px;
        background:#ffffff; border:1px solid #dce5ef;
        border-left:5px solid {accent_color}; border-radius:15px;
        box-shadow:0 4px 14px rgba(30,64,175,0.07);
    ">
        <div style="color:#172033;font-size:1.1rem;font-weight:850;margin-bottom:12px;">
            {title}
        </div>
        <div style="color:#5f6f85;font-size:0.93rem;line-height:1.8;">
            {body}
        </div>
    </div>
    """


def safe_metric(dataframe, strategy_name, column_name):
    """安全取得回測 KPI。"""

    if dataframe.empty or "策略" not in dataframe.columns:
        return None

    selected_df = dataframe[dataframe["策略"] == strategy_name]

    if selected_df.empty or column_name not in selected_df.columns:
        return None

    value = pd.to_numeric(selected_df.iloc[0][column_name], errors="coerce")
    return None if pd.isna(value) else float(value)


# ==================================================
# 載入網站資料
# ==================================================

feature_cols = load_feature_list()
stocks_df = load_stock_metadata()
cv_df, cv_file_name = load_cv_results()
portfolio_kpi_df = load_portfolio_kpis()
global_shap_df = load_global_shap()

stock_count = (
    stocks_df["StockID"].nunique()
    if not stocks_df.empty and "StockID" in stocks_df.columns
    else len(stocks_df)
)

industry_count = (
    stocks_df["Industry"].nunique()
    if not stocks_df.empty and "Industry" in stocks_df.columns
    else 0
)

feature_count = len(feature_cols)


# ==================================================
# 頁面 Hero
# ==================================================

st.html(
    """
    <div class="ai-hero">
        <div class="ai-hero-title">模型、資料與驗證設計</div>
        <div class="ai-hero-subtitle">
            說明本平台的資料來源、32 個特徵、五日 Target、
            XGBoost、TimeSeriesSplit、OOF 回測、SHAP 解釋，
            以及每日網站推論與模型開發流程的差異。
        </div>
        <div style="margin-top:18px;">
            <span class="ai-badge ai-badge-blue">XGBoost</span>
            <span class="ai-badge ai-badge-green">TimeSeriesSplit</span>
            <span class="ai-badge ai-badge-purple">32 個特徵</span>
            <span class="ai-badge ai-badge-orange">SHAP 解釋</span>
        </div>
    </div>
    """
)


# ==================================================
# 專題摘要
# ==================================================

st.subheader("專題摘要")

st.html(
    """
    <div class="ai-card">
        <div class="ai-card-title">台股 AI 智慧選股與五日趨勢預測平台</div>
        <div class="ai-card-text">
            平台整合 Yahoo Finance 台股日線價格與成交量，以及 FinMind
            外資、投信與自營商籌碼資料，建立技術面、量價面及法人籌碼特徵。
            模型使用 XGBoost 對 85 檔股票進行相對排序，並以
            TimeSeriesSplit 與 OOF 樣本外預測評估歷史表現。
            每日網站更新時載入既有正式模型，計算最新 AI 分數、排行榜與 SHAP 解釋。
        </div>
    </div>
    """
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(
    4,
    gap="medium",
)

with summary_col1:
    st.html(create_summary_card("股票數量", f"{stock_count} 檔", "跨 17 個產業與技術主題", "#2563eb"))

with summary_col2:
    st.html(create_summary_card("產業族群", f"{industry_count} 個", "每個族群原則上 5 檔股票", "#0891b2"))

with summary_col3:
    st.html(create_summary_card("模型特徵", f"{feature_count} 個", "18 個技術量價與 14 個法人籌碼", "#6d28d9"))

with summary_col4:
    st.html(create_summary_card("預測期間", "5 個交易日", "自動排除週末與市場休市日", "#d97706"))

st.divider()


# ==================================================
# 系統流程
# ==================================================

st.subheader("系統流程")

flow_col1, flow_col2 = st.columns(2, gap="large")

with flow_col1:
    st.html(
        create_content_card(
            "模型開發與驗證流程",
            """
            Yahoo Finance 與 FinMind 資料<br>
            ↓<br>
            資料清理、日期對齊與合併<br>
            ↓<br>
            建立 32 個模型特徵與五日 Target<br>
            ↓<br>
            TimeSeriesSplit 五折切分，gap = 5<br>
            ↓<br>
            每一折訓練 XGBoost 並預測驗證期間<br>
            ↓<br>
            合併 OOF 樣本外預測<br>
            ↓<br>
            AI Top 5 回測與模型評估<br>
            ↓<br>
            使用完整可用歷史資料訓練正式模型
            """,
            "#6d28d9",
        )
    )

with flow_col2:
    st.html(
        create_content_card(
            "每日網站推論流程",
            """
            更新最新股價與法人資料<br>
            ↓<br>
            重新計算最新 32 個特徵<br>
            ↓<br>
            載入既有正式 XGBoost 模型<br>
            ↓<br>
            產生 AI 分數與 85 檔股票排名<br>
            ↓<br>
            計算最新個股 SHAP 解釋<br>
            ↓<br>
            GitHub Actions 驗證並提交結果<br>
            ↓<br>
            Streamlit Community Cloud 自動同步<br><br>
            <strong>每日不重新執行 TimeSeriesSplit，也不每日重新訓練模型。</strong>
            """,
            "#0891b2",
        )
    )

st.divider()


# ==================================================
# 資料來源
# ==================================================

st.subheader("資料來源")

source_col1, source_col2 = st.columns(2, gap="large")

with source_col1:
    st.html(
        create_content_card(
            "Yahoo Finance 股價與成交量",
            """
            使用台股日線資料，包括：<br><br>
            • 交易日期 Date<br>
            • 開盤價 Open<br>
            • 最高價 High<br>
            • 最低價 Low<br>
            • 收盤價 Close<br>
            • 成交量 Volume<br><br>
            日線資料只包含實際交易日，因此五日預測指的是五個交易日。
            """,
            "#2563eb",
        )
    )

with source_col2:
    st.html(
        create_content_card(
            "FinMind 三大法人籌碼",
            """
            使用台股法人交易資料，包括：<br><br>
            • 外資每日買賣超<br>
            • 投信每日買賣超<br>
            • 自營商自行買賣與避險交易<br>
            • 三大法人合計買賣超<br>
            • 5 日與 20 日累計籌碼比例<br>
            • 法人連續買超天數<br><br>
            股價與法人資料的最新日期分別檢查，避免將尚未發布的法人資料視為缺漏。
            """,
            "#10b981",
        )
    )

st.divider()


# ==================================================
# 32 個模型特徵
# ==================================================

st.subheader("32 個模型特徵")

technical_count = sum(
    len(FEATURE_GROUP_MAP[group])
    for group in ["價格動能", "均線與趨勢", "技術指標與波動", "成交量"]
)

institution_count = sum(
    len(FEATURE_GROUP_MAP[group])
    for group in ["法人單日籌碼", "法人累計籌碼", "法人連續買超"]
)

feature_summary_col1, feature_summary_col2 = st.columns(2, gap="large")

with feature_summary_col1:
    st.html(
        create_content_card(
            f"技術面與量價面，共 {technical_count} 個",
            "價格動能 5 個、均線與趨勢 6 個、技術指標與波動 5 個，以及成交量 2 個。",
            "#2563eb",
        )
    )

with feature_summary_col2:
    st.html(
        create_content_card(
            f"法人籌碼面，共 {institution_count} 個",
            "法人單日籌碼 4 個、法人累計籌碼 7 個，以及法人連續買超天數 3 個。",
            "#10b981",
        )
    )

for group_name, group_features in FEATURE_GROUP_MAP.items():
    with st.expander(f"{group_name}｜{len(group_features)} 個特徵"):
        feature_rows = []

        for feature in group_features:
            feature_rows.append(
                {
                    "原始特徵名稱": feature,
                    "中文說明": FEATURE_NAME_MAP.get(feature, feature),
                    "正式模型使用": "是" if feature in feature_cols else "否",
                }
            )

        st.dataframe(
            pd.DataFrame(feature_rows),
            use_container_width=True,
            hide_index=True,
        )

if feature_cols != EXPECTED_FEATURES:
    st.warning(
        "目前模型特徵檔案的順序或名稱與預期 32 特徵清單不同。"
        "每日推論仍應以 model_features.pkl 儲存的正式順序為準。"
    )

st.divider()


# ==================================================
# 預測目標
# ==================================================

st.subheader("預測目標 Target")

target_col1, target_col2 = st.columns(2, gap="large")

with target_col1:
    st.html(
        create_content_card(
            "Target = 1",
            "未來第 5 個交易日收盤價相對當日收盤價的報酬率，大於 1%。",
            "#dc2626",
        )
    )

with target_col2:
    st.html(
        create_content_card(
            "Target = 0",
            "未來第 5 個交易日收盤價相對當日收盤價的報酬率，小於或等於 1%。",
            "#16a34a",
        )
    )

st.code(
    "Future_Return_5D = Close.shift(-5) / Close - 1\n"
    "Target = 1 if Future_Return_5D > 0.01 else 0",
    language="python",
)

st.info(
    "shift(-5) 是往後第 5 筆交易日資料，不是往後 5 個日曆日。"
    "週末、國定假日與市場休市日不會計入。"
)

st.divider()


# ==================================================
# TimeSeriesSplit 與 OOF
# ==================================================

st.subheader("TimeSeriesSplit 五折驗證與 OOF")

validation_col1, validation_col2 = st.columns(2, gap="large")

with validation_col1:
    st.html(
        create_content_card(
            "時間序列切分設計",
            """
            • 使用 5 折 TimeSeriesSplit<br>
            • 每一折只用較早期間訓練<br>
            • 驗證集位於訓練集之後<br>
            • 同一交易日的 85 檔股票不拆到兩側<br>
            • 訓練與驗證之間保留 5 個交易日 gap<br><br>
            gap = 5 用來降低未來五日 Target 區間重疊造成的資料洩漏。
            """,
            "#6d28d9",
        )
    )

with validation_col2:
    st.html(
        create_content_card(
            "OOF 樣本外預測",
            """
            每一折都會產生一段模型未看過的驗證預測。<br><br>
            Fold 1 驗證預測<br>
            + Fold 2 驗證預測<br>
            + Fold 3 驗證預測<br>
            + Fold 4 驗證預測<br>
            + Fold 5 驗證預測<br>
            = 完整 OOF 預測<br><br>
            AI Top 5 歷史回測使用 OOF 分數，不使用樣本內訓練預測。
            """,
            "#0891b2",
        )
    )

if not cv_df.empty:
    auc_column = next(
        (
            column
            for column in ["ROC_AUC", "ROC-AUC", "AUC", "roc_auc", "Valid_ROC_AUC"]
            if column in cv_df.columns
        ),
        None,
    )

    if auc_column is not None:
        cv_df[auc_column] = pd.to_numeric(cv_df[auc_column], errors="coerce")
        mean_auc = cv_df[auc_column].mean()
        st.metric("五折平均 ROC-AUC", f"{mean_auc:.4f}")

    st.caption(f"分類績效資料來源：{cv_file_name}")

st.divider()


# ==================================================
# XGBoost 設計
# ==================================================

st.subheader("XGBoost 過度擬合與類別不平衡控制")

parameter_data = pd.DataFrame(
    [
        ["max_depth", "4", "限制每棵樹的最大深度，降低模型複雜度"],
        ["min_child_weight", "5", "避免由過少樣本形成葉節點"],
        ["subsample", "0.8", "每棵樹使用 80% 訓練樣本"],
        ["colsample_bytree", "0.8", "每棵樹使用 80% 特徵"],
        ["reg_alpha", "0.1", "L1 正則化"],
        ["reg_lambda", "1.0", "L2 正則化"],
        ["scale_pos_weight", "每折動態計算", "負樣本數除以正樣本數，降低模型偏向多數類別"],
    ],
    columns=["參數", "設定", "作用"],
)

st.dataframe(parameter_data, use_container_width=True, hide_index=True)

st.html(
    create_content_card(
        "scale_pos_weight 的意思",
        "每一折只使用該折訓練集的負樣本數除以正樣本數。"
        "若正樣本較少，模型訓練時會提高正樣本錯判所產生的損失權重，"
        "避免模型因負樣本較多而傾向預測 Target = 0。"
        "這個參數只影響訓練階段，不是 AI 分數，也不是股票持股權重。",
        "#d97706",
    )
)

st.divider()


# ==================================================
# 回測設計與成果
# ==================================================

st.subheader("AI Top 5 回測設計")

backtest_col1, backtest_col2 = st.columns(2, gap="large")

with backtest_col1:
    st.html(
        create_content_card(
            "回測規則",
            """
            • 依 OOF AI 分數由高至低排序<br>
            • 每 5 個交易日重新平衡<br>
            • 選取 AI Top 5<br>
            • 五檔等權配置，每檔 20%<br>
            • 每期扣除完整交易成本 0.60%<br>
            • 與同期可用股票等權基準比較
            """,
            "#2563eb",
        )
    )

with backtest_col2:
    top5_cumulative = safe_metric(portfolio_kpi_df, "AI Top 5", "累積報酬")
    top5_annual = safe_metric(portfolio_kpi_df, "AI Top 5", "年化報酬")
    top5_sharpe = safe_metric(portfolio_kpi_df, "AI Top 5", "Sharpe_Ratio")
    top5_drawdown = safe_metric(portfolio_kpi_df, "AI Top 5", "最大回撤")

    result_lines = []

    if top5_cumulative is not None:
        result_lines.append(f"累積報酬：{top5_cumulative:.2%}")
    if top5_annual is not None:
        result_lines.append(f"年化報酬：{top5_annual:.2%}")
    if top5_sharpe is not None:
        result_lines.append(f"Sharpe Ratio：{top5_sharpe:.3f}")
    if top5_drawdown is not None:
        result_lines.append(f"最大回撤：{top5_drawdown:.2%}")

    result_body = "<br>".join(result_lines) if result_lines else "目前找不到回測 KPI。"

    st.html(
        create_content_card(
            "主要歷史回測指標",
            result_body
            + "<br><br>歷史 OOF 回測只代表特定股票池與測試期間的模擬結果，不代表未來績效。",
            "#10b981",
        )
    )

st.divider()


# ==================================================
# 全域 SHAP
# ==================================================

st.subheader("全域 SHAP 特徵重要性")

st.caption(
    "Mean Abs SHAP 越大，代表該特徵對目前 85 檔股票模型輸出的平均影響程度越高。"
    "全域重要性只表示影響程度，不代表固定的正向或負向關係。"
)

if global_shap_df.empty:
    st.info("目前找不到 results/global_shap_importance.csv。")
else:
    required_shap_columns = {"Feature", "Mean_Abs_SHAP"}

    if not required_shap_columns.issubset(global_shap_df.columns):
        st.info("全域 SHAP 檔案缺少 Feature 或 Mean_Abs_SHAP 欄位。")
    else:
        shap_chart_df = (
            global_shap_df
            .dropna(subset=["Feature", "Mean_Abs_SHAP"])
            .nlargest(10, "Mean_Abs_SHAP")
            .sort_values("Mean_Abs_SHAP", ascending=True)
            .copy()
        )

        shap_chart_df["特徵名稱"] = (
            shap_chart_df["Feature"]
            .map(FEATURE_NAME_MAP)
            .fillna(shap_chart_df["Feature"])
        )

        shap_figure = px.bar(
            shap_chart_df,
            x="Mean_Abs_SHAP",
            y="特徵名稱",
            orientation="h",
            color="Mean_Abs_SHAP",
            color_continuous_scale=[
                [0.0, "#c4b5fd"],
                [0.5, "#8b5cf6"],
                [1.0, "#5b21b6"],
            ],
            text="Mean_Abs_SHAP",
            custom_data=["Feature"],
            labels={
                "Mean_Abs_SHAP": "平均絕對 SHAP 值",
                "特徵名稱": "",
            },
        )

        shap_figure.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "原始特徵：%{customdata[0]}<br>"
                "Mean Abs SHAP：%{x:.6f}"
                "<extra></extra>"
            ),
        )

        shap_figure.update_layout(
            height=600,
            margin={"l": 35, "r": 90, "t": 30, "b": 85},
            coloraxis_showscale=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font={
                "family": "Microsoft JhengHei, Noto Sans TC, Arial",
                "color": "#172033",
                "size": 14,
            },
            xaxis={
                "automargin": True,
                "title": {"text": "平均絕對 SHAP 值", "standoff": 20},
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
            },
            yaxis={
                "automargin": True,
                "title": {"text": ""},
            },
        )

        st.plotly_chart(shap_figure, use_container_width=True)

        display_columns = [
            column
            for column in [
                "Importance_Rank",
                "Feature",
                "Mean_Abs_SHAP",
                "Mean_SHAP",
                "Importance_Percent",
            ]
            if column in global_shap_df.columns
        ]

        st.dataframe(
            global_shap_df.sort_values("Mean_Abs_SHAP", ascending=False)[display_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )

st.divider()


# ==================================================
# AI 分數與 SHAP 解讀
# ==================================================

st.subheader("AI 分數與 SHAP 的正確解讀")

interpret_col1, interpret_col2 = st.columns(2, gap="large")

with interpret_col1:
    st.html(
        create_content_card(
            "AI 分數",
            "AI 分數用於 85 檔股票之間的相對排序。"
            "分數越高，代表目前特徵組合越接近歷史上的 Target = 1 樣本。"
            "AI 分數尚未完成機率校準，因此 70 分不代表 70% 上漲機率。",
            "#2563eb",
        )
    )

with interpret_col2:
    st.html(
        create_content_card(
            "SHAP 貢獻",
            "正 SHAP 值代表該特徵推升 Target = 1 的模型原始輸出，"
            "負 SHAP 值代表壓低模型輸出。SHAP 值不是報酬率、機率或分數點數，"
            "而且特徵可能存在非線性與交互作用。",
            "#6d28d9",
        )
    )

st.divider()


# ==================================================
# 股票池
# ==================================================

st.subheader("85 檔股票池")

st.caption(
    "股票池由 17 個產業及技術主題族群組成。每檔股票在股票池中只出現一次，"
    "避免模型訓練、族群分析與排行榜產生重複樣本。"
)

if stocks_df.empty:
    st.info("目前找不到股票清單設定檔或排行榜資料。")
else:
    stock_display_columns = [
        column
        for column in ["StockID", "StockName", "Industry"]
        if column in stocks_df.columns
    ]

    st.dataframe(
        stocks_df[stock_display_columns]
        .sort_values(stock_display_columns[-1] if stock_display_columns else stocks_df.columns[0])
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

st.divider()


# ==================================================
# 模型限制與風險
# ==================================================

st.subheader("模型限制與風險")

model_limitations = [
    "股票市場具有高度雜訊，歷史規律不保證未來持續有效。",
    "五折平均 ROC-AUC 僅略高於 0.5，模型主要用途是相對排序，不是精準判斷所有樣本。",
    "AI Top 5 是集中投資組合，波動與最大回撤可能高於分散型投資組合。",
    "回測績效會受到股票池、測試期間、交易成本與再平衡方式影響。",
    "部分新掛牌股票的歷史資料較短，可能形成樣本期間差異。",
    "股價與法人資料的發布時間可能不同，最新日期必須分別檢查。",
    "模型尚未納入完整財務報表、總體經濟、新聞情緒與重大事件。",
    "AI 分數尚未完成機率校準。",
    "除權息、增資、停牌、制度改變與重大事件可能影響模型表現。",
    "歷史 OOF 回測結果不代表未來一定能取得相同績效。",
]

for number, limitation in enumerate(model_limitations, start=1):
    st.write(f"**{number}.** {limitation}")

st.error(
    "本平台僅供課程專題、資料分析與模型研究，"
    "不構成投資建議、買賣推薦或獲利保證。"
)


# ==================================================
# 共用頁尾
# ==================================================

render_footer()