import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==================================================
# 專案路徑設定
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICE_DIR = DATA_DIR / "raw_price"
RAW_INST_DIR = DATA_DIR / "raw_institution"
MERGED_DIR = DATA_DIR / "merged"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

STOCK_LIST_PATH = CONFIG_DIR / "stocks_85.csv"

MODEL_PATH = (
    MODEL_DIR / "multi_stock_xgb_5d.pkl"
)

FEATURE_LIST_PATH = (
    MODEL_DIR / "model_features.pkl"
)

LATEST_RANKING_PATH = (
    RESULT_DIR / "latest_rankings.csv"
)

LATEST_FEATURE_PATH = (
    RESULT_DIR / "latest_features.parquet"
)

INDUSTRY_RANKING_PATH = (
    RESULT_DIR / "industry_rankings.csv"
)

UPDATE_STATUS_PATH = (
    RESULT_DIR / "update_status.json"
)


# 讓 Python 可以讀取同資料夾的特徵工程程式
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )

from feature_engineering import (  # noqa: E402
    add_all_features,
    get_latest_valid_features,
)


# ==================================================
# 更新參數
# ==================================================

FINMIND_TOKEN = os.getenv(
    "FINMIND_TOKEN",
    "",
).strip()

FINMIND_API_URL = (
    "https://api.finmindtrade.com/api/v4/data"
)

FINMIND_DATASET = (
    "TaiwanStockInstitutionalInvestorsBuySell"
)

# Yahoo 每次重新抓取最近 180 天，
# 與既有資料合併，避免除權息或資料修正遺漏
PRICE_LOOKBACK_DAYS = 180

# 法人資料重新抓取最近 45 天
INSTITUTION_LOOKBACK_DAYS = 45

# API 呼叫之間稍微停頓
REQUEST_SLEEP_SECONDS = 0.6

EXPECTED_STOCK_COUNT = 85
EXPECTED_INDUSTRY_COUNT = 17


# ==================================================
# 基本工具函式
# ==================================================

def ensure_directories():
    """
    確認每日更新需要的資料夾存在。
    """

    directories = [
        CONFIG_DIR,
        RAW_PRICE_DIR,
        RAW_INST_DIR,
        MERGED_DIR,
        MODEL_DIR,
        RESULT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_json_atomic(
    data,
    output_path,
):
    """
    先寫入暫存檔，再取代正式 JSON。
    避免更新中斷時留下損壞檔案。
    """

    temp_path = output_path.with_suffix(
        ".tmp.json"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(output_path)


def save_csv_atomic(
    dataframe,
    output_path,
):
    """
    安全儲存 CSV。
    """

    temp_path = output_path.with_suffix(
        ".tmp.csv"
    )

    dataframe.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    temp_path.replace(output_path)


def save_parquet_atomic(
    dataframe,
    output_path,
):
    """
    安全儲存 Parquet。
    """

    temp_path = output_path.with_suffix(
        ".tmp.parquet"
    )

    dataframe.to_parquet(
        temp_path,
        index=False,
    )

    temp_path.replace(output_path)


def load_stock_list():
    """
    載入並檢查 85 檔股票清單。
    """

    if not STOCK_LIST_PATH.exists():
        raise FileNotFoundError(
            f"找不到股票清單：{STOCK_LIST_PATH}"
        )

    stocks_df = pd.read_csv(
        STOCK_LIST_PATH,
        dtype={
            "StockID": str,
            "YahooID": str,
            "StockName": str,
            "Industry": str,
            "Market": str,
        },
    )

    required_columns = [
        "StockID",
        "YahooID",
        "StockName",
        "Industry",
        "Market",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in stocks_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"股票清單缺少欄位：{missing_columns}"
        )

    stocks_df["StockID"] = (
        stocks_df["StockID"]
        .astype(str)
        .str.strip()
        .str.zfill(4)
    )

    for column in [
        "YahooID",
        "StockName",
        "Industry",
        "Market",
    ]:
        stocks_df[column] = (
            stocks_df[column]
            .astype(str)
            .str.strip()
        )

    if len(stocks_df) != EXPECTED_STOCK_COUNT:
        raise ValueError(
            f"股票清單應有 {EXPECTED_STOCK_COUNT} 檔，"
            f"目前為 {len(stocks_df)} 檔。"
        )

    if (
        stocks_df["StockID"].nunique()
        != EXPECTED_STOCK_COUNT
    ):
        raise ValueError(
            "股票清單含有重複股票代碼。"
        )

    if (
        stocks_df["Industry"].nunique()
        != EXPECTED_INDUSTRY_COUNT
    ):
        raise ValueError(
            f"股票清單應有 "
            f"{EXPECTED_INDUSTRY_COUNT} 個族群。"
        )

    return stocks_df


def combine_and_deduplicate(
    old_df,
    new_df,
    key_columns,
):
    """
    合併既有資料與新資料，並依指定欄位去重。
    新資料優先保留。
    """

    dataframes = []

    if (
        old_df is not None
        and not old_df.empty
    ):
        dataframes.append(old_df)

    if (
        new_df is not None
        and not new_df.empty
    ):
        dataframes.append(new_df)

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined_df["date"] = pd.to_datetime(
        combined_df["date"],
        errors="coerce",
    )

    combined_df = (
        combined_df
        .dropna(subset=["date"])
        .drop_duplicates(
            subset=key_columns,
            keep="last",
        )
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    return combined_df


# ==================================================
# Yahoo Finance 股價更新
# ==================================================

def normalize_yahoo_price(
    dataframe,
    stock_row,
):
    """
    整理 yfinance 回傳格式。
    """

    if (
        dataframe is None
        or dataframe.empty
    ):
        return pd.DataFrame()

    price_df = dataframe.copy()

    if isinstance(
        price_df.columns,
        pd.MultiIndex,
    ):
        price_df.columns = (
            price_df.columns.get_level_values(0)
        )

    price_df = price_df.reset_index()

    if "Date" in price_df.columns:
        price_df = price_df.rename(
            columns={"Date": "date"}
        )

    if "Datetime" in price_df.columns:
        price_df = price_df.rename(
            columns={"Datetime": "date"}
        )

    required_columns = [
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in price_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{stock_row['StockID']} Yahoo資料缺少："
            f"{missing_columns}"
        )

    price_df["date"] = pd.to_datetime(
        price_df["date"],
        errors="coerce",
    )

    # 移除 yfinance 可能附帶的時區
    try:
        price_df["date"] = (
            price_df["date"].dt.tz_localize(None)
        )
    except TypeError:
        pass

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        price_df[column] = pd.to_numeric(
            price_df[column],
            errors="coerce",
        )

    price_df["StockID"] = str(
        stock_row["StockID"]
    ).zfill(4)

    price_df["YahooID"] = str(
        stock_row["YahooID"]
    )

    price_df["StockName"] = str(
        stock_row["StockName"]
    )

    price_df["Industry"] = str(
        stock_row["Industry"]
    )

    price_df["Market"] = str(
        stock_row["Market"]
    )

    output_columns = [
        "date",
        "StockID",
        "YahooID",
        "StockName",
        "Industry",
        "Market",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    price_df = (
        price_df[output_columns]
        .dropna(
            subset=[
                "date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )
        .drop_duplicates(
            subset=["date", "StockID"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return price_df


def update_price_data(stock_row):
    """
    增量更新單檔 Yahoo Finance 股價。
    """

    stock_id = str(
        stock_row["StockID"]
    ).zfill(4)

    yahoo_id = str(
        stock_row["YahooID"]
    )

    output_path = (
        RAW_PRICE_DIR
        / f"{stock_id}_price.parquet"
    )

    if output_path.exists():
        old_df = pd.read_parquet(
            output_path
        )

        old_df["date"] = pd.to_datetime(
            old_df["date"],
            errors="coerce",
        )
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        start_date = "2020-01-01"
    else:
        latest_date = old_df["date"].max()

        start_date = (
            latest_date
            - pd.Timedelta(
                days=PRICE_LOOKBACK_DAYS
            )
        ).strftime("%Y-%m-%d")

    # end 為不包含，因此加一天
    end_date = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    downloaded_df = yf.download(
        yahoo_id,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    new_df = normalize_yahoo_price(
        downloaded_df,
        stock_row,
    )

    if new_df.empty:

        if old_df.empty:
            raise ValueError(
                f"{stock_id} {stock_row['StockName']} "
                "Yahoo股價資料為空。"
            )

        print(
            f"  Yahoo沒有新資料，保留既有股價。"
        )

        return old_df

    combined_df = combine_and_deduplicate(
        old_df=old_df,
        new_df=new_df,
        key_columns=[
            "date",
            "StockID",
        ],
    )

    if combined_df.empty:
        raise ValueError(
            f"{stock_id} 股價合併後為空。"
        )

    save_parquet_atomic(
        combined_df,
        output_path,
    )

    return combined_df


# ==================================================
# FinMind 法人資料更新
# ==================================================

def download_finmind_raw(
    stock_id,
    start_date,
    end_date,
):
    """
    下載單檔 FinMind 法人原始資料。
    """

    if not FINMIND_TOKEN:
        raise ValueError(
            "找不到 FINMIND_TOKEN。"
            "請設定環境變數或 GitHub Secret。"
        )

    params = {
        "dataset": FINMIND_DATASET,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": FINMIND_TOKEN,
    }

    response = requests.get(
        FINMIND_API_URL,
        params=params,
        timeout=90,
    )

    response.raise_for_status()

    response_data = response.json()

    status_value = response_data.get(
        "status"
    )

    if status_value not in [
        200,
        "200",
        None,
    ]:
        raise ValueError(
            response_data.get(
                "msg",
                f"FinMind status={status_value}",
            )
        )

    return pd.DataFrame(
        response_data.get("data", [])
    )


def organize_institutional_data(
    raw_df,
):
    """
    將 FinMind 法人原始資料整理成每日三大法人買賣超。
    """

    if (
        raw_df is None
        or raw_df.empty
    ):
        return pd.DataFrame()

    required_columns = [
        "date",
        "stock_id",
        "buy",
        "sell",
        "name",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"FinMind資料缺少欄位："
            f"{missing_columns}"
        )

    dataframe = raw_df.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe["stock_id"] = (
        dataframe["stock_id"]
        .astype(str)
        .str.zfill(4)
    )

    dataframe["buy"] = pd.to_numeric(
        dataframe["buy"],
        errors="coerce",
    ).fillna(0.0)

    dataframe["sell"] = pd.to_numeric(
        dataframe["sell"],
        errors="coerce",
    ).fillna(0.0)

    dataframe["net_buy"] = (
        dataframe["buy"]
        - dataframe["sell"]
    )

    institution_name = (
        dataframe["name"]
        .astype(str)
        .str.strip()
    )

    foreign_condition = (
        institution_name.str.contains(
            "Foreign",
            case=False,
            na=False,
        )
    )

    trust_condition = (
        institution_name.str.contains(
            "Investment_Trust",
            case=False,
            na=False,
        )
    )

    dealer_condition = (
        institution_name.str.contains(
            "Dealer_self|Dealer_Hedging",
            case=False,
            na=False,
            regex=True,
        )
    )

    dataframe["Foreign_NetBuy"] = (
        np.where(
            foreign_condition,
            dataframe["net_buy"],
            0.0,
        )
    )

    dataframe[
        "InvestmentTrust_NetBuy"
    ] = np.where(
        trust_condition,
        dataframe["net_buy"],
        0.0,
    )

    dataframe["Dealer_NetBuy"] = (
        np.where(
            dealer_condition,
            dataframe["net_buy"],
            0.0,
        )
    )

    daily_df = (
        dataframe
        .dropna(subset=["date"])
        .groupby(
            ["date", "stock_id"],
            as_index=False,
        )[
            [
                "Foreign_NetBuy",
                "InvestmentTrust_NetBuy",
                "Dealer_NetBuy",
            ]
        ]
        .sum()
    )

    daily_df[
        "Institutional_Total_NetBuy"
    ] = (
        daily_df["Foreign_NetBuy"]
        + daily_df[
            "InvestmentTrust_NetBuy"
        ]
        + daily_df["Dealer_NetBuy"]
    )

    daily_df = daily_df.rename(
        columns={
            "stock_id": "StockID"
        }
    )

    daily_df["StockID"] = (
        daily_df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    daily_df = (
        daily_df
        .drop_duplicates(
            subset=["date", "StockID"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_df


def update_institution_data(
    stock_row,
):
    """
    增量更新單檔法人資料。
    """

    stock_id = str(
        stock_row["StockID"]
    ).zfill(4)

    output_path = (
        RAW_INST_DIR
        / f"{stock_id}_institution.parquet"
    )

    if output_path.exists():
        old_df = pd.read_parquet(
            output_path
        )

        old_df["date"] = pd.to_datetime(
            old_df["date"],
            errors="coerce",
        )
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        start_date = "2020-01-01"
    else:
        latest_date = old_df["date"].max()

        start_date = (
            latest_date
            - pd.Timedelta(
                days=INSTITUTION_LOOKBACK_DAYS
            )
        ).strftime("%Y-%m-%d")

    end_date = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw_df = download_finmind_raw(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    new_df = organize_institutional_data(
        raw_df
    )

    if new_df.empty:

        if old_df.empty:
            raise ValueError(
                f"{stock_id} 沒有法人資料。"
            )

        print(
            "  FinMind沒有新資料，"
            "保留既有法人資料。"
        )

        return old_df

    combined_df = combine_and_deduplicate(
        old_df=old_df,
        new_df=new_df,
        key_columns=[
            "date",
            "StockID",
        ],
    )

    save_parquet_atomic(
        combined_df,
        output_path,
    )

    return combined_df


# ==================================================
# 股價與法人資料合併
# ==================================================

def merge_price_and_institution(
    price_df,
    institution_df,
):
    """
    合併股價與法人資料。

    保留完整股價交易日期，並以日期與股票代碼
    左合併法人買賣超資料。當日沒有法人紀錄時，
    法人買賣超欄位填入 0，確保所有股票可使用
    相同的最新交易日期建立 AI 排行榜。
    """

    if (
        price_df is None
        or price_df.empty
    ):
        raise ValueError(
            "股價資料為空。"
        )

    if (
        institution_df is None
        or institution_df.empty
    ):
        raise ValueError(
            "法人資料為空。"
        )

    price_data = price_df.copy()
    institution_data = institution_df.copy()

    price_data["date"] = pd.to_datetime(
        price_data["date"],
        errors="coerce",
    )

    institution_data["date"] = (
        pd.to_datetime(
            institution_data["date"],
            errors="coerce",
        )
    )

    price_data["StockID"] = (
        price_data["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    institution_data["StockID"] = (
        institution_data["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    merged_df = pd.merge(
        price_data,
        institution_data,
        on=[
            "date",
            "StockID",
        ],
        how="left",
        validate="one_to_one",
    )

    institution_columns = [
        "Foreign_NetBuy",
        "InvestmentTrust_NetBuy",
        "Dealer_NetBuy",
        "Institutional_Total_NetBuy",
    ]

    # 當日沒有法人買賣超紀錄時，視為買賣超 0
    merged_df[institution_columns] = (
        merged_df[institution_columns]
        .fillna(0.0)
    )

    merged_df = (
        merged_df
        .drop_duplicates(
            subset=["date", "StockID"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged_df


# ==================================================
# AI 分數與網站顯示
# ==================================================

def assign_ai_signal(score):
    """
    依 AI 分數產生網站訊號。
    """

    if score >= 70:
        return "高分候選"

    if score >= 60:
        return "偏多觀察"

    if score >= 50:
        return "中性觀察"

    return "暫不列入"


def assign_risk_level(
    atr_ratio,
):
    """
    根據 ATR Ratio 產生簡化風險等級。
    """

    if pd.isna(atr_ratio):
        return "資料不足"

    if atr_ratio >= 0.05:
        return "高風險"

    if atr_ratio >= 0.03:
        return "中高風險"

    if atr_ratio >= 0.015:
        return "中等風險"

    return "相對低風險"


def build_industry_ranking(
    ranking_df,
):
    """
    建立 17 個族群的 AI 排名。
    """

    industry_df = (
        ranking_df
        .groupby("Industry")
        .agg(
            Industry_AI_Score=(
                "AI_Score",
                "mean",
            ),
            Highest_AI_Score=(
                "AI_Score",
                "max",
            ),
            Average_RSI=(
                "RSI",
                "mean",
            ),
            Average_Foreign_5D_Ratio=(
                "Foreign_NetBuy_5D_Ratio",
                "mean",
            ),
            Stock_Count=(
                "StockID",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "Industry_AI_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    industry_df["Industry_Rank"] = (
        np.arange(
            1,
            len(industry_df) + 1,
        )
    )

    top_stock_df = (
        ranking_df
        .sort_values(
            [
                "Industry",
                "Predicted_Probability",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "Industry",
            as_index=False,
        )
        .head(1)[
            [
                "Industry",
                "StockID",
                "StockName",
                "AI_Score",
            ]
        ]
        .rename(
            columns={
                "StockID": "Top_StockID",
                "StockName": "Top_StockName",
                "AI_Score": (
                    "Top_Stock_AI_Score"
                ),
            }
        )
    )

    industry_df = pd.merge(
        industry_df,
        top_stock_df,
        on="Industry",
        how="left",
    )

    return (
        industry_df
        .sort_values("Industry_Rank")
        .reset_index(drop=True)
    )


# ==================================================
# 每日更新主流程
# ==================================================

def main():
    """
    執行每日 85 檔資料更新與 AI 預測。
    """

    ensure_directories()

    started_at = datetime.now()

    print("=" * 70)
    print("開始執行每日 AI 股票更新")
    print(
        "執行時間：",
        started_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )
    print("=" * 70)

    if not FINMIND_TOKEN:
        raise ValueError(
            "FINMIND_TOKEN 尚未設定。"
        )

    stocks_df = load_stock_list()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"找不到特徵清單："
            f"{FEATURE_LIST_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_LIST_PATH
    )

    print(
        f"股票數量：{len(stocks_df)}"
    )

    print(
        f"模型特徵數："
        f"{len(feature_columns)}"
    )

    all_merged_frames = []
    update_records = []

    for number, (
        _,
        stock_row,
    ) in enumerate(
        stocks_df.iterrows(),
        start=1,
    ):

        stock_id = str(
            stock_row["StockID"]
        ).zfill(4)

        stock_name = str(
            stock_row["StockName"]
        )

        print(
            f"\n[{number}/{len(stocks_df)}] "
            f"{stock_id} {stock_name}"
        )

        record = {
            "StockID": stock_id,
            "StockName": stock_name,
            "PriceDate": None,
            "InstitutionDate": None,
            "MergedRows": 0,
            "Status": "失敗",
            "Error": "",
        }

        try:
            price_df = update_price_data(
                stock_row
            )

            print(
                "  股價更新至：",
                price_df["date"]
                .max()
                .date(),
            )

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

            institution_df = (
                update_institution_data(
                    stock_row
                )
            )

            print(
                "  法人更新至：",
                institution_df["date"]
                .max()
                .date(),
            )

            merged_df = (
                merge_price_and_institution(
                    price_df,
                    institution_df,
                )
            )

            merged_path = (
                MERGED_DIR
                / f"{stock_id}_merged.parquet"
            )

            save_parquet_atomic(
                merged_df,
                merged_path,
            )

            all_merged_frames.append(
                merged_df
            )

            record["PriceDate"] = str(
                price_df["date"]
                .max()
                .date()
            )

            record["InstitutionDate"] = str(
                institution_df["date"]
                .max()
                .date()
            )

            record["MergedRows"] = len(
                merged_df
            )

            record["Status"] = "成功"

            print(
                f"  合併完成："
                f"{len(merged_df)} 筆"
            )

        except Exception as error:
            record["Error"] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "  更新失敗：",
                record["Error"],
            )

        update_records.append(record)

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    update_record_df = pd.DataFrame(
        update_records
    )

    failed_df = update_record_df[
        update_record_df["Status"]
        != "成功"
    ].copy()

    if not failed_df.empty:
        failed_stocks = (
            failed_df[
                [
                    "StockID",
                    "StockName",
                    "Error",
                ]
            ]
            .to_dict("records")
        )

        raise RuntimeError(
            f"有 {len(failed_df)} 檔股票更新失敗："
            f"{failed_stocks}"
        )

    if (
        len(all_merged_frames)
        != EXPECTED_STOCK_COUNT
    ):
        raise RuntimeError(
            "合併成功股票數不足 85 檔，"
            "不覆蓋網站排行榜。"
        )

    full_merged_df = pd.concat(
        all_merged_frames,
        ignore_index=True,
    )

    full_merged_df = (
        full_merged_df
        .drop_duplicates(
            subset=["date", "StockID"],
            keep="last",
        )
        .sort_values(
            ["date", "StockID"]
        )
        .reset_index(drop=True)
    )

    print("\n開始重新計算模型特徵")

    full_feature_df = add_all_features(
        full_merged_df
    )

    latest_feature_df = (
        get_latest_valid_features(
            feature_dataframe=(
                full_feature_df
            ),
            feature_columns=(
                feature_columns
            ),
        )
    )

    latest_stock_count = (
        latest_feature_df[
            "StockID"
        ].nunique()
    )

    print(
        "最新可預測股票數：",
        latest_stock_count,
    )

    if (
        latest_stock_count
        != EXPECTED_STOCK_COUNT
    ):
        missing_stock_ids = sorted(
            set(stocks_df["StockID"])
            - set(
                latest_feature_df[
                    "StockID"
                ]
            )
        )

        raise RuntimeError(
            "最新可預測股票不足 85 檔，"
            f"缺少：{missing_stock_ids}"
        )

    missing_features = [
        column
        for column in feature_columns
        if column
        not in latest_feature_df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"最新資料缺少模型特徵："
            f"{missing_features}"
        )

    feature_matrix = (
        latest_feature_df[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if feature_matrix.isna().any().any():
        missing_counts = (
            feature_matrix
            .isna()
            .sum()
        )

        missing_counts = (
            missing_counts[
                missing_counts > 0
            ]
            .to_dict()
        )

        raise RuntimeError(
            f"最新特徵仍有缺失值："
            f"{missing_counts}"
        )

    predicted_probability = (
        model.predict_proba(
            feature_matrix
        )[:, 1]
    )

    latest_feature_df[
        "Predicted_Probability"
    ] = predicted_probability

    latest_feature_df["AI_Score"] = (
        latest_feature_df[
            "Predicted_Probability"
        ]
        * 100
    ).round(2)

    latest_feature_df["Signal"] = (
        latest_feature_df[
            "AI_Score"
        ].apply(assign_ai_signal)
    )

    latest_feature_df["Risk_Level"] = (
        latest_feature_df[
            "ATR_Ratio"
        ].apply(assign_risk_level)
    )

    latest_feature_df = (
        latest_feature_df
        .sort_values(
            "Predicted_Probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    latest_feature_df["Rank"] = (
        np.arange(
            1,
            len(latest_feature_df) + 1,
        )
    )

    ranking_columns = [
        "Rank",
        "date",
        "StockID",
        "StockName",
        "Industry",
        "Close",
        "AI_Score",
        "Predicted_Probability",
        "Signal",
        "Risk_Level",
        "RSI",
        "Volume_Ratio",
        "ATR_Ratio",
        "Foreign_NetBuy_5D_Ratio",
        "InvestmentTrust_NetBuy_5D_Ratio",
        "Foreign_Buy_Streak",
        "InvestmentTrust_Buy_Streak",
    ]

    missing_output_columns = [
        column
        for column in ranking_columns
        if column
        not in latest_feature_df.columns
    ]

    if missing_output_columns:
        raise RuntimeError(
            "排行榜缺少輸出欄位："
            f"{missing_output_columns}"
        )

    ranking_output_df = (
        latest_feature_df[
            ranking_columns
        ].copy()
    )

    industry_ranking_df = (
        build_industry_ranking(
            latest_feature_df
        )
    )

    if (
        industry_ranking_df[
            "Industry"
        ].nunique()
        != EXPECTED_INDUSTRY_COUNT
    ):
        raise RuntimeError(
            "族群排行榜不是完整 17 個族群。"
        )

    # 全部檢查完成後，才覆蓋網站檔案
    save_csv_atomic(
        ranking_output_df,
        LATEST_RANKING_PATH,
    )

    save_parquet_atomic(
        latest_feature_df,
        LATEST_FEATURE_PATH,
    )

    save_csv_atomic(
        industry_ranking_df,
        INDUSTRY_RANKING_PATH,
    )

    finished_at = datetime.now()

    price_dates = pd.to_datetime(
        update_record_df["PriceDate"],
        errors="coerce",
    )

    institution_dates = pd.to_datetime(
        update_record_df[
            "InstitutionDate"
        ],
        errors="coerce",
    )

    status_data = {
        "status": "success",
        "updated_at": (
            finished_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        "price_data_date": str(
            price_dates.max().date()
        ),
        "institution_data_date": str(
            institution_dates.max().date()
        ),
        "ranking_data_date_min": str(
            pd.to_datetime(
                ranking_output_df["date"]
            ).min().date()
        ),
        "ranking_data_date_max": str(
            pd.to_datetime(
                ranking_output_df["date"]
            ).max().date()
        ),
        "stock_count": int(
            ranking_output_df[
                "StockID"
            ].nunique()
        ),
        "industry_count": int(
            industry_ranking_df[
                "Industry"
            ].nunique()
        ),
        "duration_seconds": round(
            (
                finished_at
                - started_at
            ).total_seconds(),
            2,
        ),
        "message": (
            "Yahoo Finance、FinMind、"
            "AI排行榜與族群排行更新成功"
        ),
    }

    save_json_atomic(
        status_data,
        UPDATE_STATUS_PATH,
    )

    print("\n" + "=" * 70)
    print("每日更新成功")
    print(
        "股價最新日期：",
        status_data[
            "price_data_date"
        ],
    )
    print(
        "法人最新日期：",
        status_data[
            "institution_data_date"
        ],
    )
    print(
        "排行榜股票數：",
        status_data[
            "stock_count"
        ],
    )
    print(
        "執行秒數：",
        status_data[
            "duration_seconds"
        ],
    )
    print("=" * 70)


if __name__ == "__main__":
    main()