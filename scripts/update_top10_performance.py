import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULT_DIR = PROJECT_ROOT / "results"
RAW_PRICE_DIR = PROJECT_ROOT / "data" / "raw_price"

RANKING_PATH = RESULT_DIR / "latest_rankings.csv"
SNAPSHOT_PATH = RESULT_DIR / "daily_top10_snapshots.csv"
PERFORMANCE_PATH = RESULT_DIR / "daily_top10_performance.csv"

TOP_N = 10
HOLDING_TRADING_DAYS = 5
TRADING_COST = 0.006

SNAPSHOT_COLUMNS = [
    "Recommendation_Date",
    "Rank",
    "StockID",
    "StockName",
    "Industry",
    "AI_Score",
    "Entry_Close",
    "Status",
    "Elapsed_Trading_Days",
    "Target_Date",
    "Exit_Close",
    "Actual_Return_5D",
    "Weight",
    "Contribution_Return",
    "Completed_Date",
]

PERFORMANCE_COLUMNS = [
    "Recommendation_Date",
    "Status",
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


def save_csv_atomic(dataframe, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_path)


def normalize_stock_id(series):
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )


def load_rankings():
    if not RANKING_PATH.exists():
        raise FileNotFoundError(f"找不到排行榜：{RANKING_PATH}")

    dataframe = pd.read_csv(RANKING_PATH, dtype={"StockID": str})
    required_columns = {
        "date", "Rank", "StockID", "StockName", "Industry", "Close", "AI_Score"
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"排行榜缺少必要欄位：{sorted(missing_columns)}")

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe["StockID"] = normalize_stock_id(dataframe["StockID"])
    for column in ["Rank", "Close", "AI_Score"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.dropna(
        subset=["date", "Rank", "StockID", "Close", "AI_Score"]
    )
    if dataframe.empty:
        raise ValueError("排行榜沒有有效資料。")

    latest_date = dataframe["date"].max()
    return (
        dataframe[dataframe["date"] == latest_date]
        .sort_values("Rank")
        .reset_index(drop=True)
    )


def load_existing_snapshots():
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    dataframe = pd.read_csv(
        SNAPSHOT_PATH,
        dtype={"StockID": str},
    )
    for column in SNAPSHOT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    dataframe["StockID"] = normalize_stock_id(dataframe["StockID"])
    for column in ["Recommendation_Date", "Target_Date", "Completed_Date"]:
        dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")

    numeric_columns = [
        "Rank", "AI_Score", "Entry_Close", "Elapsed_Trading_Days",
        "Exit_Close", "Actual_Return_5D", "Weight", "Contribution_Return",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe[SNAPSHOT_COLUMNS].copy()


def find_price_file(stock_id):
    patterns = [
        f"{stock_id}.csv",
        f"{stock_id}_*.csv",
        f"*{stock_id}*.csv",
        f"{stock_id}.parquet",
        f"{stock_id}_*.parquet",
        f"*{stock_id}*.parquet",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(RAW_PRICE_DIR.glob(pattern))
    unique_matches = sorted(set(matches))
    return unique_matches[0] if unique_matches else None


def load_price_history(stock_id):
    price_path = find_price_file(stock_id)
    if price_path is None:
        return pd.DataFrame()

    if price_path.suffix.lower() == ".parquet":
        dataframe = pd.read_parquet(price_path)
    else:
        dataframe = pd.read_csv(price_path)

    date_column = next(
        (column for column in ["date", "Date", "日期"] if column in dataframe.columns),
        None,
    )
    close_column = next(
        (column for column in ["Close", "close", "Adj Close", "Adj_Close", "收盤價"] if column in dataframe.columns),
        None,
    )
    if date_column is None or close_column is None:
        return pd.DataFrame()

    output = dataframe[[date_column, close_column]].copy()
    output.columns = ["date", "Close"]
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["Close"] = pd.to_numeric(output["Close"], errors="coerce")
    return (
        output.dropna(subset=["date", "Close"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def append_latest_top10(snapshots, rankings):
    recommendation_date = rankings["date"].max().normalize()
    top5 = rankings.head(TOP_N).copy()

    existing_keys = set(
        zip(
            snapshots["Recommendation_Date"].dt.strftime("%Y-%m-%d"),
            snapshots["StockID"],
        )
    ) if not snapshots.empty else set()

    new_rows = []
    for _, row in top5.iterrows():
        stock_id = str(row["StockID"]).zfill(4)
        key = (recommendation_date.strftime("%Y-%m-%d"), stock_id)
        if key in existing_keys:
            continue

        new_rows.append({
            "Recommendation_Date": recommendation_date,
            "Rank": int(row["Rank"]),
            "StockID": stock_id,
            "StockName": str(row["StockName"]),
            "Industry": str(row["Industry"]),
            "AI_Score": float(row["AI_Score"]),
            "Entry_Close": float(row["Close"]),
            "Status": "觀察中",
            "Elapsed_Trading_Days": 0,
            "Target_Date": pd.NaT,
            "Exit_Close": np.nan,
            "Actual_Return_5D": np.nan,
            "Weight": 1 / TOP_N,
            "Contribution_Return": np.nan,
            "Completed_Date": pd.NaT,
        })

    if new_rows:
        snapshots = pd.concat(
            [snapshots, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    return snapshots


def update_snapshot_results(snapshots):
    updated = snapshots.copy()

    for index, row in updated.iterrows():
        if row["Status"] == "已完成":
            continue

        recommendation_date = pd.Timestamp(row["Recommendation_Date"]).normalize()
        price_df = load_price_history(row["StockID"])
        if price_df.empty:
            updated.at[index, "Status"] = "資料不足"
            continue

        future_prices = price_df[price_df["date"] > recommendation_date].copy()
        elapsed_days = len(future_prices)
        updated.at[index, "Elapsed_Trading_Days"] = min(
            elapsed_days,
            HOLDING_TRADING_DAYS,
        )

        if elapsed_days < HOLDING_TRADING_DAYS:
            updated.at[index, "Status"] = "觀察中"
            continue

        exit_row = future_prices.iloc[HOLDING_TRADING_DAYS - 1]
        entry_close = pd.to_numeric(row["Entry_Close"], errors="coerce")
        exit_close = pd.to_numeric(exit_row["Close"], errors="coerce")

        if pd.isna(entry_close) or entry_close <= 0 or pd.isna(exit_close):
            updated.at[index, "Status"] = "資料不足"
            continue

        actual_return = exit_close / entry_close - 1
        updated.at[index, "Status"] = "已完成"
        updated.at[index, "Target_Date"] = exit_row["date"]
        updated.at[index, "Exit_Close"] = exit_close
        updated.at[index, "Actual_Return_5D"] = actual_return
        updated.at[index, "Contribution_Return"] = actual_return * row["Weight"]
        updated.at[index, "Completed_Date"] = exit_row["date"]

    return updated


def build_performance(snapshots):
    rows = []

    for recommendation_date, group in snapshots.groupby("Recommendation_Date", sort=True):
        completed = group[group["Status"] == "已完成"].copy()
        completed_count = len(completed)
        total_count = len(group)

        if completed_count == total_count and total_count == TOP_N:
            gross_return = completed["Actual_Return_5D"].mean()
            net_return = gross_return - TRADING_COST
            status = "已完成"
            win_count = int((completed["Actual_Return_5D"] > 0).sum())
            loss_count = int((completed["Actual_Return_5D"] < 0).sum())
            win_rate = win_count / total_count
        elif (group["Status"] == "資料不足").any():
            gross_return = np.nan
            net_return = np.nan
            status = "資料不足"
            win_count = np.nan
            loss_count = np.nan
            win_rate = np.nan
        else:
            gross_return = np.nan
            net_return = np.nan
            status = "觀察中"
            win_count = np.nan
            loss_count = np.nan
            win_rate = np.nan

        rows.append({
            "Recommendation_Date": recommendation_date,
            "Status": status,
            "Completed_Stocks": completed_count,
            "Total_Stocks": total_count,
            "Gross_Return": gross_return,
            "Trading_Cost": TRADING_COST if status == "已完成" else np.nan,
            "Net_Return": net_return,
            "Win_Count": win_count,
            "Loss_Count": loss_count,
            "Win_Rate": win_rate,
        })

    performance = pd.DataFrame(rows)
    if performance.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    performance = performance.sort_values("Recommendation_Date").reset_index(drop=True)
    completed_mask = performance["Status"] == "已完成"
    equity = 1.0
    equity_values = []

    for _, row in performance.iterrows():
        if row["Status"] == "已完成" and pd.notna(row["Net_Return"]):
            equity *= 1 + row["Net_Return"]
        equity_values.append(equity)

    performance["Net_Equity"] = equity_values
    performance["Running_Max_Equity"] = performance["Net_Equity"].cummax()
    performance["Drawdown"] = (
        performance["Net_Equity"] / performance["Running_Max_Equity"] - 1
    )
    performance.loc[~completed_mask, ["Net_Equity", "Running_Max_Equity", "Drawdown"]] = np.nan
    return performance[PERFORMANCE_COLUMNS]


def format_dates_for_csv(dataframe, columns):
    output = dataframe.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return output


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings()
    snapshots = load_existing_snapshots()
    snapshots = append_latest_top10(snapshots, rankings)
    snapshots = update_snapshot_results(snapshots)
    snapshots = snapshots.sort_values(
        ["Recommendation_Date", "Rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    performance = build_performance(snapshots)

    snapshot_output = format_dates_for_csv(
        snapshots,
        ["Recommendation_Date", "Target_Date", "Completed_Date"],
    )
    performance_output = format_dates_for_csv(
        performance,
        ["Recommendation_Date"],
    )

    save_csv_atomic(snapshot_output, SNAPSHOT_PATH)
    save_csv_atomic(performance_output, PERFORMANCE_PATH)

    latest_date = rankings["date"].max().strftime("%Y-%m-%d")
    latest_period = performance.iloc[-1] if not performance.empty else None

    print("=" * 70)
    print("AI Top 10 實績追蹤更新完成")
    print(f"排行榜日期：{latest_date}")
    print(f"快照總筆數：{len(snapshots)}")
    print(f"追蹤期數：{len(performance)}")
    if latest_period is not None:
        print(f"最新一期狀態：{latest_period['Status']}")
        print(
            "完成股票數："
            f"{int(latest_period['Completed_Stocks'])}/"
            f"{int(latest_period['Total_Stocks'])}"
        )
    print(f"已儲存：{SNAPSHOT_PATH}")
    print(f"已儲存：{PERFORMANCE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
