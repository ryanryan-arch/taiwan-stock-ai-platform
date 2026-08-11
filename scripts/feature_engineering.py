import numpy as np
import pandas as pd
import ta


INSTITUTIONAL_COLUMNS = [
    "Foreign_NetBuy",
    "InvestmentTrust_NetBuy",
    "Dealer_NetBuy",
    "Institutional_Total_NetBuy",
]


def calculate_buy_streak(series):
    """
    計算法人連續買超天數。

    買賣超大於 0：
        連續買超天數加 1

    買賣超小於或等於 0：
        連續買超天數歸零
    """

    streak_values = []
    current_streak = 0

    for value in series.fillna(0):

        if value > 0:
            current_streak += 1
        else:
            current_streak = 0

        streak_values.append(current_streak)

    return pd.Series(
        streak_values,
        index=series.index,
        dtype="int64",
    )


def validate_feature_input(dataframe):
    """
    檢查特徵工程需要的必要欄位。
    """

    required_columns = [
        "date",
        "StockID",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"特徵工程缺少必要欄位：{missing_columns}"
        )


def add_all_features(dataframe):
    """
    建立模型使用的技術面、量價面與法人籌碼特徵。

    此函式必須與 Colab 訓練模型時的特徵邏輯保持一致。
    """

    if dataframe is None or dataframe.empty:
        raise ValueError(
            "輸入資料為空，無法建立模型特徵。"
        )

    validate_feature_input(dataframe)

    source_df = dataframe.copy()

    source_df["date"] = pd.to_datetime(
        source_df["date"],
        errors="coerce",
    )

    source_df["StockID"] = (
        source_df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    numeric_price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_price_columns:

        if column in source_df.columns:
            source_df[column] = pd.to_numeric(
                source_df[column],
                errors="coerce",
            )

    # 若法人欄位不存在，先建立為 0
    for column in INSTITUTIONAL_COLUMNS:

        if column not in source_df.columns:
            source_df[column] = 0.0

        source_df[column] = pd.to_numeric(
            source_df[column],
            errors="coerce",
        ).fillna(0.0)

    source_df = (
        source_df
        .dropna(
            subset=[
                "date",
                "StockID",
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
        .sort_values(
            ["StockID", "date"]
        )
        .reset_index(drop=True)
    )

    result_frames = []

    for stock_id, stock_df in source_df.groupby(
        "StockID",
        sort=False,
    ):

        stock_df = (
            stock_df
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        close = pd.to_numeric(
            stock_df["Close"],
            errors="coerce",
        )

        high = pd.to_numeric(
            stock_df["High"],
            errors="coerce",
        )

        low = pd.to_numeric(
            stock_df["Low"],
            errors="coerce",
        )

        volume = pd.to_numeric(
            stock_df["Volume"],
            errors="coerce",
        )

        safe_close = close.replace(
            0,
            np.nan,
        )

        safe_volume = volume.replace(
            0,
            np.nan,
        )

        # ==========================================
        # 1. 價格動能特徵
        # ==========================================

        stock_df["Return_1D"] = (
            close.pct_change(1)
        )

        stock_df["Return_3D"] = (
            close.pct_change(3)
        )

        stock_df["Return_5D"] = (
            close.pct_change(5)
        )

        stock_df["Return_10D"] = (
            close.pct_change(10)
        )

        stock_df["Return_20D"] = (
            close.pct_change(20)
        )

        # ==========================================
        # 2. 移動平均線
        # ==========================================

        stock_df["MA5"] = (
            close.rolling(5).mean()
        )

        stock_df["MA10"] = (
            close.rolling(10).mean()
        )

        stock_df["MA20"] = (
            close.rolling(20).mean()
        )

        stock_df["MA60"] = (
            close.rolling(60).mean()
        )

        # ==========================================
        # 3. 趨勢與均線相對位置
        # ==========================================

        stock_df["Close_MA5_Ratio"] = (
            close
            / stock_df["MA5"]
            - 1
        )

        stock_df["Close_MA10_Ratio"] = (
            close
            / stock_df["MA10"]
            - 1
        )

        stock_df["Close_MA20_Ratio"] = (
            close
            / stock_df["MA20"]
            - 1
        )

        stock_df["Close_MA60_Ratio"] = (
            close
            / stock_df["MA60"]
            - 1
        )

        stock_df["MA5_MA20_Ratio"] = (
            stock_df["MA5"]
            / stock_df["MA20"]
            - 1
        )

        stock_df["MA20_MA60_Ratio"] = (
            stock_df["MA20"]
            / stock_df["MA60"]
            - 1
        )

        # ==========================================
        # 4. RSI
        # ==========================================

        rsi_indicator = ta.momentum.RSIIndicator(
            close=close,
            window=14,
        )

        stock_df["RSI"] = (
            rsi_indicator.rsi()
        )

        # ==========================================
        # 5. MACD
        # ==========================================

        macd_indicator = ta.trend.MACD(
            close=close,
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )

        stock_df["MACD"] = (
            macd_indicator.macd()
        )

        stock_df["MACD_Signal"] = (
            macd_indicator.macd_signal()
        )

        stock_df["MACD_Diff"] = (
            macd_indicator.macd_diff()
        )

        stock_df["MACD_Diff_Ratio"] = (
            stock_df["MACD_Diff"]
            / safe_close
        )

        # ==========================================
        # 6. ATR
        # ==========================================

        atr_indicator = (
            ta.volatility.AverageTrueRange(
                high=high,
                low=low,
                close=close,
                window=14,
            )
        )

        stock_df["ATR"] = (
            atr_indicator.average_true_range()
        )

        stock_df["ATR_Ratio"] = (
            stock_df["ATR"]
            / safe_close
        )

        # ==========================================
        # 7. 波動率
        # ==========================================

        stock_df["Volatility_5D"] = (
            stock_df["Return_1D"]
            .rolling(5)
            .std()
        )

        stock_df["Volatility_20D"] = (
            stock_df["Return_1D"]
            .rolling(20)
            .std()
        )

        # ==========================================
        # 8. 成交量特徵
        # ==========================================

        stock_df["Volume_MA5"] = (
            volume.rolling(5).mean()
        )

        stock_df["Volume_MA20"] = (
            volume.rolling(20).mean()
        )

        stock_df["Volume_Ratio"] = (
            volume
            / stock_df["Volume_MA20"]
            .replace(0, np.nan)
        )

        stock_df["Volume_5D_20D_Ratio"] = (
            stock_df["Volume_MA5"]
            / stock_df["Volume_MA20"]
            .replace(0, np.nan)
        )

        # ==========================================
        # 9. 法人多日累計買賣超
        # ==========================================

        stock_df["Foreign_NetBuy_5D"] = (
            stock_df["Foreign_NetBuy"]
            .rolling(5)
            .sum()
        )

        stock_df["Foreign_NetBuy_20D"] = (
            stock_df["Foreign_NetBuy"]
            .rolling(20)
            .sum()
        )

        stock_df[
            "InvestmentTrust_NetBuy_5D"
        ] = (
            stock_df["InvestmentTrust_NetBuy"]
            .rolling(5)
            .sum()
        )

        stock_df[
            "InvestmentTrust_NetBuy_20D"
        ] = (
            stock_df["InvestmentTrust_NetBuy"]
            .rolling(20)
            .sum()
        )

        stock_df["Dealer_NetBuy_5D"] = (
            stock_df["Dealer_NetBuy"]
            .rolling(5)
            .sum()
        )

        stock_df["Institutional_NetBuy_5D"] = (
            stock_df["Institutional_Total_NetBuy"]
            .rolling(5)
            .sum()
        )

        stock_df[
            "Institutional_NetBuy_20D"
        ] = (
            stock_df["Institutional_Total_NetBuy"]
            .rolling(20)
            .sum()
        )

        # ==========================================
        # 10. 法人單日買賣超比例
        # ==========================================

        stock_df["Foreign_NetBuy_Ratio"] = (
            stock_df["Foreign_NetBuy"]
            / safe_volume
        )

        stock_df[
            "InvestmentTrust_NetBuy_Ratio"
        ] = (
            stock_df["InvestmentTrust_NetBuy"]
            / safe_volume
        )

        stock_df["Dealer_NetBuy_Ratio"] = (
            stock_df["Dealer_NetBuy"]
            / safe_volume
        )

        stock_df[
            "Institutional_NetBuy_Ratio"
        ] = (
            stock_df["Institutional_Total_NetBuy"]
            / safe_volume
        )

        # ==========================================
        # 11. 法人多日買賣超比例
        # ==========================================

        volume_5d_sum = (
            volume
            .rolling(5)
            .sum()
            .replace(0, np.nan)
        )

        volume_20d_sum = (
            volume
            .rolling(20)
            .sum()
            .replace(0, np.nan)
        )

        stock_df[
            "Foreign_NetBuy_5D_Ratio"
        ] = (
            stock_df["Foreign_NetBuy_5D"]
            / volume_5d_sum
        )

        stock_df[
            "Foreign_NetBuy_20D_Ratio"
        ] = (
            stock_df["Foreign_NetBuy_20D"]
            / volume_20d_sum
        )

        stock_df[
            "InvestmentTrust_NetBuy_5D_Ratio"
        ] = (
            stock_df[
                "InvestmentTrust_NetBuy_5D"
            ]
            / volume_5d_sum
        )

        stock_df[
            "InvestmentTrust_NetBuy_20D_Ratio"
        ] = (
            stock_df[
                "InvestmentTrust_NetBuy_20D"
            ]
            / volume_20d_sum
        )

        stock_df[
            "Dealer_NetBuy_5D_Ratio"
        ] = (
            stock_df["Dealer_NetBuy_5D"]
            / volume_5d_sum
        )

        stock_df[
            "Institutional_NetBuy_5D_Ratio"
        ] = (
            stock_df[
                "Institutional_NetBuy_5D"
            ]
            / volume_5d_sum
        )

        stock_df[
            "Institutional_NetBuy_20D_Ratio"
        ] = (
            stock_df[
                "Institutional_NetBuy_20D"
            ]
            / volume_20d_sum
        )

        # ==========================================
        # 12. 法人連續買超天數
        # ==========================================

        stock_df["Foreign_Buy_Streak"] = (
            calculate_buy_streak(
                stock_df["Foreign_NetBuy"]
            )
        )

        stock_df[
            "InvestmentTrust_Buy_Streak"
        ] = (
            calculate_buy_streak(
                stock_df[
                    "InvestmentTrust_NetBuy"
                ]
            )
        )

        stock_df["Dealer_Buy_Streak"] = (
            calculate_buy_streak(
                stock_df["Dealer_NetBuy"]
            )
        )

        result_frames.append(stock_df)

    if not result_frames:
        raise ValueError(
            "特徵工程沒有產生任何股票資料。"
        )

    result_df = pd.concat(
        result_frames,
        ignore_index=True,
    )

    result_df = result_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result_df = (
        result_df
        .sort_values(
            ["date", "StockID"]
        )
        .reset_index(drop=True)
    )

    return result_df


def get_latest_valid_features(
    feature_dataframe,
    feature_columns,
):
    """
    從完整特徵資料中，
    取得每檔股票最新且 32 個特徵均完整的一筆資料。
    """

    if feature_dataframe is None:
        raise ValueError(
            "特徵資料不存在。"
        )

    if feature_dataframe.empty:
        raise ValueError(
            "特徵資料為空。"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in feature_dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"缺少模型特徵：{missing_features}"
        )

    latest_valid_df = (
        feature_dataframe
        .dropna(
            subset=feature_columns
        )
        .sort_values(
            ["StockID", "date"]
        )
        .groupby(
            "StockID",
            group_keys=False,
        )
        .tail(1)
        .copy()
    )

    latest_valid_df = (
        latest_valid_df
        .sort_values("StockID")
        .reset_index(drop=True)
    )

    return latest_valid_df