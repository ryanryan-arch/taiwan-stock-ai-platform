import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ==================================================
# 專案路徑
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_PATH = (
    MODEL_DIR
    / "multi_stock_xgb_5d.pkl"
)

FEATURE_LIST_PATH = (
    MODEL_DIR
    / "model_features.pkl"
)

LATEST_FEATURE_PATH = (
    RESULT_DIR
    / "latest_features.parquet"
)

LATEST_RANKING_PATH = (
    RESULT_DIR
    / "latest_rankings.csv"
)

SHAP_VALUE_PATH = (
    RESULT_DIR
    / "latest_shap_values.parquet"
)

GLOBAL_IMPORTANCE_PATH = (
    RESULT_DIR
    / "global_shap_importance.csv"
)

SHAP_STATUS_PATH = (
    RESULT_DIR
    / "shap_status.json"
)


EXPECTED_STOCK_COUNT = 85
EXPECTED_FEATURE_COUNT = 32


# ==================================================
# 安全儲存函式
# ==================================================

def save_parquet_atomic(
    dataframe,
    output_path,
):
    """
    先寫入暫存檔，再取代正式檔案。
    """

    temp_path = output_path.with_suffix(
        ".tmp.parquet"
    )

    dataframe.to_parquet(
        temp_path,
        index=False,
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


def save_json_atomic(
    data,
    output_path,
):
    """
    安全儲存 JSON。
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


# ==================================================
# 載入模型與資料
# ==================================================

def load_required_files():
    """
    載入模型、特徵清單與最新 85 檔資料。
    """

    required_paths = [
        MODEL_PATH,
        FEATURE_LIST_PATH,
        LATEST_FEATURE_PATH,
        LATEST_RANKING_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "缺少 SHAP 必要檔案："
            f"{missing_paths}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_LIST_PATH
    )

    latest_feature_df = pd.read_parquet(
        LATEST_FEATURE_PATH
    )

    ranking_df = pd.read_csv(
        LATEST_RANKING_PATH,
        dtype={"StockID": str},
    )

    latest_feature_df["StockID"] = (
        latest_feature_df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    ranking_df["StockID"] = (
        ranking_df["StockID"]
        .astype(str)
        .str.zfill(4)
    )

    if "date" in latest_feature_df.columns:
        latest_feature_df["date"] = (
            pd.to_datetime(
                latest_feature_df["date"],
                errors="coerce",
            )
        )

    if (
        len(feature_columns)
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "模型特徵應為 "
            f"{EXPECTED_FEATURE_COUNT} 個，"
            f"目前為 {len(feature_columns)} 個。"
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature
        not in latest_feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            "最新資料缺少模型特徵："
            f"{missing_features}"
        )

    stock_count = (
        latest_feature_df["StockID"]
        .nunique()
    )

    if stock_count != EXPECTED_STOCK_COUNT:
        raise ValueError(
            "最新特徵應有 "
            f"{EXPECTED_STOCK_COUNT} 檔股票，"
            f"目前為 {stock_count} 檔。"
        )

    return (
        model,
        feature_columns,
        latest_feature_df,
        ranking_df,
    )


# ==================================================
# SHAP 計算
# ==================================================

def calculate_shap_results(
    model,
    feature_columns,
    latest_feature_df,
):
    """
    計算最新 85 檔、32 個特徵的 SHAP 值。
    """

    feature_matrix = (
        latest_feature_df[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
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

        raise ValueError(
            "SHAP 輸入仍有缺失值："
            f"{missing_counts}"
        )

    print("建立 SHAP TreeExplainer")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation=(
            "tree_path_dependent"
        ),
        model_output="raw",
    )

    print(
        "開始計算最新 85 檔 SHAP 值"
    )

    shap_explanation = explainer(
        feature_matrix
    )

    shap_values = np.asarray(
        shap_explanation.values
    )

    # 處理部分版本二元分類的三維輸出
    if shap_values.ndim == 3:

        if shap_values.shape[-1] == 2:
            shap_values = (
                shap_values[:, :, 1]
            )

        elif shap_values.shape[0] == 2:
            shap_values = (
                shap_values[1]
            )

        else:
            raise ValueError(
                "無法判斷 SHAP 三維輸出格式："
                f"{shap_values.shape}"
            )

    expected_shape = (
        len(latest_feature_df),
        len(feature_columns),
    )

    if shap_values.shape != expected_shape:
        raise ValueError(
            "SHAP 結果形狀不正確，"
            f"預期 {expected_shape}，"
            f"實際 {shap_values.shape}。"
        )

    metadata_columns = [
        column
        for column in [
            "date",
            "StockID",
            "StockName",
            "Industry",
            "AI_Score",
            "Predicted_Probability",
            "Rank",
        ]
        if column
        in latest_feature_df.columns
    ]

    metadata_df = (
        latest_feature_df[
            metadata_columns
        ]
        .reset_index(drop=True)
        .copy()
    )

    feature_value_df = (
        feature_matrix
        .reset_index(drop=True)
        .copy()
    )

    shap_wide_df = pd.DataFrame(
        shap_values,
        columns=feature_columns,
    )

    long_frames = []

    for feature_name in feature_columns:

        feature_frame = (
            metadata_df.copy()
        )

        feature_frame["Feature"] = (
            feature_name
        )

        feature_frame["Feature_Value"] = (
            feature_value_df[
                feature_name
            ].to_numpy()
        )

        feature_frame["SHAP_Value"] = (
            shap_wide_df[
                feature_name
            ].to_numpy()
        )

        feature_frame["Abs_SHAP_Value"] = (
            feature_frame[
                "SHAP_Value"
            ].abs()
        )

        feature_frame[
            "Impact_Direction"
        ] = np.where(
            (
                feature_frame[
                    "SHAP_Value"
                ] > 0
            ),
            "推升分數",
            np.where(
                (
                    feature_frame[
                        "SHAP_Value"
                    ] < 0
                ),
                "壓低分數",
                "影響中性",
            ),
        )

        long_frames.append(
            feature_frame
        )

    shap_long_df = pd.concat(
        long_frames,
        ignore_index=True,
    )

    shap_long_df = (
        shap_long_df
        .sort_values(
            [
                "StockID",
                "Abs_SHAP_Value",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    global_importance_df = (
        shap_long_df
        .groupby(
            "Feature",
            as_index=False,
        )
        .agg(
            Mean_Abs_SHAP=(
                "Abs_SHAP_Value",
                "mean",
            ),
            Mean_SHAP=(
                "SHAP_Value",
                "mean",
            ),
            Positive_Count=(
                "SHAP_Value",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            Negative_Count=(
                "SHAP_Value",
                lambda values: int(
                    (values < 0).sum()
                ),
            ),
        )
        .sort_values(
            "Mean_Abs_SHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    global_importance_df[
        "Importance_Rank"
    ] = np.arange(
        1,
        len(global_importance_df) + 1,
    )

    importance_total = (
        global_importance_df[
            "Mean_Abs_SHAP"
        ].sum()
    )

    global_importance_df[
        "Importance_Percent"
    ] = (
        global_importance_df[
            "Mean_Abs_SHAP"
        ]
        / importance_total
        * 100
    )

    return (
        shap_long_df,
        global_importance_df,
    )


# ==================================================
# 主程式
# ==================================================

def main():

    started_at = datetime.now()

    print("=" * 70)
    print("開始計算最新 SHAP 模型解釋")
    print(
        "執行時間：",
        started_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )
    print("=" * 70)

    (
        model,
        feature_columns,
        latest_feature_df,
        ranking_df,
    ) = load_required_files()

    (
        shap_long_df,
        global_importance_df,
    ) = calculate_shap_results(
        model=model,
        feature_columns=feature_columns,
        latest_feature_df=(
            latest_feature_df
        ),
    )

    save_parquet_atomic(
        shap_long_df,
        SHAP_VALUE_PATH,
    )

    save_csv_atomic(
        global_importance_df,
        GLOBAL_IMPORTANCE_PATH,
    )

    finished_at = datetime.now()

    status_data = {
        "status": "success",
        "updated_at": (
            finished_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        "stock_count": int(
            shap_long_df[
                "StockID"
            ].nunique()
        ),
        "feature_count": int(
            shap_long_df[
                "Feature"
            ].nunique()
        ),
        "shap_row_count": int(
            len(shap_long_df)
        ),
        "duration_seconds": round(
            (
                finished_at
                - started_at
            ).total_seconds(),
            2,
        ),
        "model_output": "raw",
        "explanation_note": (
            "正 SHAP 值推升 Target 1 的模型輸出，"
            "負 SHAP 值壓低 Target 1 的模型輸出。"
        ),
    }

    save_json_atomic(
        status_data,
        SHAP_STATUS_PATH,
    )

    print("\nSHAP 計算完成")
    print(
        "股票數：",
        status_data["stock_count"],
    )
    print(
        "特徵數：",
        status_data["feature_count"],
    )
    print(
        "SHAP 資料列數：",
        status_data["shap_row_count"],
    )
    print(
        "全域最重要特徵：",
        global_importance_df.iloc[0][
            "Feature"
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