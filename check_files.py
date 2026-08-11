from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent

required_files = [
    PROJECT_ROOT / "models" / "multi_stock_xgb_5d.pkl",
    PROJECT_ROOT / "models" / "model_features.pkl",
    PROJECT_ROOT / "results" / "latest_rankings.csv",
    PROJECT_ROOT / "results" / "latest_features.parquet",
    PROJECT_ROOT / "results" / "industry_rankings.csv",
    PROJECT_ROOT / "config" / "stocks_85.csv",
]


print("網站檔案檢查")
print("=" * 60)

all_files_exist = True

for file_path in required_files:

    exists = file_path.exists()

    print(
        "存在" if exists else "缺少",
        file_path.relative_to(PROJECT_ROOT)
    )

    if not exists:
        all_files_exist = False


if not all_files_exist:
    raise FileNotFoundError(
        "部分核心檔案不存在，請檢查解壓縮位置。"
    )


ranking_df = pd.read_csv(
    PROJECT_ROOT
    / "results"
    / "latest_rankings.csv",
    dtype={"StockID": str}
)

industry_df = pd.read_csv(
    PROJECT_ROOT
    / "results"
    / "industry_rankings.csv"
)

latest_features_df = pd.read_parquet(
    PROJECT_ROOT
    / "results"
    / "latest_features.parquet"
)

feature_cols = joblib.load(
    PROJECT_ROOT
    / "models"
    / "model_features.pkl"
)

model = joblib.load(
    PROJECT_ROOT
    / "models"
    / "multi_stock_xgb_5d.pkl"
)


print("\n資料內容檢查")
print("=" * 60)
print("排行榜股票數：", ranking_df["StockID"].nunique())
print("排行榜資料筆數：", len(ranking_df))
print("族群排行榜數量：", len(industry_df))
print(
    "最新特徵股票數：",
    latest_features_df["StockID"].astype(str).nunique()
)
print("模型特徵數：", len(feature_cols))
print("模型類型：", type(model).__name__)


if ranking_df["StockID"].nunique() != 85:
    print(
        "警告：latest_rankings.csv 不是完整85檔。"
    )

if len(industry_df) != 17:
    print(
        "警告：industry_rankings.csv 不是完整17族群。"
    )

if len(feature_cols) != 32:
    print(
        "警告：模型特徵數不是32個。"
    )

print("\n核心檔案與資料載入成功")