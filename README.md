# 台股 AI 智慧選股與五日趨勢預測平台

本專題整合 Yahoo Finance 股價資料與 FinMind 三大法人籌碼資料，
建立技術面、量價面與法人籌碼面特徵，
並使用 XGBoost 預測個股未來五個交易日是否上漲超過 1%。

系統涵蓋 17 個產業族群，共 85 檔台股，
每天依據模型產生的 AI 分數進行排序，
提供每日 AI 排行榜、個股分析、法人籌碼、族群排行與策略回測功能。

---

## 專題特色

- 85 檔台股 AI 智慧排行榜
- 17 個產業族群強弱比較
- Yahoo Finance 日線價格與成交量
- FinMind 三大法人籌碼資料
- 32 個技術面、量價面及法人特徵
- XGBoost 五日趨勢分類模型
- TimeSeriesSplit 五折時間序列驗證
- OOF 樣本外預測
- AI Top 5 投資組合回測
- 交易成本與最大回撤分析
- Streamlit 多頁式互動網站

---

## 網站功能

### 今日 AI 排行榜

- 顯示最新 85 檔股票排名
- 顯示每日 AI Top 10
- 依產業族群篩選
- 搜尋股票代碼或名稱
- 顯示 AI 分數、模型訊號與風險等級

### 個股分析

- AI 排名與 AI 分數
- 最新收盤價
- K 線圖
- MA5、MA20、MA60
- RSI、ATR 與成交量比率
- 技術面與法人籌碼訊號解讀

### 法人籌碼

- 外資、投信與自營商每日買賣超
- 外資與投信五日買賣超比例
- 法人連續買超天數
- 近 20、60、120 日法人籌碼圖表

### 族群排行

- 17 個產業族群 AI 強度排行
- 族群平均 AI 分數
- 族群代表股
- 族群內五檔股票排名

### 模型回測

- AI Top 5 累積資產曲線
- 可用股票等權基準
- 年化報酬
- 年化波動率
- 最大回撤
- Sharpe Ratio
- TimeSeriesSplit 分期績效
- 五折分類評估結果

### 模型說明

- 資料來源
- Target 定義
- 32 個模型特徵
- XGBoost 模型方法
- TimeSeriesSplit 驗證方式
- OOF 預測與回測方法
- 模型限制與風險說明

---

## 資料來源

### Yahoo Finance

使用 Yahoo Finance 取得：

- 開盤價
- 最高價
- 最低價
- 收盤價
- 成交量
- 交易日期

### FinMind

使用 FinMind 取得：

- 外資買賣超
- 投信買賣超
- 自營商買賣超
- 三大法人合計買賣超

---

## 預測目標

模型預測個股未來五個交易日的報酬是否超過 1%。

```text
Target = 1
未來五日報酬率 > 1%

Target = 0
未來五日報酬率 <= 1%
```

未來五日報酬率：

```text
未來五日報酬率
= 未來第 5 個交易日收盤價 ÷ 當日收盤價 - 1
```

Target 為 0 不一定代表股票下跌。

---

## 模型特徵

模型共使用 32 個特徵，主要分為：

- 價格動能特徵
- 趨勢與均線特徵
- 技術指標與波動特徵
- 成交量特徵
- 法人單日籌碼特徵
- 法人累計籌碼特徵
- 法人連續買超特徵

主要指標包含：

```text
Return
MA
RSI
MACD
ATR
Volatility
Volume Ratio
Foreign Net Buy
Investment Trust Net Buy
Dealer Net Buy
Institutional Buy Streak
```

---

## 模型方法

本專題使用 XGBoost 分類模型。

主要參數：

```text
n_estimators = 400
max_depth = 4
learning_rate = 0.03
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 1.0
objective = binary:logistic
```

---

## 時間序列驗證

股票資料具有明確時間順序，因此不使用隨機切分。

本專題使用：

```text
TimeSeriesSplit
n_splits = 5
gap = 5
```

模型依不重複交易日期切分，確保：

- 訓練資料早於驗證資料
- 同一日期的股票不會被拆到訓練與驗證兩側
- 訓練與驗證期間保留五個交易日間隔
- 降低未來五日 Target 重疊造成的資料洩漏

---

## 回測策略

```text
股票範圍：85 檔
選股方式：AI 分數由高到低
持股數量：Top 5
資金配置：五檔等權
重新平衡：每五個交易日
持有期間：五個交易日
完整交易成本：0.6%
預測來源：TimeSeriesSplit OOF 預測
比較基準：同期間可用股票等權策略
```

### 主要回測成果

```text
AI Top 5 累積報酬：約 260.12%
AI Top 5 年化報酬：約 28.69%
AI Top 5 Sharpe Ratio：約 0.830
AI Top 5 最大回撤：約 -53.17%
```

歷史回測結果不代表未來能取得相同績效。

---

## AI 分數說明

AI 分數是 XGBoost 根據 32 個特徵產生的分類分數，再乘以 100。

```text
AI 分數
= XGBoost Target 1 模型分數 × 100
```

分數主要用於 85 檔股票之間的相對排序。

```text
70 分以上：高分候選
60 至 70 分：偏多觀察
50 至 60 分：中性觀察
50 分以下：暫不列入
```

AI 分數尚未經過機率校準，因此不等於真實上漲機率。

---

## 專案結構

```text
AI_Stock_Web/
├── config/
│   └── stocks_85.csv
├── data/
│   ├── merged/
│   ├── processed/
│   ├── raw_institution/
│   └── raw_price/
├── models/
│   ├── model_features.pkl
│   └── multi_stock_xgb_5d.pkl
├── pages/
│   ├── 1_個股分析.py
│   ├── 2_法人籌碼.py
│   ├── 3_族群排行.py
│   ├── 4_模型回測.py
│   └── 5_模型說明.py
├── results/
│   ├── latest_rankings.csv
│   ├── latest_features.parquet
│   ├── industry_rankings.csv
│   ├── portfolio_kpis.csv
│   └── top5_portfolio_backtest.csv
├── 今日_AI_排行榜.py
├── check_files.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 本機執行方式

### 建立虛擬環境

```powershell
py -3.11 -m venv .venv
```

### 啟用虛擬環境

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 安裝套件

```powershell
python -m pip install -r requirements.txt
```

### 啟動網站

```powershell
python -m streamlit run ".\今日_AI_排行榜.py"
```

網站預設網址：

```text
http://localhost:8501
```

---

## 使用技術

- Python 3.11
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- TimeSeriesSplit
- Yahoo Finance
- FinMind
- Streamlit
- Plotly
- Joblib
- PyArrow

---

## 模型限制

- 股票市場具有高度雜訊
- 歷史規律不保證未來持續有效
- 模型主要價值在相對排序
- AI 分數不是保證上漲機率
- Top 5 集中策略具有較高波動
- 最大回撤超過 50%
- 法人資料可能晚於股價資料發布
- 部分新掛牌股票的歷史資料較短
- 模型尚未納入新聞、財報及總體經濟資料

---

## 免責聲明

本平台僅供課程專題、資料分析及模型研究。

網站內容不構成投資建議、買賣推薦或獲利保證。
投資人應自行評估投資風險並承擔投資決策結果。