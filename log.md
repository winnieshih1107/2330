# Session Log — 2026-06-15

## 任務概述
使用 Auto Regression（sklearn）與 ARIMA 模型預測台積電（2330.TW）收盤價，比較兩者表現，並推送至 GitHub。

---

## Prompt 紀錄

### Prompt 1
**使用者：**
> 提供一段 Python 程式碼，使用 `yfinance` 下載 2330.TW 一年資料，分別建立：
> - sklearn `LinearRegression`（Lag=20）
> - `ARIMA(20, 1, 0)`
>
> 比較 RMSE、R²，並預測明日股價。

**處理：**
- 安裝缺少的套件 `yfinance`
- 修正 yfinance 1.x 回傳 MultiIndex 欄位（需 `droplevel(1)`）
- 修正 ARIMA `extend()` 使用 `.values` 避免 DatetimeIndex 頻率不符的錯誤
- 修正 Windows CP950 下 `R²` 符號 UnicodeEncodeError（改為 `R2`）
- 執行成功，產生比較圖表 `comparison.png`

**關鍵技術決策：**
原始程式碼將 sklearn 的 1-step-ahead rolling 預測與 ARIMA 的 multi-step dynamic 預測相比，方法不對等。
→ 統一改為兩個模型皆使用 **1-step-ahead rolling** 預測進行公平比較。

---

### Prompt 2
**使用者：**
> `D:\wi\L10_2330` 存在這（指定工作目錄）

**處理：**
- 將 `predict_compare.py` 寫入 `D:\wi\L10_2330`
- 在該目錄下執行腳本，確認成功

---

### Prompt 3
**使用者：**
> 讀取 `D:\wi\L10_2330\CLAUDE.md`，校正成果

**CLAUDE.md 核心準則：**
1. Think Before Coding — 明確假設，遇到歧義先問
2. Simplicity First — 最少程式碼解決問題，不寫多餘功能
3. Surgical Changes — 只動必要的部分，不改無關的程式碼
4. Goal-Driven Execution — 定義可驗證的成功標準

**校正內容：**

| 項目 | 原始版本 | 校正後 |
|---|---|---|
| 程式碼行數 | 177 行 | 75 行 |
| 不必要的 `def main()` 包裝 | ✗ 有 | ✓ 移除 |
| Dynamic（多步）預測比較 | ✗ 有（未被要求） | ✓ 移除 |
| MAE 指標 | ✗ 有（未被要求） | ✓ 移除 |
| 雙路徑存圖 | ✗ 有（多餘） | ✓ 移除 |
| 保留原始中文註解風格 | ✗ 未保留 | ✓ 保留 |

---

### Prompt 4
**使用者：**
> Continue

**處理：**
- 執行校正後的 `predict_compare.py`，確認成功
- 輸出結果：

| 模型 | RMSE | R² Score |
|---|---|---|
| sklearn (Linear AR, Lag=20) | 52.33 | 0.7656 |
| ARIMA(20, 1, 0) | 53.18 | 0.7579 |

**明日預測（2026-06-16）：**

| | 價格 (TWD) |
|---|---|
| 最新收盤 (2026-06-15) | 2375.00 |
| sklearn 預測 | 2371.06 |
| ARIMA 預測 | 2362.92 |

---

### Prompt 5
**使用者：**
> 成果推到 https://github.com/winnieshih1107/2330

**處理：**
- `git init` 初始化本地儲存庫
- `git remote add origin https://github.com/winnieshih1107/2330.git`
- `git add predict_compare.py comparison.png CLAUDE.md`

---

### Prompt 6
**使用者：**
> Continue

**處理：**
- `git commit -m "Add TSMC 2330.TW price prediction: Linear AR vs ARIMA(20,1,0) comparison"`
- `git push -u origin master` → 推送成功 ✅

**已上傳檔案：**
- `predict_compare.py` — 主程式
- `comparison.png` — 預測結果圖表
- `CLAUDE.md` — 程式碼撰寫準則

---

### Prompt 7
**使用者：**
> 整理今天 prompt，整理成 log.md

**處理：**
- 整理本 session 所有 prompt 與處理結果，輸出為 `log.md`

---

## 產出檔案

| 檔案 | 路徑 |
|---|---|
| 主程式 | `D:\wi\L10_2330\predict_compare.py` |
| 比較圖表 | `D:\wi\L10_2330\comparison.png` |
| 準則文件 | `D:\wi\L10_2330\CLAUDE.md` |
| Session Log | `D:\wi\L10_2330\log.md` |
| GitHub | https://github.com/winnieshih1107/2330 |
