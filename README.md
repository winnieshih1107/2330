# 台積電 (2330.TW) 股價預測與比較：Linear AR vs. ARIMA

本專案使用 `yfinance` 獲取台積電（2330.TW）歷史股價資料，並實作與比較兩種時間序列模型在**單步滾動預測（1-step-ahead rolling forecast）**上的表現：
1. **sklearn Linear Regression (Lag = 20)**：使用過去 20 天的收盤價作為特徵進行線性自迴歸。
2. **statsmodels ARIMA(20, 1, 0)**：自迴歸整合移動平均模型。

---

## 📊 預測結果比較

依據最近一年的歷史數據訓練與測試（最後 20% 作為測試集），兩模型的表現如下：

| 模型 | RMSE (均方根誤差) | R² Score (決定係數) |
| :--- | :---: | :---: |
| **Linear AR (Lag=20)** | 52.33 | 0.7656 |
| **ARIMA(20, 1, 0)** | 53.18 | 0.7579 |

*註：RMSE 越低越好，R² 越接近 1 越好。在此數據集下，簡易的 Linear AR 表現略優於 ARIMA。*

### 📈 預測走勢圖
在執行程式後會產生預測結果對比圖 `comparison.png`：

![預測走勢圖](comparison.png)

---

## 🛠️ 安裝與執行說明

### 1. 安裝依賴套件
執行前請確認已安裝以下 Python 套件：
```bash
pip install numpy pandas yfinance scikit-learn statsmodels matplotlib
```

### 2. 執行預測程式
```bash
python predict_compare.py
```
執行後將會：
- 下載最新一年的 2330.TW 歷史數據。
- 進行模型訓練與預測。
- 於終端機輸出預測的評估指標以及明日股價預測值。
- 輸出視覺化圖表 `comparison.png`。

---

## 📁 專案檔案結構
* `predict_compare.py`：主程式邏輯。
* `comparison.png`：最新的預測比較圖表。
* `CLAUDE.md`：開發與編碼規範指引。
* `log.md`：對話 Prompt 與模型疊代歷程紀錄。
* `README.md`：專案說明文件（本檔案）。