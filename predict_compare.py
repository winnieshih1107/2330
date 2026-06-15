import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. 資料獲取與準備
# ==========================================
ticker = "2330.TW"
df = yf.download(ticker, period="1y")
df.columns = df.columns.droplevel(1)  # yfinance 1.x 返回 MultiIndex，需降維
prices = df["Close"].dropna()

lag = 20

# 建立 sklearn 所需的特徵矩陣
data = pd.DataFrame({"Target": prices})
for i in range(1, lag + 1):
    data[f"Lag_{i}"] = prices.shift(i)
data = data.dropna()

X = data[[f"Lag_{i}" for i in range(1, lag + 1)]]
y = data["Target"]

# 切分訓練集與測試集 (最後 20% 作為測試)
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

train_series = prices.loc[:y_train.index[-1]]
test_series  = prices.loc[y_test.index[0]:y_test.index[-1]]

# ==========================================
# 2. 模型一：sklearn Linear Regression (Lag 20)
#    使用 1-step-ahead rolling 預測（與 ARIMA 公平對齊）
# ==========================================
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
y_pred_sk = pd.Series(sk_model.predict(X_test), index=y_test.index)

# ==========================================
# 3. 模型二：ARIMA(20, 1, 0)
#    使用 extend() 進行 1-step-ahead rolling 預測
#    ─ 固定在訓練集上估計的係數，逐步注入真實觀測值
#    ─ 對應 sklearn 的 predict(X_test) 行為，確保公平比較
# ==========================================
arima_result = ARIMA(train_series, order=(20, 1, 0)).fit()

# extend() 接受 numpy array 以避免 DatetimeIndex 頻率不符的錯誤
extended = arima_result.extend(test_series.values)
y_pred_arima = pd.Series(
    extended.predict(start=0, end=len(test_series) - 1).values,
    index=y_test.index
)

# ==========================================
# 4. 效能評估與對比
# ==========================================
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2

rmse_sk,    r2_sk    = metrics(y_test, y_pred_sk)
rmse_arima, r2_arima = metrics(y_test, y_pred_arima)

performance_df = pd.DataFrame({
    "Model":         ["sklearn (Linear AR)", "statsmodels (ARIMA)"],
    "Config":        [f"Lag={lag}",          "order=(20, 1, 0)"],
    "RMSE":          [f"{rmse_sk:.2f}",      f"{rmse_arima:.2f}"],
    "R2 Score":      [f"{r2_sk:.4f}",        f"{r2_arima:.4f}"]
})

print("\n=== 模型表現橫向對比 (1-Step-Ahead Rolling) ===")
print(performance_df.to_string(index=False))

# ==========================================
# 5. 明日與未來 30 天股價預測
# ==========================================
# 在全量數據上重新擬合模型以進行未來預測
sk_full = LinearRegression()
sk_full.fit(X, y)

arima_full = ARIMA(prices, order=(20, 1, 0)).fit()

# A. 明日預測 (1-Step-Ahead)
latest_data_sk = prices.iloc[-lag:].values[::-1]
next_day_sk    = sk_full.predict(pd.DataFrame([latest_data_sk], columns=X.columns))[0]
next_day_arima = arima_full.forecast(steps=1).iloc[0]

print("\n=== 明日收盤價預測對比 ===")
print(f"最新實際收盤價 ({prices.index[-1].strftime('%Y-%m-%d')}): {prices.iloc[-1]:.2f}")
print(f"sklearn 預測明日價格: {next_day_sk:.2f}")
print(f"ARIMA  預測明日價格: {next_day_arima:.2f}")

# B. 未來 30 天預測 (Multi-Step Future Forecast)
# 生成未來 30 個工作日的時間序列
start_date = prices.index[-1] + pd.Timedelta(days=1)
future_dates = pd.bdate_range(start=start_date, periods=30)

# ARIMA 未來 30 天預測與信賴區間
arima_forecast_res = arima_full.get_forecast(steps=30)
arima_forecast_mean = pd.Series(arima_forecast_res.predicted_mean.values, index=future_dates)
ci_95 = pd.DataFrame(arima_forecast_res.conf_int(alpha=0.05).values, index=future_dates)
ci_68 = pd.DataFrame(arima_forecast_res.conf_int(alpha=0.32).values, index=future_dates)

# sklearn 未來 30 天遞迴預測 (Recursive Forecast)
last_lags = list(prices.iloc[-lag:].values[::-1])
sk_forecasts = []
for _ in range(30):
    pred = sk_full.predict(pd.DataFrame([last_lags], columns=X.columns))[0]
    sk_forecasts.append(pred)
    last_lags = [pred] + last_lags[:-1]
sk_forecast_mean = pd.Series(sk_forecasts, index=future_dates)

# 輸出未來 30 天預測結果表格
forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%Y-%m-%d"),
    "sklearn AR": sk_forecast_mean.values,
    "ARIMA Mean": arima_forecast_mean.values,
    "ARIMA 68% CI Lower": ci_68.iloc[:, 0].values,
    "ARIMA 68% CI Upper": ci_68.iloc[:, 1].values,
    "ARIMA 95% CI Lower": ci_95.iloc[:, 0].values,
    "ARIMA 95% CI Upper": ci_95.iloc[:, 1].values
})
print("\n=== 未來 30 天股價預測與信賴區間 ===")
print(forecast_df.to_string(index=False, max_rows=15))

# ==========================================
# 6. 視覺化：1-Step-Ahead 驗證對比圖 (存為 comparison.png)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(12, 5))

window = prices.iloc[-(len(y_test) + 40):]
ax1.plot(window.index, window.values, label="Actual Price", color="#1e293b", linewidth=1.5)
ax1.plot(y_test.index, y_pred_sk,    label="sklearn (Linear AR)", color="#3b82f6", linestyle="--", linewidth=1.5)
ax1.plot(y_test.index, y_pred_arima, label="ARIMA(20,1,0)",       color="#10b981", linestyle=":",  linewidth=2)

ax1.set_title("TSMC (2330.TW) — 1-Step-Ahead Prediction: Linear AR vs ARIMA(20,1,0)",
             fontsize=13, fontweight="bold")
ax1.set_ylabel("Price (TWD)")
ax1.set_xlabel("Date")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nValidation plot saved: comparison.png")

# ==========================================
# 7. 視覺化：未來 30 天預測信賴區間喇叭圖 (存為 future_forecast.png)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

# 取歷史最後 60 天數據進行視覺化對比，讓喇叭圖細節更明顯
hist_window = prices.iloc[-60:]
ax2.plot(hist_window.index, hist_window.values, label="Historical Price", color="#1e293b", linewidth=2)

# 為確保圖表平滑無縫銜接，預測線與信賴區間皆加入最後一天歷史收盤價作為起點
plot_dates = pd.Index([prices.index[-1]]).append(future_dates)
plot_arima_mean = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(arima_forecast_mean)
plot_sk_mean = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(sk_forecast_mean)

plot_ci68_low = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(ci_68.iloc[:, 0])
plot_ci68_high = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(ci_68.iloc[:, 1])
plot_ci95_low = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(ci_95.iloc[:, 0])
plot_ci95_high = pd.Series([prices.iloc[-1]], index=[prices.index[-1]])._append(ci_95.iloc[:, 1])

# 繪製預測中心線
ax2.plot(plot_dates, plot_arima_mean, label="ARIMA Forecast Mean", color="#10b981", linewidth=2)
ax2.plot(plot_dates, plot_sk_mean, label="sklearn AR Forecast Mean", color="#3b82f6", linestyle="--", linewidth=1.5)

# 繪製 68% 與 95% 信賴區間的「喇叭圖」 (Fan Chart)
ax2.fill_between(plot_dates, plot_ci68_low, plot_ci68_high, color="#10b981", alpha=0.3, label="68% Confidence Interval")
ax2.fill_between(plot_dates, plot_ci95_low, plot_ci95_high, color="#10b981", alpha=0.15, label="95% Confidence Interval")

# 設定圖表標籤與設計
ax2.set_title("TSMC (2330.TW) — 30-Day Future Price Forecast & Confidence Intervals (Fan Chart)",
             fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("Price (TWD)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=False)

# 在預測起點繪製一條垂直虛線區隔歷史與未來
ax2.axvline(x=prices.index[-1], color="#ef4444", linestyle="--", alpha=0.7, linewidth=1.2)
ax2.text(prices.index[-1], ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.05,
         "  Forecast Start", color="#ef4444", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("future_forecast.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Future forecast fan chart saved: future_forecast.png")

