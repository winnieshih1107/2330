import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

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
# 5. 明日股價預測
# ==========================================
latest_data_sk = prices.iloc[-lag:].values[::-1]
next_day_sk    = sk_model.predict([latest_data_sk])[0]

next_day_arima = ARIMA(prices, order=(20, 1, 0)).fit().forecast(steps=1).iloc[0]

print("\n=== 明日收盤價預測對比 ===")
print(f"最新實際收盤價 ({prices.index[-1].strftime('%Y-%m-%d')}): {prices.iloc[-1]:.2f}")
print(f"sklearn 預測明日價格: {next_day_sk:.2f}")
print(f"ARIMA  預測明日價格: {next_day_arima:.2f}")

# ==========================================
# 6. 視覺化
# ==========================================
fig, ax = plt.subplots(figsize=(12, 5))

window = prices.iloc[-(len(y_test) + 40):]
ax.plot(window.index, window.values, label="Actual Price", color="#1e293b", linewidth=1.5)
ax.plot(y_test.index, y_pred_sk,    label="sklearn (Linear AR)", color="#3b82f6", linestyle="--", linewidth=1.5)
ax.plot(y_test.index, y_pred_arima, label="ARIMA(20,1,0)",       color="#10b981", linestyle=":",  linewidth=2)

ax.set_title("TSMC (2330.TW) — 1-Step-Ahead Prediction: Linear AR vs ARIMA(20,1,0)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Price (TWD)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: comparison.png")
