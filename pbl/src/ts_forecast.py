# src/ts_forecast.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_series(series, order=(1,1,1), steps=5):
    # series: pd.Series indexed by Year
    model = ARIMA(series.astype(float), order=order)
    res = model.fit()
    pred = res.get_forecast(steps=steps)
    df_pred = pred.summary_frame(alpha=0.05)  # mean, mean_ci_lower, mean_ci_upper
    return df_pred, res
