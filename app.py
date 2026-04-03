from flask import Flask, render_template, request
import joblib
import os
import io
import base64
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq

os.environ["GROQ_API_KEY"] = ""

model = joblib.load("foodexp.pkl")

client = Groq()
app = Flask(__name__)

# Fetch DBS stock data
def get_dbs_data(period="1y"):
    ticker = yf.Ticker("D05.SI")
    df = ticker.history(period=period)
    return df

# Convert matplotlib chart to base64 
def chart_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#FFFFFF', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@app.route("/", methods=['get','post'])
def index():
    return (render_template("index.html"))

@app.route("/main", methods=['get','post'])
def main():
    return (render_template("main.html"))

@app.route("/ethics", methods=['get','post'])
def ethics():
    return (render_template("ethics.html"))

@app.route("/correct", methods=['get','post'])
def correct():
    return (render_template("correct.html"))

@app.route("/wrong", methods=['get','post'])
def wrong():
    return (render_template("wrong.html"))

@app.route("/econ", methods=['get','post'])
def econ():
    return (render_template("econ.html"))

@app.route("/foodExp", methods=['get','post'])
def foodExp():
    q = float(request.form.get("q"))
    r = model.predict([[q]])
    return (render_template("foodExp.html", r=r[0][0]))

@app.route("/chatbot", methods=['get','post'])
def chatbot():
    return (render_template("chatbot.html"))

@app.route("/roe", methods=['get','post'])
def roe():
    r = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [
        {"role": "system", "content": "Please explain RoE in 20 words."}
    ])                         
    return (render_template("roe.html", r = r.choices[0].message.content))

@app.route("/generalQuestion", methods=['get','post'])
def generalQuestion():
    return (render_template("generalQuestion.html"))

@app.route("/groqReply", methods=['get','post'])
def groqReply():
    q = request.form.get("q")
    r = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [
        {"role": "system", "content": q}
    ])  
    return (render_template("groqReply.html", r = r.choices[0].message.content))

@app.route("/equity", methods=['get','post'])
def equity():
    return (render_template("equity.html"))

@app.route("/apple", methods=['get','post'])
def apple():
    return (render_template("apple.html"))

# DBS Analysis Entry Page
@app.route("/dbs", methods=['get','post'])
def dbs():
    return (render_template("dbs.html"))

# DBS ARIMA Prediction
@app.route("/dbs_predict", methods=['get','post'])
def dbs_predict():
    try:
        df = get_dbs_data("1y")
        close = df['Close']

        # Fit ARIMA model (p=5, d=1, q=0)
        model_arima = ARIMA(close, order=(5, 1, 0))
        model_fit = model_arima.fit()

        # Forecast next 30 days
        forecast = model_fit.forecast(steps=30)
        last_date = close.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=30, freq='B')

        # Generate chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(close.index[-90:], close.values[-90:],
                color='#0F1A2E', linewidth=2, label='Actual Price')
        ax.plot(forecast_dates, forecast.values,
                color='#B8924A', linewidth=2, linestyle='--', label='ARIMA Forecast')
        ax.fill_between(forecast_dates, forecast.values * 0.97, forecast.values * 1.03,
                        color='#B8924A', alpha=0.1)
        ax.set_title('DBS Bank (D05.SI) — ARIMA(5,1,0) Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (SGD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        chart = chart_to_base64(fig)

        # Key stats
        current_price = round(close.values[-1], 2)
        predicted_price = round(forecast.values[-1], 2)
        change_pct = round(((predicted_price - current_price) / current_price) * 100, 2)

        return render_template("dbs_predict.html",
                                chart=chart,
                                current=current_price,
                                predicted=predicted_price,
                                change=change_pct)
    except Exception as e:
        return render_template("dbs.html", error=str(e))

# DBS Risk Analysis (VaR + Sharpe) 
@app.route("/dbs_risk", methods=['get','post'])
def dbs_risk():
    try:
        df = get_dbs_data("1y")
        close = df['Close']

        # Daily returns
        returns = close.pct_change().dropna()

        # Value at Risk (95% confidence, 1-day)
        var_95 = round(np.percentile(returns, 5) * 100, 2)

        # Sharpe Ratio (annualized, risk-free rate = 3.5% for Singapore T-bills)
        risk_free_rate = 0.035
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)
        sharpe = round((annual_return - risk_free_rate) / annual_std, 2)

        # Generate returns distribution chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns * 100, bins=50, color='#0F1A2E', alpha=0.7, edgecolor='white')
        ax.axvline(x=var_95, color='#DC2626', linewidth=2, linestyle='--',
                    label=f'VaR 95%: {var_95}%')
        ax.set_title('DBS Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        chart = chart_to_base64(fig)

        # Extra context
        annual_return_pct = round(annual_return * 100, 2)
        annual_std_pct = round(annual_std * 100, 2)

        return render_template("dbs_risk.html",
                                chart=chart,
                                var_95=var_95,
                                sharpe=sharpe,
                                annual_return=annual_return_pct,
                                annual_std=annual_std_pct)
    except Exception as e:
        return render_template("dbs.html", error=str(e))

if __name__ == "__main__":
    app.run()

    