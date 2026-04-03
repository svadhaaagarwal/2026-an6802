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

os.environ["GROQ_API_KEY"] = "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"

model = joblib.load("foodexp.pkl")

client = Groq()
FINANCIAL_PERSONA = """You are a passionate financial enthusiast and advisor with deep knowledge of 
stock markets, risk management, predictive techniques like ARIMA and Monte Carlo, 
and Singapore's financial regulations (MAS guidelines). You explain concepts clearly 
with real-world examples, and you always relate answers back to practical investing 
and financial decision-making. Keep responses concise but insightful."""

app = Flask(__name__)

# Chart helper function - Food expense
def chart_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#FFFFFF', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

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

ethics_questions = [
    {
        "id": 1,
        "question": "A manager can disclose confidential employee information to unauthorized personnel.",
        "answer": False,
        "explanation": "Confidential employee data is protected under data privacy laws such as Singapore's PDPA. Sharing it without authorization breaches trust and can result in legal consequences."
    },
    {
        "id": 2,
        "question": "Accepting gifts from a client in a competitive bidding process is ethically permissible.",
        "answer": False,
        "explanation": "Accepting gifts during a bidding process creates a conflict of interest and undermines fair competition. Most corporate codes of conduct and anti-corruption policies strictly prohibit this."
    },
    {
        "id": 3,
        "question": "It is ethical for a software developer to copy code from an open-source project without attribution if it saves time on a tight deadline.",
        "answer": False,
        "explanation": "Open-source licenses require proper attribution. Failing to credit the original author violates intellectual property rights and the terms of the license, regardless of time pressure."
    },
    {
        "id": 4,
        "question": "A doctor can ethically prioritize treatment based on a patient's ability to pay.",
        "answer": False,
        "explanation": "Medical ethics require treatment decisions to be based on clinical need, not financial status. Prioritizing by ability to pay violates the principle of justice and equal access to healthcare."
    },
    {
        "id": 5,
        "question": "It is ethical for a teacher to grade students differently based on personal liking.",
        "answer": False,
        "explanation": "Grading must be based on objective academic criteria. Personal bias in assessment violates principles of fairness and equity, and can constitute discrimination."
    }
]

@app.route("/ethics", methods=['get','post'])
@app.route("/ethics/<int:qnum>/<int:score>", methods=['get','post'])
def ethics(qnum=1, score=0):
    if qnum < 1 or qnum > len(ethics_questions):
        qnum = 1
    q = ethics_questions[qnum - 1]
    return render_template("ethics.html", q=q, total=len(ethics_questions), score=score)

@app.route("/ethics_answer", methods=['post'])
def ethics_answer():
    qnum = int(request.form.get("qnum"))
    score = int(request.form.get("score"))
    user_answer = request.form.get("answer") == "True"
    q = ethics_questions[qnum - 1]
    is_correct = (user_answer == q["answer"])
    if is_correct:
        score += 1
    next_qnum = qnum + 1 if qnum < len(ethics_questions) else None
    return render_template("ethics_result.html",
                           is_correct=is_correct,
                           q=q,
                           qnum=qnum,
                           next_qnum=next_qnum,
                           total=len(ethics_questions),
                           score=score)

@app.route("/econ", methods=['get','post'])
def econ():
    return (render_template("econ.html"))

@app.route("/foodExp", methods=['get','post'])
def foodExp():
    q = float(request.form.get("q"))
    r = model.predict([[q]])[0][0]

    # Generate salary vs food expense curve
    salaries = np.linspace(50, 2000, 100)
    predictions = [model.predict([[s]])[0][0] for s in salaries]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(salaries, predictions, color='#0F1A2E', linewidth=2, label='Predicted Trend')
    ax.scatter([q], [r], color='#B8924A', s=120, zorder=5, label=f'You (${q:.0f} → ${r:.2f})')
    ax.set_xlabel('Weekly Salary (SGD)', fontsize=12)
    ax.set_ylabel('Predicted Food Expense (SGD)', fontsize=12)
    ax.set_title('Salary vs Food Expenditure: Model Prediction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    chart = chart_to_base64(fig)

    return render_template("foodExp.html", r=r, q=q, chart=chart)

@app.route("/chatbot", methods=['get','post'])
def chatbot():
    return (render_template("chatbot.html"))

@app.route("/roe", methods=['get','post'])
def roe():
    r = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [
        {"role": "system", "content": FINANCIAL_PERSONA},
        {"role": "user", "content": "Explain Return on Equity (RoE) — what it is, how it's calculated, and why investors care about it. Keep it under 80 words."}
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
        {"role": "system", "content": FINANCIAL_PERSONA},
        {"role": "user", "content": q}
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

    