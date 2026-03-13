from flask import Flask, render_template, request
import joblib
import os

os.environ["GROQ_API_KEY"] = ""

model = joblib.load("foodexp.pkl")

app = Flask(__name__)

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
    return (render_template("roe.html"))

@app.route("/generalQuestion", methods=['get','post'])
def generalQuestion():
    return (render_template("generalQuestion.html"))

if __name__ == "__main__":
    app.run()

    