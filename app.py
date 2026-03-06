from flask import Flask, render_template, request
import joblib

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

if __name__ == "__main__":
    app.run()

    