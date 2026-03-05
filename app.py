from flask import Flask, render_template, request

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

if __name__ == "__main__":
    app.run()

    