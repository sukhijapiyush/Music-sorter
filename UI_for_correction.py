from flask import Flask, render_template, request, redirect, jsonify
import pandas as pd

app = Flask(__name__)

# Load CSV data
df = pd.read_csv("language_identification_results.csv")

@app.route("/")
def index():
    return render_template("index.html", rows=df.to_dict(orient="records"))

@app.route("/edit/<int:row_id>", methods=["POST"])
def edit(row_id):
    # Update the row with form data
    for key in request.form:
        df.at[row_id, key] = request.form[key]
    df.to_csv("data.csv", index=False)
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)
