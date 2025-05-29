from flask import Flask
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

app = Flask(__name__)

def load_data():
    data = load_iris(as_frame=True)
    return data.frame

def train_and_evaluate(df):
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    return accuracy, report

@app.route("/")
def run_pipeline():
    df = load_data()
    accuracy, report = train_and_evaluate(df)
    return f"""
    <h2>✅ Iris Classifier Pipeline</h2>
    <p><strong>Accuracy:</strong> {accuracy:.4f}</p>
    <pre>{pd.DataFrame(report).round(2).to_html()}</pre>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
