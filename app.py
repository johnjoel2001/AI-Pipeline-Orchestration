from flask import Flask
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import time

app = Flask(__name__)

def load_data():
    data = load_iris(as_frame=True)
    return data.frame, data.target_names

def train_and_evaluate(df):
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
    samples = pd.DataFrame({"True Label": y_test.values, "Predicted": predictions}).head(10)
    model_params = model.get_params()

    return accuracy, report, cm, samples, model_params

@app.route("/")
def run_pipeline():
    start_time = time.time()

    df, target_names = load_data()
    accuracy, report, cm, samples, model_params = train_and_evaluate(df)

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    report_df = pd.DataFrame(report).round(2)
    param_df = pd.DataFrame(model_params.items(), columns=["Parameter", "Value"])

    elapsed = round(time.time() - start_time, 2)

    html = f"""
    <html>
    <head><title>Iris Classifier</title></head>
    <body style="font-family:sans-serif; padding:20px">
    <h2 style="color:green;"> Iris Classifier Pipeline</h2>
    <p><strong> Time Taken:</strong> {elapsed} seconds</p>
    <p><strong> Accuracy:</strong> {accuracy:.4f}</p>

    <h3>Classification Report</h3>
    {report_df.to_html()}

    <h3>Confusion Matrix</h3>
    {cm_df.to_html()}

    <h3>Sample Predictions</h3>
    {samples.to_html(index=False)}

    <h3>Model Parameters</h3>
    {param_df.to_html(index=False)}

    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
