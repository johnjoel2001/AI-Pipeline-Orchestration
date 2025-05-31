from flask import Flask
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tenacity import retry, stop_after_attempt, wait_fixed
import pandas as pd
import logging

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retry loading in case of transient errors
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def load_data():
    logger.info(" Loading Iris dataset...")
    data = load_iris(as_frame=True)
    logger.info("Dataset loaded successfully")
    return data.frame

def preprocess_data(df):
    logger.info(" Preprocessing data...")
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    logger.info(" Training RandomForest model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logger.info(" Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info(" Evaluating model performance...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    logger.info(f" Accuracy: {accuracy:.4f}")
    return accuracy, report

@app.route("/")
def run_pipeline():
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)

        return f"""
        <h2> Iris Classification Pipeline</h2>
        <p><strong>Accuracy:</strong> {accuracy:.4f}</p>
        <h3>Classification Report</h3>
        <pre>{pd.DataFrame(report).round(2).to_html()}</pre>
        """

    except Exception as e:
        logger.error(" An error occurred during pipeline execution", exc_info=True)
        return f"<h2>Pipeline Error</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
