"""Train the churn model and save it with pickle (ML Zoomcamp 5.1 / 5.2)."""
import os
import pickle
import urllib.request

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/"
    "master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DATA_PATH = "data/telco-churn.csv"
OUTPUT_FILE = "model_C=1.0.bin"

C = 1.0
N_SPLITS = 5

numerical = ["tenure", "monthlycharges", "totalcharges"]
categorical = [
    "gender", "seniorcitizen", "partner", "dependents", "phoneservice",
    "multiplelines", "internetservice", "onlinesecurity", "onlinebackup",
    "deviceprotection", "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]


def load_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # works with old (object) and new (str) pandas string dtypes
    string_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
    df["churn"] = (df["churn"] == "yes").astype(int)
    return df


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    df = load_data()
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    # Cross-validation (5.2)
    print(f"Doing validation with C={C}")
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        dv, model = train(df_tr, df_tr.churn.values, C=C)
        y_pred = predict(df_val, dv, model)
        scores.append(roc_auc_score(df_val.churn.values, y_pred))
    print("C=%s AUC %.3f +- %.3f" % (C, sum(scores) / len(scores), pd.Series(scores).std()))

    # Final model
    print("Training the final model")
    dv, model = train(df_full_train, df_full_train.churn.values, C=C)
    y_pred = predict(df_test, dv, model)
    print(f"Test AUC = {roc_auc_score(df_test.churn.values, y_pred):.3f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {OUTPUT_FILE}")
