# src/train.py
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


DATA_PATH = pathlib.Path("data/tickets.csv")
OUT_DIR = pathlib.Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "classifier.joblib"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python download_data.py"
        )
    df = pd.read_csv(DATA_PATH)
    # expect columns: text, category
    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    return df


def main():
    df = load_data()

    n_samples = len(df)
    n_classes = df["category"].nunique()

    # Choose split robustly for tiny datasets
    if n_samples < 50:
        test_size = 0.5
        # Can stratify only if every class has >=2 examples AND test set will have >=1 per class
        min_per_class = df["category"].value_counts().min()
        can_stratify = (min_per_class >= 2) and (int(n_samples * test_size) >= n_classes)
        stratify_arg = df["category"] if can_stratify else None
    else:
        test_size = 0.2
        stratify_arg = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"],
        test_size=test_size,
        random_state=42,
        stratify=stratify_arg
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=300))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
