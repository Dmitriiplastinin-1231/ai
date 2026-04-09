import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


TRAIN_PATH = "prices_train.csv"
TEST_PATH = "prices_test.csv"
OUTPUT_PATH = "submission.csv"
TARGET = "Y house price of unit area"
N_ESTIMATORS = 700
MIN_SAMPLES_LEAF = 1
GB_N_ESTIMATORS = 500
GB_LEARNING_RATE = 0.03
GB_MAX_DEPTH = 3


def load_features(path: str, target: str | None = None):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        index_values = df["Unnamed: 0"]
        df = df.drop(columns=["Unnamed: 0"])
    else:
        index_values = pd.Series(np.arange(len(df)))

    if target is None:
        return df, index_values

    y = df[target]
    x = df.drop(columns=[target])
    return x, y


def build_model_candidates():
    common = [("imputer", SimpleImputer(strategy="median"))]
    return {
        "extra_trees": Pipeline(
            common
            + [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            common
            + [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            common
            + [
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=GB_N_ESTIMATORS,
                        learning_rate=GB_LEARNING_RATE,
                        max_depth=GB_MAX_DEPTH,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def main():
    x_train, y_train = load_features(TRAIN_PATH, TARGET)
    x_test, test_index = load_features(TEST_PATH)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    best_name = None
    best_rmse = float("inf")
    best_model = None

    for name, model in build_model_candidates().items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
        rmse = -scores.mean()
        print(f"{name}: CV RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    print(f"Selected model: {best_name} (RMSE={best_rmse:.4f})")
    best_model.fit(x_train, y_train)
    preds = best_model.predict(x_test)

    out = pd.DataFrame({"index": test_index.astype(int), TARGET: preds})
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
