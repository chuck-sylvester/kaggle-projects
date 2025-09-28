# === Improved Random Forest with simple feature engineering & CV ===
# This cell leaves your earlier work intact and creates a new submission file.
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- Safety checks: ensure train_df and test_df exist ---
assert "train_df" in globals(), "Expected 'train_df' to be defined earlier in the notebook."
assert "test_df" in globals(), "Expected 'test_df' to be defined earlier in the notebook."

# --- Minimal feature engineering ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Family size & IsAlone
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    # Name length (a light, often useful signal)
    out["NameLength"] = out["Name"].astype(str).str.len()
    return out

train_fe = add_features(train_df)
test_fe  = add_features(test_df)

# --- Select columns ---
target_col = "Survived"
numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch", "FamilySize", "IsAlone", "NameLength"]
categorical_features = ["Sex", "Embarked"]

X = train_fe[numeric_features + categorical_features]
y = train_fe[target_col]
X_test = test_fe[numeric_features + categorical_features]

# --- Preprocess ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))  # robust to outliers
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing Embarked/Sex if any
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --- Tuned RandomForest ---
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,            # let trees grow; RF handles variance with many trees
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[("prep", preprocess), ("rf", rf)])

# --- Cross-validation to sanity-check improvements ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"CV accuracy (mean ± std over 5 folds): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Fit on full training data and predict test ---
model.fit(X, y)
test_pred = model.predict(X_test).astype(int)

# --- Build submission ---
assert "PassengerId" in test_df.columns, "Expected 'PassengerId' in test_df for submission."

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_pred
})

out_csv = "submission_rf_tuned.csv"
submission.to_csv(out_csv, index=False)
print(f"Submission file written: {out_csv}")
