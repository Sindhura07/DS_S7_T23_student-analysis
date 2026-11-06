
"""
Student Performance Classification - end-to-end ML pipeline
Run:  python train_model.py                (uses student_data.csv if present, else synthetic data)
      python train_model.py --data my.csv --target Performance
 
Expected CSV: one row per student, a categorical target column
(e.g. Performance = Low / Medium / High). If your dataset has a numeric final
grade (e.g. UCI 'G3'), pass --grade-col G3 and it is binned into 3 classes.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
 
warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RNG = 42
 
 
# ---------------------------------------------------------------- 1. DATA
def make_synthetic(n=1000, seed=RNG):
    """Synthetic dataset with academic, demographic and behavioural attributes."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Gender": r.choice(["Male", "Female"], n),
        "Age": r.integers(15, 22, n),
        "ParentEducation": r.choice(["HighSchool", "Bachelors", "Masters", "PhD"], n, p=[.35, .4, .2, .05]),
        "StudyHours": np.clip(r.normal(3.5, 1.6, n), 0, 10).round(1),
        "Attendance": np.clip(r.normal(82, 12, n), 35, 100).round(1),
        "PreviousGrade": np.clip(r.normal(65, 14, n), 15, 100).round(1),
        "Absences": np.clip(r.poisson(5, n), 0, 40),
        "InternetAccess": r.choice(["Yes", "No"], n, p=[.85, .15]),
        "ExtraCurricular": r.choice(["Yes", "No"], n),
        "SleepHours": np.clip(r.normal(7, 1.2, n), 3, 10).round(1),
        "FailuresBefore": r.choice([0, 1, 2, 3], n, p=[.7, .18, .08, .04]),
    })
    score = (0.45 * df.PreviousGrade + 4.0 * df.StudyHours + 0.25 * df.Attendance
             - 0.6 * df.Absences - 5 * df.FailuresBefore + 2 * (df.InternetAccess == "Yes")
             + r.normal(0, 6, n))
    q = np.quantile(score, [.33, .66])
    df["Performance"] = np.select([score < q[0], score < q[1]], ["Low", "Medium"], "High")
    # inject realistic mess for the preprocessing stage to clean
    for c in ["StudyHours", "Attendance", "ParentEducation"]:
        df.loc[r.choice(n, 25, replace=False), c] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # duplicates
    return df
 
 
def load_data(path, target, grade_col):
    if path and os.path.exists(path):
        df = pd.read_csv(path, sep=None, engine="python")
        print(f"Loaded {path}: {df.shape}")
    else:
        df = make_synthetic()
        print("No dataset found -> using synthetic data:", df.shape)
    if grade_col and grade_col in df.columns:       # numeric grade -> classes
        df[target] = pd.cut(df[grade_col], [-1, 9, 14, 100], labels=["Low", "Medium", "High"])
        df = df.drop(columns=[grade_col])
    return df
 
 
# ------------------------------------------------------- 2. PREPROCESSING
def preprocess(df, target):
    print("\n--- Data validation ---")
    print(df.dtypes.to_string())
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("Duplicates:", df.duplicated().sum())
 
    df = df.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)
 
    num = df.drop(columns=[target]).select_dtypes(include="number").columns.tolist()
    cat = df.drop(columns=[target]).select_dtypes(exclude="number").columns.tolist()
    for c in num:
        df[c] = df[c].fillna(df[c].median())
    for c in cat:
        df[c] = df[c].fillna(df[c].mode()[0])
 
    # outliers: IQR capping (winsorising keeps all rows)
    for c in num:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
 
    return df, num, cat
 
 
def eda(df, num, target):
    order = ["Low", "Medium", "High"] if set(df[target]) == {"Low", "Medium", "High"} else None
    plt.figure(figsize=(5, 4)); sns.countplot(x=target, data=df, order=order)
    plt.title("Class distribution"); plt.tight_layout(); plt.savefig(f"{OUT}/eda_class_distribution.png", dpi=120); plt.close()
 
    corr = df[num].corr()
    plt.figure(figsize=(9, 7)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig(f"{OUT}/eda_correlation.png", dpi=120); plt.close()
 
    df[num].hist(figsize=(12, 8), bins=20); plt.tight_layout()
    plt.savefig(f"{OUT}/eda_histograms.png", dpi=120); plt.close()
 
    top = num[:3]
    fig, ax = plt.subplots(1, len(top), figsize=(5 * len(top), 4))
    for a, c in zip(np.atleast_1d(ax), top):
        sns.boxplot(x=target, y=c, data=df, order=order, ax=a)
    plt.tight_layout(); plt.savefig(f"{OUT}/eda_boxplots.png", dpi=120); plt.close()
 
 
# ------------------------------------------------------------- 3. MODELS
def main(args):
    df = load_data(args.data, args.target, args.grade_col)
    df, num, cat = preprocess(df, args.target)
    eda(df, num, args.target)
 
    # encode categoricals (one-hot) and target (label)
    X = pd.get_dummies(df.drop(columns=[args.target]), columns=cat, drop_first=True)
    le = LabelEncoder(); y = le.fit_transform(df[args.target])
    print("\nClasses:", dict(zip(le.classes_, range(len(le.classes_)))))
 
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
 
    # feature selection (ANOVA F-test) fitted on train only
    k = min(args.k_features, X.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    feats = X.columns[selector.get_support()].tolist()
    print(f"Selected {k} features:", feats)
    X_tr, X_te = X_tr[feats], X_te[feats]
 
    # normalisation of numeric columns (fit on train only)
    scale_cols = [c for c in feats if c in num]
    scaler = StandardScaler().fit(X_tr[scale_cols])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
    X_te[scale_cols] = scaler.transform(X_te[scale_cols])
 
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", C=2, probability=True, random_state=RNG),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RNG),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=RNG),
    }
 
    rows, fitted = [], {}
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    for name, m in models.items():
        m.fit(X_tr, y_tr); pred = m.predict(X_te); fitted[name] = m
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average="weighted"),
            "Recall": recall_score(y_te, pred, average="weighted"),
            "F1-score": f1_score(y_te, pred, average="weighted"),
            "CV Accuracy (5-fold)": cross_val_score(m, X_tr, y_tr, cv=cv).mean(),
        })
        cm = confusion_matrix(y_te, pred)
        plt.figure(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                                  xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"{name} - confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout(); plt.savefig(f"{OUT}/cm_{name.replace(' ', '_').lower()}.png", dpi=120); plt.close()
 
    res = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    res.round(4).to_csv(f"{OUT}/model_comparison.csv", index=False)
    print("\n=== Model comparison ===\n", res.round(4).to_string(index=False))
 
    best = res.loc[0, "Model"]
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, fitted[best].predict(X_te), target_names=le.classes_))
 
    ax = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot.bar(figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison"); plt.tight_layout()
    plt.savefig(f"{OUT}/model_comparison.png", dpi=120); plt.close()
 
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    imp.plot.barh(figsize=(7, 5), title="Feature importance (Random Forest)"); plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=120); plt.close()
 
    # save everything the dashboard needs
    joblib.dump({
        "model": fitted[best], "best_name": best, "scaler": scaler, "scale_cols": scale_cols,
        "features": feats, "all_columns": X.columns.tolist(), "label_encoder": le,
        "num_cols": num, "cat_cols": cat,
        "cat_options": {c: sorted(df[c].unique().tolist()) for c in cat},
        "num_ranges": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in num},
        "results": res,
    }, "student_model.joblib")
    print("Saved student_model.joblib and plots in ./outputs")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="student_data.csv")
    p.add_argument("--target", default="Performance")
    p.add_argument("--grade-col", default=None, help="numeric grade column to bin into Low/Medium/High")
    p.add_argument("--k-features", type=int, default=10)
    main(p.parse_args())
 
