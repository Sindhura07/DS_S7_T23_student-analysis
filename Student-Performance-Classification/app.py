

"""
Student Performance Classification - web app
Run:  streamlit run app.py
Upload a CSV in the sidebar, train the 5 models, explore results, and predict for a new student.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from train_model import make_synthetic
 
warnings.filterwarnings("ignore")
RNG = 42
st.set_page_config(page_title="Student Performance Classifier", page_icon="🎓", layout="wide")
 
 
# ----------------------------------------------------------------- helpers
def prepare(df, target, drop_cols, cut1, cut2):
    df = df.drop(columns=drop_cols, errors="ignore").drop_duplicates().copy()
    # numeric target with many distinct values -> Low / Medium / High
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 6:
        df[target] = pd.cut(df[target], [-np.inf, cut1, cut2, np.inf],
                            labels=["Low", "Medium", "High"]).astype(str)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df[target] = df[target].astype(str)
 
    feats = df.drop(columns=[target])
    num = feats.select_dtypes(include="number").columns.tolist()
    cat = feats.select_dtypes(exclude="number").columns.tolist()
    for c in num:
        df[c] = df[c].fillna(df[c].median())
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    for c in cat:
        df[c] = df[c].fillna(df[c].mode()[0])
    return df, num, cat
 
 
def train_all(df, target, num, cat, k_features):
    X = pd.get_dummies(df.drop(columns=[target]), columns=cat, drop_first=True).astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
 
    k = min(k_features, X.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    feats = X.columns[sel.get_support()].tolist()
    X_tr, X_te = X_tr[feats].copy(), X_te[feats].copy()
 
    scale_cols = [c for c in feats if c in num]
    scaler = StandardScaler().fit(X_tr[scale_cols])
    X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
    X_te[scale_cols] = scaler.transform(X_te[scale_cols])
 
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", C=2, probability=True, random_state=RNG),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RNG),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                             early_stopping=True, random_state=RNG),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    rows, cms, fitted = [], {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        fitted[name] = m
        cms[name] = confusion_matrix(y_te, pred)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_te, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_te, pred, average="weighted", zero_division=0),
            "CV Accuracy": cross_val_score(m, X_tr, y_tr, cv=cv).mean(),
        })
    res = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=feats).sort_values()
    return dict(models=fitted, results=res, cms=cms, le=le, scaler=scaler, scale_cols=scale_cols,
                feats=feats, all_cols=X.columns.tolist(), num=num, cat=cat, importance=imp,
                best=res.loc[0, "Model"], n_train=len(X_tr), n_test=len(X_te))
 
 
# ----------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Setup")
upload = st.sidebar.file_uploader("Upload student dataset (CSV)", type=["csv"])
if upload is not None:
    raw = pd.read_csv(upload, sep=None, engine="python")
    source = upload.name
else:
    raw = make_synthetic()
    source = "synthetic sample data"
st.sidebar.caption(f"Using: **{source}** ({raw.shape[0]} rows, {raw.shape[1]} columns)")
 
default_t = raw.columns.get_loc("Performance") if "Performance" in raw.columns else \
    (raw.columns.get_loc("G3") if "G3" in raw.columns else len(raw.columns) - 1)
target = st.sidebar.selectbox("Target column (what to predict)", raw.columns, index=default_t)
drop_default = [c for c in ["G1", "G2"] if c in raw.columns and c != target]
drop_cols = st.sidebar.multiselect("Columns to exclude", [c for c in raw.columns if c != target],
                                   default=drop_default,
                                   help="Exclude ID columns or earlier grades that leak the answer.")
cut1 = cut2 = 0
if pd.api.types.is_numeric_dtype(raw[target]) and raw[target].nunique() > 6:
    mx = float(raw[target].max())
    st.sidebar.write("Numeric target → split into Low / Medium / High")
    cut1 = st.sidebar.number_input("Low if ≤", value=round(mx * 0.5, 1))
    cut2 = st.sidebar.number_input("Medium if ≤ (else High)", value=round(mx * 0.75, 1))
k_features = st.sidebar.slider("Features to keep", 3, 30, 10)
 
if st.sidebar.button("🚀 Train models", type="primary", width="stretch"):
    with st.spinner("Preprocessing and training 5 models..."):
        clean, num, cat = prepare(raw, target, drop_cols, cut1, cut2)
        st.session_state["clean"] = clean
        st.session_state["target"] = target
        st.session_state["art"] = train_all(clean, target, num, cat, k_features)
 
# ------------------------------------------------------------------- main
st.title("Student Performance Classification")
st.caption("End-to-end ML system: preprocessing → EDA → 5 classifiers → evaluation → prediction")
 
tab_data, tab_eda, tab_res, tab_pred = st.tabs(
    ["📂 Dataset", "📊 EDA", "🏆 Model comparison", "🔮 Predict a student"])
 
with tab_data:
    st.subheader("Raw data preview")
    st.dataframe(raw.head(50), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw.shape[0])
    c2.metric("Missing values", int(raw.isna().sum().sum()))
    c3.metric("Duplicate rows", int(raw.duplicated().sum()))
    st.write("Column types")
    st.dataframe(raw.dtypes.astype(str).rename("type").to_frame().T, width="stretch")
    if "art" not in st.session_state:
        st.info("👈 Pick the target column and click **Train models** in the sidebar.")
 
if "art" not in st.session_state:
    for t in (tab_eda, tab_res, tab_pred):
        with t:
            st.info("Train the models first (sidebar).")
    st.stop()
 
art, clean, tgt = st.session_state["art"], st.session_state["clean"], st.session_state["target"]
 
with tab_eda:
    order = [c for c in ["Low", "Medium", "High"] if c in clean[tgt].unique()] or None
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x=tgt, data=clean, order=order, ax=ax)
        ax.set_title("Class distribution"); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if len(art["num"]) > 1:
            sns.heatmap(clean[art["num"]].corr(), cmap="coolwarm", annot=len(art["num"]) <= 12,
                        fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap"); st.pyplot(fig)
    col = st.selectbox("Explore a feature against performance", art["num"] + art["cat"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if col in art["num"]:
        sns.boxplot(x=tgt, y=col, data=clean, order=order, ax=ax)
    else:
        sns.countplot(x=col, hue=tgt, data=clean, hue_order=order, ax=ax)
        plt.xticks(rotation=30)
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    art["importance"].plot.barh(ax=ax, title="Feature importance (Random Forest)")
    st.pyplot(fig)
 
with tab_res:
    st.success(f"Best model by F1-score: **{art['best']}**   "
               f"(train {art['n_train']} / test {art['n_test']} students)")
    st.dataframe(art["results"].round(4).style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-score", "CV Accuracy"], color="#c6f6d5"),
        width="stretch")
    fig, ax = plt.subplots(figsize=(9, 4))
    art["results"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot.bar(
        ax=ax, rot=0)
    ax.set_ylim(0, 1.05); st.pyplot(fig)
    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for i, (name, cm) in enumerate(art["cms"].items()):
        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=art["le"].classes_, yticklabels=art["le"].classes_, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        cols[i % 3].pyplot(fig)
 
with tab_pred:
    st.subheader("Enter student details")
    model_name = st.selectbox("Model", art["results"]["Model"].tolist(), index=0)
    feats_df = clean.drop(columns=[tgt])
    with st.form("student_form"):
        cols = st.columns(3)
        inp = {}
        for i, c in enumerate(feats_df.columns):
            with cols[i % 3]:
                if c in art["num"]:
                    lo, hi, med = float(feats_df[c].min()), float(feats_df[c].max()), float(feats_df[c].median())
                    if pd.api.types.is_integer_dtype(raw[c]) if c in raw.columns else False:
                        inp[c] = st.slider(c, int(lo), int(hi), int(med))
                    else:
                        inp[c] = st.slider(c, lo, hi, med)
                else:
                    inp[c] = st.selectbox(c, sorted(feats_df[c].unique().tolist()))
        go = st.form_submit_button("Predict performance", type="primary")
    if go:
        row = pd.get_dummies(pd.DataFrame([inp]), columns=art["cat"])
        row = row.reindex(columns=art["all_cols"], fill_value=0)[art["feats"]].astype(float)
        row[art["scale_cols"]] = art["scaler"].transform(row[art["scale_cols"]])
        m = art["models"][model_name]
        pred = art["le"].inverse_transform(m.predict(row))[0]
        icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(pred, "🎯")
        st.markdown(f"## {icon} Predicted performance: **{pred}**")
        if hasattr(m, "predict_proba"):
            proba = pd.Series(m.predict_proba(row)[0], index=art["le"].classes_, name="probability")
            st.bar_chart(proba)
 
