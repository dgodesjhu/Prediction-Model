import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import random
from google.oauth2.service_account import Credentials
import gspread
from datetime import datetime

# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ── Page config & style ───────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Prediction Model — AIM26", layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background-color: #f4faff;
    color: #003366;
    font-family: 'Segoe UI', sans-serif;
}
.block-container > h1, h2, h3, h4 {
    background-color: #003366;
    color: white !important;
    padding: 0.5em 1em;
    border-radius: 6px;
}
button[kind="primary"] {
    background-color: #0073e6 !important;
    color: white !important;
    border: none;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Google Sheets ─────────────────────────────────────────────────────────────

SHEET_ID = "1z96SWNSEyTmb7wR8up1ePFveP44qZCM0JCedRHda5mk"
SCOPES   = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource
def get_sheet():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        # Fix newlines in private key in case they got mangled
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds  = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(SHEET_ID).sheet1

        # Write headers if sheet is empty
        if not sheet.get_all_values():
            sheet.update("A1", [[
                "JHU_ID", "Student_Name", "Timestamp", "Model_Type",
                "Val_AUC", "Holdout_AUC",
                "Val_F1", "Holdout_F1",
                "Train_Count", "Hyperparameters"
            ]])
        return sheet
    except Exception as e:
        st.error(f"Google Sheets connection failed: {e}")
        return None

def get_existing_submissions(sheet):
    """Return dict of {jhu_id: row_data} for all submitted IDs."""
    try:
        records = sheet.get_all_records()
        return {str(r["JHU_ID"]).strip().upper(): r for r in records}
    except:
        return {}

def is_competition_active():
    """Check config sheet for active flag."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds  = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        config = client.open_by_key(SHEET_ID).worksheet("config")
        val    = config.cell(1, 2).value
        return str(val).strip().lower() == "true"
    except:
        return False

def write_submission(sheet, row):
    try:
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Failed to write to leaderboard: {e}")
        return False

# ── Valid IDs ─────────────────────────────────────────────────────────────────

@st.cache_data
def get_valid_ids():
    try:
        ids = st.secrets["VALID_IDS"]
        return set(i.strip().upper() for i in ids.split(","))
    except:
        return set()

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except:
        return None

# ── ANN ───────────────────────────────────────────────────────────────────────

class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, nodes, activation):
        super().__init__()
        act_fn = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[activation]
        layers = [nn.Linear(input_dim, nodes), act_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(nodes, nodes), act_fn]
        layers.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Prediction Model Explorer — AIM26")

# ── Competition gate ───────────────────────────────────────────────────────────

if not is_competition_active():
    st.info("🔒 The competition is not currently active. Please wait for your instructor to open it.")
    st.stop()

# ── Step 1: ID entry ──────────────────────────────────────────────────────────

st.subheader("Step 1: Enter your JHU ID")
st.write("You will use this ID to submit your final model for holdout scoring. Enter it now — you cannot change it later.")

jhu_id_input = st.text_input("JHU ID (6 characters)", max_chars=6).strip().upper()

valid_ids = get_valid_ids()

if jhu_id_input:
    if jhu_id_input not in valid_ids:
        st.error("❌ ID not recognised. Please check your JHU ID or see your instructor.")
        st.stop()
    else:
        st.success(f"✅ Welcome! ID {jhu_id_input} confirmed.")
        st.session_state["confirmed_id"] = jhu_id_input
else:
    st.info("Enter your JHU ID above to begin.")
    st.stop()

# ── Step 2: Dataset (fixed to Customer Retention/Churn for in-class) ──────────

st.subheader("Step 2: Dataset")
st.write("This session uses the **Customer Retention / Churn** dataset.")

BASE_URL    = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/retentionchurn"
HOLDOUT_URL = f"{BASE_URL}/test.csv"

train_df   = load_data(f"{BASE_URL}/train.csv")
valid_df   = load_data(f"{BASE_URL}/validation.csv")
holdout_df = load_data(HOLDOUT_URL)

if train_df is None or valid_df is None:
    st.error("Failed to load dataset. Check your internet connection.")
    st.stop()

if holdout_df is None:
    st.error("Failed to load holdout dataset. Contact your instructor.")
    st.stop()

n_features = train_df.shape[1] - 1

with st.expander("Preview training data", expanded=False):
    st.dataframe(train_df.head())

# ── Step 3: Model settings ────────────────────────────────────────────────────

st.subheader("Step 3: Choose and tune your model")

st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Model Type", ["Decision Tree", "Random Forest", "Boosted Trees", "ANN"])

if model_type == "ANN":
    st.sidebar.subheader("Architecture")
    hidden_layers   = st.sidebar.slider("Hidden Layers",    0, 10, 2)
    nodes_per_layer = st.sidebar.slider("Nodes per Layer",  4, max(n_features, 4), 4, step=1)
    activation      = st.sidebar.selectbox("Activation Function", ['relu', 'sigmoid', 'tanh'])
    st.sidebar.subheader("Training")
    learning_rate   = st.sidebar.slider("Learning Rate",  0.001, 0.01, 0.001, step=0.001, format="%.4f")
    epochs          = st.sidebar.slider("Epochs",         3, 250, 20, step=5)
    standardize     = st.sidebar.checkbox("Standardize features", value=True)

elif model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_type == "Random Forest":
    criterion    = st.sidebar.selectbox("Splitting Criterion", ['gini', 'entropy'])
    max_depth    = st.sidebar.slider("Max Depth",              1,  20,  5)
    n_estimators = st.sidebar.slider("Number of Trees",       10, 200, 50, step=10)
    max_features = st.sidebar.slider("Max Features per Split", 1,  10,  3)

elif model_type == "Boosted Trees":
    max_depth    = st.sidebar.slider("Max Depth",       1,  20,  5)
    n_estimators = st.sidebar.slider("Boosting Rounds", 10, 200, 50, step=10)

# ── Train counter ─────────────────────────────────────────────────────────────

if "train_count" not in st.session_state:
    st.session_state["train_count"] = 0

# ── Train button ──────────────────────────────────────────────────────────────

if st.button("🚀 Train and Evaluate"):
    st.session_state["train_count"] += 1

    try:
        X_train = train_df.iloc[:, :-1].values.astype(float)
        y_train = train_df.iloc[:, -1].values.astype(float)
        X_val   = valid_df.iloc[:, :-1].values.astype(float)
        y_val   = valid_df.iloc[:, -1].values.astype(float)

        if model_type == "ANN":
            if standardize:
                scaler  = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val   = scaler.transform(X_val)
                st.session_state["scaler"] = scaler

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
            X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
            y_val_t   = torch.tensor(y_val.reshape(-1, 1),   dtype=torch.float32)

            model     = SimpleANN(X_train.shape[1], hidden_layers, nodes_per_layer, activation)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_losses = [], []
            progress = st.progress(0, text="Training…")

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_train_t), y_train_t)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_t), y_val_t)
                val_losses.append(val_loss.item())
                progress.progress((epoch + 1) / epochs, text=f"Epoch {epoch+1}/{epochs}")

            progress.empty()

            with torch.no_grad():
                y_val_probs = torch.sigmoid(model(X_val_t)).numpy().flatten()
            y_val_pred = (y_val_probs >= 0.5).astype(int)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Loss Curve")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(train_losses, label="Train Loss", color="#0073e6")
                ax.plot(val_losses,   label="Val Loss",   color="#e65c00")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
                plt.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
                st.image(buf, use_column_width=True); plt.close(fig)

            gap = val_losses[-1] - train_losses[-1]
            if gap > 0.05:
                st.warning(f"⚠️ Possible overfitting — val loss is {gap:.3f} above train loss.")
            elif val_losses[-1] < val_losses[0]:
                st.success("✅ Model is learning — val loss decreased.")

            # Store hyperparams string
            hp_str = f"layers={hidden_layers}, nodes={nodes_per_layer}, act={activation}, lr={learning_rate}, epochs={epochs}"

        else:
            if model_type == "Decision Tree":
                model  = DecisionTreeClassifier(max_depth=max_depth)
                hp_str = f"max_depth={max_depth}"
            elif model_type == "Random Forest":
                model  = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                max_features=max_features, criterion=criterion)
                hp_str = f"n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, criterion={criterion}"
            elif model_type == "Boosted Trees":
                model  = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
                hp_str = f"n_estimators={n_estimators}, max_depth={max_depth}"

            model.fit(X_train, y_train)
            y_val_probs = model.predict_proba(X_val)[:, 1]
            y_val_pred  = model.predict(X_val)

            col1, col2 = st.columns(2)
            with col1:
                st.info("Tree models don't have loss curves.")
                if hasattr(model, "feature_importances_"):
                    st.subheader("Feature Importance")
                    fi_df = pd.DataFrame({
                        "Feature":    train_df.columns[:-1].tolist(),
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=True).tail(10)
                    fig, ax = plt.subplots(figsize=(4, max(2, len(fi_df) * 0.35)))
                    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#0073e6")
                    ax.set_xlabel("Importance"); plt.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
                    st.image(buf, use_column_width=True); plt.close(fig)

        # ── Confusion matrix ──
        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()

        with col2:
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=["Pred 0","Pred 1"], yticklabels=["Actual 0","Actual 1"],
                        annot_kws={"size":11}, linewidths=0.5, linecolor='gray', ax=ax_cm)
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            buf = io.BytesIO(); fig_cm.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
            st.image(buf, use_column_width=True); plt.close(fig_cm)

        # ── Metrics ──
        val_auc = roc_auc_score(y_val, y_val_probs)
        val_f1  = f1_score(y_val, y_val_pred)

        st.subheader("Validation Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC",         round(val_auc, 3))
        m2.metric("F1 Score",    round(val_f1, 3))
        m3.metric("Precision",   round(precision_score(y_val, y_val_pred), 3))
        m4.metric("Sensitivity", round(recall_score(y_val, y_val_pred), 3))
        m5.metric("Specificity", round(tn / (tn + fp) if (tn + fp) > 0 else 0, 3))

        st.caption(f"🔁 You have trained {st.session_state['train_count']} time(s) this session.")

        # ── Update run tracker ──
        if "run_history" not in st.session_state:
            st.session_state["run_history"] = []
        st.session_state["run_history"].append({
            "Run":        len(st.session_state["run_history"]) + 1,
            "Model":      model_type,
            "Val AUC":    round(val_auc, 4),
            "Val F1":     round(val_f1, 4),
            "Settings":   hp_str
        })

        # ── Save state for submission ──
        st.session_state["trained_model"]  = model
        st.session_state["model_type"]     = model_type
        st.session_state["val_auc"]        = val_auc
        st.session_state["val_f1"]         = val_f1
        st.session_state["hp_str"]         = hp_str
        st.session_state["model_ready"]    = True
        if model_type == "ANN":
            st.session_state["standardize"] = standardize
            st.session_state["ann_params"]  = {
                "hidden_layers": hidden_layers,
                "nodes": nodes_per_layer,
                "activation": activation
            }

    except Exception as e:
        st.error(f"Training error: {str(e)}")

# ── Run tracker ───────────────────────────────────────────────────────────────

if st.session_state.get("run_history"):
    st.divider()
    st.subheader("📋 Your Training History")
    st.caption("Use this to identify your best configuration before submitting.")
    history_df = pd.DataFrame(st.session_state["run_history"])
    best_idx   = history_df["Val AUC"].idxmax()
    st.dataframe(
        history_df.style.highlight_max(subset=["Val AUC"], color="#d4edda"),
        use_container_width=True,
        hide_index=True
    )
    best = history_df.loc[best_idx]
    st.success(f"⭐ Best run so far: Run {int(best['Run'])} — {best['Model']} with Val AUC {best['Val AUC']} | Settings: {best['Settings']}")

# ── Step 4: Holdout submission ─────────────────────────────────────────────────

if st.session_state.get("model_ready"):
    st.divider()
    st.subheader("Step 4: Submit for Holdout Scoring")
    st.warning("⚠️ You only get **one shot** at holdout scoring. Make sure you're happy with your model before submitting.")

    sheet = get_sheet()

    if sheet:
        existing = get_existing_submissions(sheet)
        already_submitted = st.session_state["confirmed_id"] in existing

        if already_submitted:
            prev = existing[st.session_state["confirmed_id"]]
            st.error(f"❌ ID {st.session_state['confirmed_id']} has already submitted. Your holdout AUC was **{prev['Holdout_AUC']}**.")
        else:
            student_name = st.text_input("Your full name (for the leaderboard)", key="student_name").strip()

            if st.button("🏆 Submit for Holdout Scoring"):
                if not student_name:
                    st.warning("Please enter your name before submitting.")
                else:
                    try:
                        model      = st.session_state["trained_model"]
                        model_type = st.session_state["model_type"]

                        X_holdout = holdout_df.iloc[:, :-1].values.astype(float)
                        y_holdout = holdout_df.iloc[:, -1].values.astype(float)

                        if model_type == "ANN":
                            if st.session_state.get("standardize") and "scaler" in st.session_state:
                                X_holdout = st.session_state["scaler"].transform(X_holdout)
                            X_holdout_t = torch.tensor(X_holdout, dtype=torch.float32)
                            with torch.no_grad():
                                holdout_probs = torch.sigmoid(model(X_holdout_t)).numpy().flatten()
                            holdout_pred = (holdout_probs >= 0.5).astype(int)
                        else:
                            holdout_probs = model.predict_proba(X_holdout)[:, 1]
                            holdout_pred  = model.predict(X_holdout)

                        holdout_auc = roc_auc_score(y_holdout, holdout_probs)
                        holdout_f1  = f1_score(y_holdout, holdout_pred)
                        gap         = holdout_auc - st.session_state["val_auc"]

                        row = [
                            st.session_state["confirmed_id"],
                            student_name,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            model_type,
                            round(st.session_state["val_auc"], 4),
                            round(holdout_auc, 4),
                            round(st.session_state["val_f1"], 4),
                            round(holdout_f1, 4),
                            st.session_state["train_count"],
                            st.session_state["hp_str"]
                        ]

                        if write_submission(sheet, row):
                            st.success(f"✅ Submitted! Your holdout AUC: **{round(holdout_auc, 3)}**")
                            st.info(f"Validation AUC: **{round(st.session_state['val_auc'], 3)}** → Holdout AUC: **{round(holdout_auc, 3)}** | Gap: **{round(gap, 3)}**")
                            if gap < -0.05:
                                st.warning("📉 Your holdout AUC was notably lower than validation — possible overfitting.")

                    except Exception as e:
                        st.error(f"Holdout scoring failed: {str(e)}")
    else:
        st.button("🏆 Submit for Holdout Scoring", disabled=True)
