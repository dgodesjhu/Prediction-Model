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
import io
import random
import smtplib
from email.message import EmailMessage

# ---------------------- Config & Setup ----------------------
st.set_page_config(page_title="Prediction Model", layout="wide")

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
        .stSlider > div > div {
            color: #003366;
        }
        .stSelectbox, .stTextInput, .stFileUploader {
            background-color: white;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Prediction Model")

# ---------------------- Load Dataset or Upload ----------------------
st.sidebar.header("Choose Dataset")
data_option = st.sidebar.radio("Data Source", ["Use Provided Dataset", "Upload Your Own Data"])

if data_option == "Use Provided Dataset":
    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/bank_marketing"
    train_df = load_data(f"{base_url}/train.csv")
    valid_df = load_data(f"{base_url}/validation.csv")
    test_df = load_data(f"{base_url}/test.csv")
else:
    uploaded_train = st.sidebar.file_uploader("Upload Training Data", type="csv")
    uploaded_valid = st.sidebar.file_uploader("Upload Validation Data", type="csv")
    uploaded_test = st.sidebar.file_uploader("Upload Test Data (optional)", type="csv")

    if uploaded_train and uploaded_valid:
        train_df = pd.read_csv(uploaded_train).dropna()
        valid_df = pd.read_csv(uploaded_valid).dropna()
        test_df = pd.read_csv(uploaded_test).dropna() if uploaded_test else None
    else:
        st.warning("Please upload both training and validation data to proceed.")
        st.stop()

# ---------------------- Model Selection ----------------------
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Model Type", ["ANN", "Decision Tree", "Random Forest", "Boosted Trees"])

input_dim = train_df.shape[1] - 1
standardize = st.sidebar.checkbox("Standardize Features (ANN only)", value=True)

if model_type == "ANN":
    hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
    nodes = st.sidebar.slider("Nodes per Layer", 4, input_dim, min(input_dim, 16))
    activation = st.sidebar.selectbox("Activation", ["relu", "sigmoid", "tanh"])
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.001, step=0.001, format="%.4f")
    epochs = st.sidebar.slider("Epochs", 10, 250, 50, step=10)

if model_type in ["Decision Tree", "Random Forest"]:
    criterion = st.sidebar.selectbox("Splitting Criterion", ["gini", "entropy"])

if model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    max_features = st.sidebar.slider("Max Features", 1, input_dim, min(5, input_dim))

if model_type == "Boosted Trees":
    n_estimators = st.sidebar.slider("Boosting Rounds", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)

class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, nodes, activation):
        super(SimpleANN, self).__init__()
        act_fn = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[activation]
        layers = [nn.Linear(input_dim, nodes), act_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(nodes, nodes), act_fn]
        layers.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

if st.button("Train and Predict"):
    try:
        X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
        X_val, y_val = valid_df.iloc[:, :-1].values, valid_df.iloc[:, -1].values

        if model_type == "ANN":
            if standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

            model = SimpleANN(X_train.shape[1], hidden_layers, nodes, activation)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_losses = [], []
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_train_t), y_train_t)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                with torch.no_grad():
                    val_loss = criterion(model(X_val_t), y_val_t).item()
                    val_losses.append(val_loss)

            with torch.no_grad():
                y_val_probs = torch.sigmoid(model(X_val_t)).numpy().flatten()
                y_val_pred = (y_val_probs >= 0.5).astype(int)

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(train_losses, label="Train Loss")
            ax.plot(val_losses, label="Validation Loss")
            ax.set_title("Loss Curve", fontsize=12)
            ax.set_xlabel("Epochs", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, caption="Training & Validation Loss", width=400)
            plt.close(fig)

        else:
            if model_type == "Decision Tree":
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion)
            elif model_type == "Boosted Trees":
                model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)

            model.fit(X_train, y_train)
            y_val_probs = model.predict_proba(X_val)[:, 1]
            y_val_pred = model.predict(X_val)

        st.subheader("Validation Metrics")
        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = recall_score(y_val, y_val_pred, pos_label=1)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(y_val, y_val_probs)
        f1 = f1_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, pos_label=1)
        recall = sensitivity

        fig_cm, ax_cm = plt.subplots(figsize=(2.5, 2.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"],
                    annot_kws={"size": 10}, linewidths=0.5, linecolor='gray', ax=ax_cm)
        ax_cm.set_title("Confusion Matrix", fontsize=12)
        ax_cm.set_xlabel("Predicted", fontsize=10)
        ax_cm.set_ylabel("Actual", fontsize=10)
        ax_cm.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        buf = io.BytesIO()
        fig_cm.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Confusion Matrix", width=300)
        plt.close(fig_cm)

        st.json({
            "Precision": round(precision, 3),
            "Recall (Sensitivity)": round(recall, 3),
            "Specificity": round(specificity, 3),
            "AUC": round(auc, 3),
            "F1 Score": round(f1, 3)
        })

        st.session_state["trained_model"] = model
        st.session_state["model_ready"] = True
        st.session_state["model_params"] = {
            "input_dim": input_dim,
            "hidden_layers": hidden_layers,
            "nodes": nodes,
            "activation": activation,
            "standardize": standardize
        }

    except Exception as e:
        st.error(f"Training failed: {str(e)}")

# ---------------------- Submission Form ----------------------
if st.session_state.get("model_ready"):
    st.subheader("Submit Your Trained Model")
    with st.form("submit_model_form"):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        jhu_id = st.text_input("JHU ID (9 digits)")
        section = st.selectbox("Section", ["Select One", "DC", "HE"])
        download_model = st.checkbox("Download a copy of your model", value=True)
        submitted = st.form_submit_button("Submit Model")

        if submitted:
            if (not first_name.strip() or not last_name.strip() or not jhu_id.strip().isdigit()
                or len(jhu_id.strip()) < 2 or section == "Select One"):
                st.warning("Please complete all fields correctly before submitting.")
            else:
                try:
                    model = st.session_state["trained_model"]
                    params = st.session_state["model_params"]
                    filename = f"Pred file {last_name}{first_name}{jhu_id[:2]}.pt"

                    torch.save({
                        "state_dict": model.state_dict(),
                        **params
                    }, filename)

                    msg = EmailMessage()
                    msg["Subject"] = f"{section} File Submission {last_name}{first_name}"
                    msg["From"] = "davegodes1@gmail.com"
                    msg["To"] = "dgodes@jhu.edu"
                    msg.set_content(f"Submission from {first_name} {last_name}, ID: {jhu_id}")

                    with open(filename, "rb") as f:
                        msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=filename)

                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                        smtp.login("davegodes1@gmail.com", st.secrets["EMAIL_PASSWORD"])
                        smtp.send_message(msg)

                    st.session_state["submission_success"] = True
                    st.session_state["model_file"] = filename
                    st.session_state["allow_download"] = download_model
                    st.success("✅ Model submitted successfully!")

                except Exception as e:
                    st.error(f"❌ Failed to submit model: {str(e)}")

if st.session_state.get("submission_success") and st.session_state.get("allow_download"):
    try:
        with open(st.session_state["model_file"], "rb") as f:
            st.download_button("Download Your Model", f, file_name=st.session_state["model_file"])
    except Exception as e:
        st.error(f"Could not prepare model for download: {str(e)}")
