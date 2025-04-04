import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import random
import joblib
import smtplib
from email.message import EmailMessage

# ---------------------- Config & Setup ----------------------
st.set_page_config(page_title="Prediction Model", layout="wide")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

st.title("Prediction Model")

# ---------------------- Load Dataset ----------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/bank_marketing"
train_df = load_data(f"{base_url}/train.csv")
valid_df = load_data(f"{base_url}/validation.csv")
test_df = load_data(f"{base_url}/test.csv")

# ---------------------- ANN Model Class ----------------------
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

# ---------------------- Model Training ----------------------
st.sidebar.header("Model Settings")
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
nodes = st.sidebar.slider("Nodes per Layer", 4, 64, 16)
activation = st.sidebar.selectbox("Activation", ["relu", "sigmoid", "tanh"])
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.001, step=0.001, format="%.4f")
epochs = st.sidebar.slider("Epochs", 10, 250, 50, step=10)
standardize = st.sidebar.checkbox("Standardize Features", value=True)

if st.button("Train and Predict"):
    try:
        X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
        X_val, y_val = valid_df.iloc[:, :-1].values, valid_df.iloc[:, -1].values

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

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_probs = torch.sigmoid(model(X_val_t)).numpy().flatten()
            y_val_pred = (y_val_probs >= 0.5).astype(int)

        st.subheader("Validation Metrics")
        st.write({
            "Precision": round(precision_score(y_val, y_val_pred), 3),
            "Recall": round(recall_score(y_val, y_val_pred), 3),
            "Specificity": round(confusion_matrix(y_val, y_val_pred)[0, 0] / sum(confusion_matrix(y_val, y_val_pred)[0]), 3),
            "AUC": round(roc_auc_score(y_val, y_val_probs), 3),
            "F1": round(f1_score(y_val, y_val_pred), 3)
        })

        st.session_state["trained_model"] = model
        st.session_state["model_ready"] = True

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
            if not first_name or not last_name or len(jhu_id) < 2 or section == "Select One":
                st.warning("Please complete all fields.")
            else:
                try:
                    model = st.session_state["trained_model"]
                    filename = f"Pred file {last_name}{first_name}{jhu_id[:2]}.pkl"
                    joblib.dump(model, filename)

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

# ---------------------- Download Option ----------------------
if st.session_state.get("submission_success") and st.session_state.get("allow_download"):
    try:
        with open(st.session_state["model_file"], "rb") as f:
            st.download_button("Download Your Model", f, file_name=st.session_state["model_file"])
    except Exception as e:
        st.error(f"Could not prepare model for download: {str(e)}")
