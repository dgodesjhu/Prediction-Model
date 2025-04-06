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
import smtplib
from email.message import EmailMessage

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
#        st.success(f"Successfully loaded data from {url}")
        return data
    except Exception as e:
#        st.error(f"Failed to load data from {url}. Error: {e}")
        return None

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediction Model", layout="wide")

# --- Custom Style ---
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
st.write("Upload training and validation data, choose your model and hyperparameters, and evaluate performance.")

# --- File Uploads in Sidebar ---
# --- In Sidebar: Dataset Selection ---
st.sidebar.header("Choose Dataset")
data_option = st.sidebar.radio(
    "Select data source:",
    ["Use Provided Dataset", "Upload Your Own Data"]
)

example_dataset = st.sidebar.selectbox(
        "Choose dataset:",
        ["Bank Marketing", "Customer Retention/Churn"]
    ) if data_option == "Use Provided Dataset" else None    

# --- Reset old state to avoid mixing files ---
if data_option == "Use Provided Dataset":
    st.session_state.pop("train_file", None)
    st.session_state.pop("valid_file", None)
    st.session_state.pop("test_file", None)
    
if data_option == "Use Provided Dataset":
    # Function to load data from a URL
    @st.cache_data
    def load_data(url):
        try:
            data = pd.read_csv(url)
    #        st.success(f"Successfully loaded data from {url}")
            return data
        except Exception as e:
    #        st.error(f"Failed to load data from {url}. Error: {e}")
            return None

    if example_dataset == "Bank Marketing":
        base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/bank_marketing"
        train_url = f"{base_url}/train.csv"
        valid_url = f"{base_url}/validation.csv"
        test_url = f"{base_url}/test.csv"
    
        train_df = load_data(train_url)
        valid_df = load_data(valid_url)
        test_df = load_data(test_url)
    
        if train_df is not None and valid_df is not None and test_df is not None:
            st.session_state["train_df"] = train_df
            st.session_state["valid_df"] = valid_df
            st.session_state["test_df"] = test_df
        else:
            st.error("Failed to load one or more datasets. Please check the URLs and try again.")
            st.stop()
            
    if example_dataset == "Customer Retention/Churn":
        base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/retentionchurn"
        train_url = f"{base_url}/train.csv"
        valid_url = f"{base_url}/validation.csv"
        
        train_df = load_data(train_url)
        valid_df = load_data(valid_url)
        test_df = None
        
        if train_df is not None and valid_df is not None:
            st.session_state["train_df"] = train_df
            st.session_state["valid_df"] = valid_df
        else:
            st.error("Failed to load one or more datasets. Please check the URLs and try again.")
            st.stop()
            
# --- Upload option fallback ---
else:
    st.session_state.pop("train_df", None)
    st.session_state.pop("valid_df", None)
    st.session_state.pop("test_df", None)
    
    uploaded_train = st.sidebar.file_uploader("Training CSV (with labels)", type="csv", key="train")
    uploaded_valid = st.sidebar.file_uploader("Validation CSV (with labels)", type="csv", key="valid")
    uploaded_test = st.sidebar.file_uploader("Test CSV (no labels)", type="csv", help="Optional", key="test")

    if uploaded_train:
        st.session_state["train_file"] = uploaded_train
    if uploaded_valid:
        st.session_state["valid_file"] = uploaded_valid
    if uploaded_test:
        st.session_state["test_file"] = uploaded_test

    train_file = st.session_state.get("train_file", None)
    valid_file = st.session_state.get("valid_file", None)
    test_file = st.session_state.get("test_file", None)

    train_df_cached = None
    if train_file:
        try:
            train_file.seek(0)
            train_df_cached = pd.read_csv(train_file).apply(pd.to_numeric, errors='coerce').dropna()
            st.session_state["train_df_cached"] = train_df_cached
        except Exception as e:
            st.warning("Could not parse training CSV file.")
            train_df_cached = None

if data_option == "Upload Your Own Data":
    if uploaded_train:
        st.session_state["train_file"] = uploaded_train
    if uploaded_valid:
        st.session_state["valid_file"] = uploaded_valid
    if uploaded_test:
        st.session_state["test_file"] = uploaded_test
        
train_file = st.session_state.get("train_file", None)
valid_file = st.session_state.get("valid_file", None)
test_file = st.session_state.get("test_file", None)

# --- Cache parsed training data to avoid re-reading errors ---
train_df_cached = None
if train_file:
    try:
        train_file.seek(0)
        train_df_cached = pd.read_csv(train_file).apply(pd.to_numeric, errors='coerce').dropna()
        st.session_state["train_df_cached"] = train_df_cached
    except Exception as e:
        st.warning("Could not parse training CSV file.")
        train_df_cached = None

# --- Model Selection ---
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Select Model Type", ["ANN", "Decision Tree", "Random Forest", "Boosted Trees"])

# --- ANN Hyperparameters ---
def get_input_feature_count():
    if data_option == "Upload Your Own Data":
        if "train_df_cached" in st.session_state:
            return st.session_state["train_df_cached"].shape[1] - 1
        elif "train_file" in st.session_state:
            try:
                st.session_state["train_file"].seek(0)
                df = pd.read_csv(st.session_state["train_file"]).apply(pd.to_numeric, errors='coerce').dropna()
                return df.shape[1] - 1
            except:
                return 4
    elif data_option == "Use Provided Dataset":
        if "train_df" in st.session_state:
            return st.session_state["train_df"].shape[1] - 1
    return 4  # fallback

if model_type == "ANN":
    num_features = get_input_feature_count()
    max_nodes = max(num_features, 4)

    hidden_layers = st.sidebar.slider("Hidden Layers", 0, 10, 2)
    nodes_per_layer = st.sidebar.slider("Nodes per Layer", 4, max_nodes, 4, step=1)
    activation = st.sidebar.selectbox("Activation Function", ['relu', 'sigmoid', 'tanh'])
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.001, step=0.001, format="%.4f")
    epochs = st.sidebar.slider("Epochs", 3, 250, 20, step=5)
    standardize = st.sidebar.checkbox("Standardize features", value=True)

# --- Tree Model Hyperparameters ---
if model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_type == "Random Forest":
    criterion = st.sidebar.selectbox("Splitting Criterion", ['gini', 'entropy'])
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 50, step=10)
    max_features = st.sidebar.slider("Max Features per Split", 1, 10, 3)

elif model_type == "Boosted Trees":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    n_estimators = st.sidebar.slider("Boosting Rounds", 10, 200, 50, step=10)

# --- ANN Model Class ---
set_seed(42)

class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, nodes, activation):
        super(SimpleANN, self).__init__()
        layers = []
        act_fn = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[activation]
        layers.append(nn.Linear(input_dim, nodes))
        layers.append(act_fn)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(nodes, nodes))
            layers.append(act_fn)
        layers.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Train & Predict ---
if st.button("Train and Predict"):
    # Determine dataset source
    train_df = None
    valid_df = None
    test_df = None

    if data_option == "Use Provided Dataset":
        train_df = st.session_state.get("train_df", None)
        valid_df = st.session_state.get("valid_df", None)
        test_df = st.session_state.get("test_df", None)
    else:
        if train_file:
            train_file.seek(0)
            train_df = pd.read_csv(train_file).apply(pd.to_numeric, errors='coerce').dropna()
        if valid_file:
            valid_file.seek(0)
            valid_df = pd.read_csv(valid_file).apply(pd.to_numeric, errors='coerce').dropna()
        if test_file:
            test_file.seek(0)
            test_df = pd.read_csv(test_file).apply(pd.to_numeric, errors='coerce').dropna()

    st.write("train_df:", type(train_df), "valid_df:", type(valid_df))
    if train_df is None or valid_df is None:
        st.error("Training and validation data must be available.")
    else:
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

                model = SimpleANN(X_train.shape[1], hidden_layers, nodes_per_layer, activation)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                train_losses = []
                val_losses = []

                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_t)
                    loss = criterion(outputs, y_train_t)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_t)
                        val_loss = criterion(val_outputs, y_val_t)
                        val_losses.append(val_loss.item())

                with torch.no_grad():
                    y_val_probs = torch.sigmoid(model(X_val_t)).numpy().flatten()
                    y_val_pred = (y_val_probs >= 0.5).astype(int)

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(train_losses, label="Train Loss")
                ax.plot(val_losses, label="Val Loss")
                ax.set_title("Loss Curve", fontsize=12)
                ax.set_xlabel("Epochs", fontsize=10)
                ax.set_ylabel("Loss", fontsize=10)
                ax.legend()
                plt.tight_layout()
                buf_loss = io.BytesIO()
                fig.savefig(buf_loss, format="png", dpi=100, bbox_inches="tight")
                buf_loss.seek(0)
                st.image(buf_loss, caption="Training & Validation Loss", width=400)
                plt.close(fig)

            else:
                if model_type == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=max_depth)
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                   max_features=max_features, criterion=criterion)
                elif model_type == "Boosted Trees":
                    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)

                model.fit(X_train, y_train)
                y_val_probs = model.predict_proba(X_val)[:, 1]
                y_val_pred = model.predict(X_val)
                st.info("Tree models are not trained in epochs like ANNs, so loss curves aren't applicable here.")

            cm = confusion_matrix(y_val, y_val_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = recall_score(y_val, y_val_pred, pos_label=1)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            auc = roc_auc_score(y_val, y_val_probs)
            f1 = f1_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, pos_label=1)
            recall = sensitivity  # for clarity

            st.subheader("Validation Metrics")

            fig_cm, ax_cm = plt.subplots(figsize=(2.5, 2.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"],
                annot_kws={"size": 10},
                linewidths=0.5,
                linecolor='gray',
                ax=ax_cm
            )
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
                
            if model_type == "ANN":
                st.session_state["model_params"] = {
                    "input_dim": X_train.shape[1],
                    "hidden_layers": hidden_layers,
                    "nodes": nodes_per_layer,
                    "activation": activation,
                    "standardize": standardize
                }

            if test_df is not None:
                X_test = test_df.values
                if model_type == "ANN":
                    if standardize:
                        X_test = scaler.transform(X_test)
                    X_test_t = torch.tensor(X_test, dtype=torch.float32)
                    with torch.no_grad():
                        preds = torch.sigmoid(model(X_test_t)).numpy().flatten()
                else:
                    preds = model.predict_proba(X_test)[:, 1]
                st.subheader("Predictions on Test Set")
                st.json(preds.tolist())
                
        except Exception as e:
            st.error(f"Error during training or evaluation: {str(e)}")

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
