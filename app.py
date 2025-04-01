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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    ["Use Example Dataset", "Upload Your Own Data"]
)

example_dataset = None
if data_option == "Use Example Dataset":
    example_dataset = st.sidebar.selectbox(
        "Choose example dataset:",
        ["Bank Marketing", "Customer Retention/Churn (coming soon)"]
    )

# --- Load Example Dataset if selected ---
if data_option == "Use Example Dataset" and example_dataset == "Bank Marketing":
    @st.cache_data
    def load_example_data():
        base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/bank_marketing"
        train = pd.read_csv(f"{base_url}/train.csv").apply(pd.to_numeric, errors='coerce').dropna()
        valid = pd.read_csv(f"{base_url}/validation.csv").apply(pd.to_numeric, errors='coerce').dropna()
        test = pd.read_csv(f"{base_url}/test.csv").apply(pd.to_numeric, errors='coerce').dropna()
        return train, valid, test

    train_df_cached, valid_df_cached, test_df_cached = load_example_data()

    st.session_state["train_df_cached"] = train_df_cached
    st.session_state["valid_df_cached"] = valid_df_cached
    st.session_state["test_df_cached"] = test_df_cached
    train_file = valid_file = test_file = None  # disables uploads

# --- Upload option fallback ---
else:
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
max_nodes = 128
if model_type == "ANN":
    if train_df_cached is not None:
        max_nodes = train_df_cached.shape[1] - 1

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
    if not train_file or not valid_file:
        st.error("Please upload both training and validation files.")
    else:
        try:
            train_file.seek(0)
            train_df = pd.read_csv(train_file).apply(pd.to_numeric, errors='coerce').dropna()

            valid_file.seek(0)
            valid_df = pd.read_csv(valid_file).apply(pd.to_numeric, errors='coerce').dropna()

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

            if test_file:
                test_file.seek(0)
                test_df = pd.read_csv(test_file).apply(pd.to_numeric, errors='coerce').dropna()
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
