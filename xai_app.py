import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load dataset
def load_data():
    train_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/retentionchurn/train.csv"
    df = pd.read_csv(train_url)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Train Neural Network
def train_nn(X_train, y_train, input_dim):
    model = SimpleNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    for epoch in range(250):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# Main Streamlit App
def main():
    st.title("Explainable AI: Churn Prediction Demo")

    df = load_data()
    X = df.drop('Retained.in.2012.', axis=1)
    y = df['Retained.in.2012.']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_choice = st.sidebar.selectbox('Choose Model', ['Random Forest', 'Neural Network'])
    explainer_choice = st.sidebar.selectbox('Choose Explanation Method', ['LIME', 'SHAP'])

    # Only show instance index for LIME
    idx = 0
    if explainer_choice == 'LIME':
        idx = st.sidebar.number_input('Test instance index:', min_value=0, max_value=X_test.shape[0]-1, value=0)
        instance = X_test[idx].reshape(1, -1)

    if model_choice == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predict_fn = model.predict_proba
    else:
        model = train_nn(X_train, y_train.values, X_train.shape[1])
        def predict_fn(data):
            with torch.no_grad():
                logits = model(torch.FloatTensor(data)).numpy()
                exp_logits = np.exp(logits)
                return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if explainer_choice == 'LIME':
        st.subheader("LIME Explanation (Instance Level)")
        lime_explainer = LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['Stay', 'Churn'], discretize_continuous=True)
        explanation = lime_explainer.explain_instance(instance[0], predict_fn)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)

        st.subheader("Prediction for Selected Instance")
        pred_proba = predict_fn(instance)[0]
        st.write(f"Probability of Staying: {pred_proba[0]:.2f}")
        st.write(f"Probability of Churn: {pred_proba[1]:.2f}")

    elif explainer_choice == 'SHAP':
        st.subheader("SHAP Explanation (Global Feature Importance)")
        if model_choice == 'Random Forest':
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_test)
            X_test_df = pd.DataFrame(X_test, columns=X.columns)
            
            if isinstance(shap_values, list):
                # For each class: average over samples → (n_features,)
                per_class_mean = [np.abs(sv).mean(axis=0) for sv in shap_values]
                # Then average over classes → (n_features,)
                mean_abs_shap = np.mean(per_class_mean, axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
            st.write("mean_abs_shap shape:", mean_abs_shap.shape)
            st.write("X.columns length:", len(X.columns))
            
            if mean_abs_shap.shape[0] != len(X.columns):
                st.error("Shape mismatch between SHAP values and features! Cannot plot.")
            else:
                shap_df = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs_shap})
                shap_df = shap_df.sort_values(by='mean_abs_shap', ascending=True)
    
                plt.figure(figsize=(8, 6))
                plt.barh(shap_df['feature'], shap_df['mean_abs_shap'])
                plt.xlabel('Mean Absolute SHAP Value')
                plt.title('Feature Importance (SHAP)')
                st.pyplot(plt.gcf())
        else:
            st.write("SHAP for neural networks requires a specialized setup and may be added in a future version.")

if __name__ == "__main__":
    main()
