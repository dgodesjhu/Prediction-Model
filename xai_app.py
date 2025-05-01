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


# Load dataset
def load_data():
    train_url="https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/retentionchurn/train.csv"
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
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
                return model(torch.FloatTensor(data)).numpy()

    if explainer_choice == 'LIME':
        st.subheader("LIME Explanation")
        lime_explainer = LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['Stay', 'Churn'], discretize_continuous=True)
        explanation = lime_explainer.explain_instance(instance[0], predict_fn)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)

    elif explainer_choice == 'SHAP':
        st.subheader("SHAP Explanation")
        if model_choice == 'Random Forest':
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_test)
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, plot_type='bar', show=False)
            st.pyplot(plt.gcf())
        else:
            st.write("SHAP for neural networks requires a specialized setup and may be added in a future version.")

    st.subheader("Prediction")
    pred_proba = predict_fn(instance)[0]
    st.write(f"Probability of Staying: {pred_proba[0]:.2f}")
    st.write(f"Probability of Churn: {pred_proba[1]:.2f}")

if __name__ == "__main__":
    main()
