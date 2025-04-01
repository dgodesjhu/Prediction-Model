# --- In Sidebar: Dataset Selection ---
st.sidebar.header("Choose Dataset")
data_option = st.sidebar.radio(
    "Select data source:",
    ["Use Provided Dataset", "Upload Your Own Data"]
)

example_dataset = None
train_df_cached = valid_df_cached = test_df_cached = None
train_file = valid_file = test_file = None

if data_option == "Use Provided Dataset":
    example_dataset = st.sidebar.selectbox(
        "Choose example dataset:",
        ["Bank Marketing", "Customer Retention/Churn (coming soon)"]
    )

    if example_dataset == "Bank Marketing":
        def load_example_data():
            base_url = "https://raw.githubusercontent.com/dgodesjhu/Prediction-Model/main/data/bank_marketing"
            st.write("🔍 Trying to read example data from:", base_url)
            train = pd.read_csv(f"{base_url}/train.csv").apply(pd.to_numeric, errors='coerce').dropna()
            valid = pd.read_csv(f"{base_url}/validation.csv").apply(pd.to_numeric, errors='coerce').dropna()
            test = pd.read_csv(f"{base_url}/test.csv").apply(pd.to_numeric, errors='coerce').dropna()
            return train, valid, test

        train_df_cached, valid_df_cached, test_df_cached = load_example_data()
        st.session_state["train_df_cached"] = train_df_cached
        st.session_state["valid_df_cached"] = valid_df_cached
        st.session_state["test_df_cached"] = test_df_cached

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

    if train_file:
        try:
            train_file.seek(0)
            train_df_cached = pd.read_csv(train_file).apply(pd.to_numeric, errors='coerce').dropna()
            st.session_state["train_df_cached"] = train_df_cached
        except Exception as e:
            st.warning("Could not parse training CSV file.")

# --- Training Trigger ---
if st.button("Train and Predict"):
    train_df = st.session_state.get("train_df_cached", None)
    valid_df = st.session_state.get("valid_df_cached", None)

    if train_df is None or valid_df is None:
        st.error("No training/validation data found. Please upload files or select a dataset.")
    else:
        # now safe to proceed with training
        st.success("✅ Data successfully loaded. Proceeding with training...")

        # The rest of your training logic goes here (same as before)...
