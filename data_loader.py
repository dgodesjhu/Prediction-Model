  import pandas as pd
  import streamlit as st

  @st.cache_data
  def load_data(url):
      try:
          data = pd.read_csv(url)
          st.success(f"Successfully loaded data from {url}")
          return data
      except Exception as e:
          st.error(f"Failed to load data from {url}. Error: {e}")
          return None
