import streamlit as st
import pandas as pd


def load_data(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main():
    st.set_page_config(page_title="Customer Churn Explorer", layout="wide")
    st.title("Customer Churn Explorer")

    st.sidebar.header("Data source")
    data_path = st.sidebar.text_input("CSV path", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = load_data(data_path)
    if df is None:
        st.info("Place the dataset CSV next to this app or update the path in the sidebar.")
        return

    st.subheader("Dataset preview")
    st.write(df.head())

    st.subheader("Churn distribution")
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        st.bar_chart(churn_counts)
    else:
        st.warning("No `Churn` column found in dataset.")

    st.subheader("Quick stats")
    st.write(df.describe(include='all'))


if __name__ == "__main__":
    main()
