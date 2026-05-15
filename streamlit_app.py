import os
import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


MODEL_FILE = "model.joblib"


def load_data(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df


def build_and_train_model(df: pd.DataFrame):
    df = prepare_dataframe(df)
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if 'Churn' in df.columns else None

    numeric_feats = [c for c in X.columns if X[c].dtype in [np.float64, np.int64] or c in ['tenure', 'MonthlyCharges', 'TotalCharges']]
    categorical_feats = [c for c in X.columns if c not in numeric_feats]

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, categorical_feats),
        ]
    )

    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])

    if y is not None:
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        return clf
    else:
        st.error("No target `Churn` column found to train model.")
        return None


def load_or_train(path: str):
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            return model
        except Exception:
            os.remove(MODEL_FILE)

    df = load_data(path)
    if df is None:
        return None
    return build_and_train_model(df)


def predict_single(model, input_df: pd.DataFrame):
    probs = model.predict_proba(input_df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def retention_tip(row: pd.Series, prob: float) -> str:
    tips = []
    if row.get('Contract') == 'Month-to-month':
        tips.append('Offer a discount for switching to a 1-year contract')
    if row.get('InternetService') == 'Fiber optic' and row.get('MonthlyCharges', 0) > 80:
        tips.append('Offer a promotional discount on Fiber plan')
    if row.get('tenure', 0) <= 6:
        tips.append('Provide a welcome offer or onboarding support')
    if row.get('PaymentMethod') == 'Electronic check':
        tips.append('Offer alternate payment options or incentives')
    if not tips:
        tips.append('Send targeted retention email with loyalty benefits')
    return ' / '.join(tips) + f' (churn probability {prob*100:.0f}%)'


def app_ui():
    st.set_page_config(page_title="Customer Churn Explorer", layout="wide")
    st.title("Customer Churn Explorer")

    st.sidebar.header("Data & Model")
    data_path = st.sidebar.text_input("CSV path", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if st.sidebar.button('(Re)train model'):
        with st.spinner('Training model...'):
            model = load_or_train(data_path)
            if model is not None:
                st.sidebar.success('Model trained and saved locally')

    model = None
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
        except Exception:
            model = load_or_train(data_path)

    df = load_data(data_path)
    if df is None:
        st.info("Place the dataset CSV next to this app or update the path in the sidebar.")
        return

    df = prepare_dataframe(df)

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Churn distribution")
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        st.bar_chart(churn_counts)
    else:
        st.warning("No `Churn` column found in dataset.")

    st.sidebar.header('Single-customer prediction')
    with st.sidebar.form('single_pred'):
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=200, value=int(df['tenure'].median()))
        monthly = st.number_input('MonthlyCharges', min_value=0.0, value=float(df['MonthlyCharges'].median()))
        contract = st.selectbox('Contract', options=sorted(df['Contract'].dropna().unique()))
        internet = st.selectbox('InternetService', options=sorted(df['InternetService'].dropna().unique()))
        tech = st.selectbox('TechSupport', options=sorted(df['TechSupport'].dropna().unique()))
        payment = st.selectbox('PaymentMethod', options=sorted(df['PaymentMethod'].dropna().unique()))
        submit = st.form_submit_button('Predict single')

    if submit:
        if model is None:
            st.error('Model not available. Click (Re)train model in sidebar.')
        else:
            input_df = pd.DataFrame([{
                'tenure': tenure,
                'MonthlyCharges': monthly,
                'TotalCharges': monthly * max(tenure, 1),
                'Contract': contract,
                'InternetService': internet,
                'TechSupport': tech,
                'PaymentMethod': payment,
            }])
            preds, probs = predict_single(model, input_df)
            churn_prob = float(probs[0])
            st.metric('Churn probability', f'{churn_prob*100:.1f}%')
            tip = retention_tip(input_df.iloc[0], churn_prob)
            st.info(tip)

    st.subheader('Batch scoring')
    uploaded = st.file_uploader('Upload CSV to score', type=['csv'])
    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        uploaded_df = prepare_dataframe(uploaded_df)
        if model is None:
            st.error('Model not available. Click (Re)train model in sidebar.')
        else:
            X = uploaded_df.drop(columns=['customerID', 'Churn'], errors='ignore')
            probs = model.predict_proba(X)[:, 1]
            uploaded_df['churn_probability'] = probs
            uploaded_df['retention_tip'] = uploaded_df.apply(lambda r: retention_tip(r, r['churn_probability']), axis=1)
            st.dataframe(uploaded_df.head())
            csv_bytes = uploaded_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download scored CSV', data=csv_bytes, file_name='scored_customers.csv', mime='text/csv')


def main():
    app_ui()


if __name__ == "__main__":
    main()
