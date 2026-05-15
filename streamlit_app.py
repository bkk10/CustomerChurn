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

    # Ensure target has no missing values
    if y is not None:
        mask = y.notnull()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()

    # Explicit numeric features we care about
    numeric_feats = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in X.columns]
    # Remaining features treated as categorical
    categorical_feats = [c for c in X.columns if c not in numeric_feats]

    # Coerce numeric columns and impute missing values
    for c in numeric_feats:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X[c].fillna(X[c].median(), inplace=True)

    # Fill missing categorical values with a placeholder
    if categorical_feats:
        X[categorical_feats] = X[categorical_feats].fillna('missing')

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, categorical_feats),
        ]
    )

    if y is None:
        st.error("No target `Churn` column found to train model.")
        return None

    # Fit preprocessor separately so we can inspect the transformed matrix
    try:
        preprocessor.fit(X)
        Xt = preprocessor.transform(X)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

    # Convert sparse to dense for validation
    try:
        if hasattr(Xt, 'toarray'):
            Xt_arr = Xt.toarray()
        else:
            Xt_arr = np.asarray(Xt)
    except Exception:
        Xt_arr = np.asarray(Xt)

    # Replace non-finite values with column medians (or zero if undefined)
    if not np.isfinite(Xt_arr).all():
        st.warning('Non-finite values detected in transformed features; imputing with 0s.')
        Xt_arr[~np.isfinite(Xt_arr)] = 0.0

    # Fit classifier on transformed array
    try:
        clf_final = LogisticRegression(max_iter=1000)
        clf_final.fit(Xt_arr, y)
    except Exception as e:
        st.error(f"Classifier training failed: {e}")
        return None

    # Assemble final pipeline using fitted preprocessor and classifier
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_final)])
    # Save the feature column order so we can recreate inputs at prediction time
    try:
        clf.feature_columns = list(X.columns)
    except Exception:
        clf.feature_columns = list(X.columns)
    joblib.dump(clf, MODEL_FILE)
    return clf


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
    # Try direct prediction first
    try:
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
    except Exception:
        # Build full feature row from preprocessor's expected columns
        feature_cols = None
        try:
            pre = model.named_steps.get('preprocessor', None)
            if pre is not None and hasattr(pre, 'transformers_'):
                cols = []
                for name, trans, columns in pre.transformers_:
                    if columns == 'drop' or columns == 'passthrough':
                        continue
                    # columns may be slice, list, or array-like
                    try:
                        cols.extend(list(columns))
                    except Exception:
                        # fallback if columns is a boolean or other type
                        pass
                feature_cols = cols
        except Exception:
            feature_cols = None

        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = getattr(model, 'feature_columns', list(input_df.columns))

        full = pd.DataFrame({})
        for c in feature_cols:
            if c in input_df.columns:
                full[c] = input_df[c].values
            else:
                if c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                    full[c] = [0]
                else:
                    full[c] = ['missing']

        # Ensure single-row shape
        full = full.astype(object)

        probs = model.predict_proba(full)[:, 1]
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


def get_preprocessor_feature_names(preprocessor, X_columns):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if cols == 'drop':
            continue
        if transformer == 'passthrough':
            try:
                feature_names.extend(list(cols))
            except Exception:
                pass
            continue

        # Numeric columns: keep names as-is
        if hasattr(transformer, 'named_steps') and 'scaler' in transformer.named_steps:
            try:
                feature_names.extend(list(cols))
            except Exception:
                pass
            continue

        # Categorical: expand one-hot feature names
        try:
            # transformer may be a Pipeline with OneHotEncoder
            ohe = None
            if hasattr(transformer, 'named_steps'):
                for step in transformer.named_steps.values():
                    if hasattr(step, 'get_feature_names_out'):
                        ohe = step
                        break
            elif hasattr(transformer, 'get_feature_names_out'):
                ohe = transformer

            if ohe is not None:
                names = list(ohe.get_feature_names_out(cols))
                feature_names.extend(names)
            else:
                try:
                    feature_names.extend(list(cols))
                except Exception:
                    pass
        except Exception:
            try:
                feature_names.extend(list(cols))
            except Exception:
                pass

    # Fallback if empty
    if not feature_names:
        feature_names = list(X_columns)
    return feature_names


def explain_prediction(model, X: pd.DataFrame, top_n: int = 3):
    try:
        pre = model.named_steps.get('preprocessor')
        clf = model.named_steps.get('classifier')
        if pre is None or clf is None:
            return []

        feature_names = get_preprocessor_feature_names(pre, X.columns)
        Xt = pre.transform(X)
        # convert to array
        if hasattr(Xt, 'toarray'):
            Xt_arr = Xt.toarray()
        else:
            Xt_arr = np.asarray(Xt)

        coefs = np.asarray(clf.coef_).reshape(-1)
        sample = Xt_arr[0]
        contribs = coefs * sample
        total = contribs.sum() + float(clf.intercept_[0])

        abs_total = np.abs(contribs).sum()
        if abs_total == 0:
            abs_total = np.abs(total) if np.abs(total) > 0 else 1.0

        items = []
        for name, val in zip(feature_names, contribs):
            items.append((name, float(val), abs(float(val)) / abs_total))

        items.sort(key=lambda x: abs(x[1]), reverse=True)
        top = items[:top_n]

        explanations = []
        for name, val, frac in top:
            direction = 'increases' if val > 0 else 'decreases'
            pct = frac * 100
            # make name human-friendly
            if '_' in name:
                parts = name.split('_')
                pretty = f"{parts[0]} = {'_'.join(parts[1:])}"
            else:
                pretty = name
            explanations.append({'feature': pretty, 'direction': direction, 'contribution': val, 'pct': pct})

        return explanations
    except Exception:
        return []


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

            # Explanation of prediction
            expl = explain_prediction(model, input_df, top_n=3)
            if expl:
                st.subheader('Why the model predicted this')
                for e in expl:
                    sign = '+' if e['contribution'] > 0 else '-'
                    st.write(f"- **{e['feature']}**: {e['direction']} churn ({sign}{abs(e['contribution']):.3f}, {e['pct']:.0f}% of signal)")

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
