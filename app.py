import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-Time Fraud Detection System")

# File uploader
uploaded_file = st.file_uploader("📤 Upload transaction data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # Load trained model
    try:
        model = joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file 'model.pkl' not found. Please train the model first.")
        st.stop()

    # Required features
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']

    # Preprocess and get dummies
    try:
        df_model = pd.get_dummies(df[features], columns=['type'])
    except KeyError as e:
        st.error(f"❌ Missing required column in CSV: {e}")
        st.stop()

    # Ensure all model columns are present
    for col in model.feature_names_in_:
        if col not in df_model.columns:
            df_model[col] = 0  # Add missing dummy column
    df_model = df_model[model.feature_names_in_]  # Align column order

    # Predict
    st.subheader("🧠 Running Predictions...")
    preds = model.predict(df_model)
    df['Prediction'] = preds

    # Show full prediction table
    st.subheader("✅ Prediction Results")
    st.dataframe(
        df.style.applymap(
            lambda x: 'background-color: #ffcccc' if x == 1 else '',
            subset=['Prediction']
        )
    )

    # Extract fraud predictions
    frauds = df[df['Prediction'] == 1]
    fraud_count = len(frauds)

    st.subheader(f"⚠️ Detected Fraudulent Transactions ({fraud_count})")

    if fraud_count > 0:
        st.dataframe(frauds)

        st.success("🚨 Sending fraud alerts to n8n...")

        # ✅ Updated webhook URL to use ngrok public link
        webhook_url = "https://9a6e-2409-40f4-304a-15d-313d-7a4-c889-476a.ngrok-free.app/webhook/fraud-alert"

        for idx, row in frauds.iterrows():
            payload = row.to_dict()
            payload["Prediction"] = 1  # Ensure it's included

            try:
                res = requests.post(webhook_url, json=payload)
                if res.status_code == 200:
                    st.info(f"✅ Alert sent for transaction at index {idx}")
                else:
                    st.warning(f"⚠️ Failed to send alert (HTTP {res.status_code}) at index {idx}")
            except Exception as e:
                st.error(f"❌ Exception sending alert at index {idx}: {e}")
    else:
        st.info("✅ No fraud detected in this dataset.")

    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download All Predictions",
            data=df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
    with col2:
        if fraud_count > 0:
            st.download_button(
                label="⬇️ Download Only Fraudulent Transactions",
                data=frauds.to_csv(index=False),
                file_name="fraud_alerts.csv",
                mime="text/csv"
            )
