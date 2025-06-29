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

    # Preprocess
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']
    
    try:
        df_model = pd.get_dummies(df[features], columns=['type'])
    except KeyError as e:
        st.error(f"❌ Missing column in uploaded CSV: {e}")
        st.stop()

    # Add any missing columns that model expects
    for col in model.feature_names_in_:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[model.feature_names_in_]

    # Predict
    st.subheader("🧠 Running Predictions...")
    preds = model.predict(df_model)
    df['Prediction'] = preds

    # Display all predictions
    st.subheader("✅ Prediction Results")
    st.dataframe(df.style.applymap(
        lambda x: 'background-color: #ffcccc' if x == 1 else '',
        subset=['Prediction']
    ))

    # Filter fraudulent transactions
    frauds = df[df['Prediction'] == 1]
    fraud_count = len(frauds)
    st.subheader(f"⚠️ Detected Fraudulent Transactions ({fraud_count})")

    if fraud_count > 0:
        st.dataframe(frauds)

        # Send fraud alerts to n8n
        st.success("🚨 Sending fraud alerts to n8n...")
        webhook_url = "http://localhost:5678/webhook/fraud-alert"

        for idx, row in frauds.iterrows():
            payload = row.to_dict()

            # 🔒 Explicitly include Prediction: 1
            payload["Prediction"] = 1

            try:
                res = requests.post(webhook_url, json=payload)
                if res.status_code == 200:
                    st.info(f"✅ Alert sent for index {idx}")
                else:
                    st.warning(f"⚠️ HTTP {res.status_code} at index {idx}")
            except Exception as e:
                st.error(f"❌ Error sending alert at index {idx}: {e}")
    else:
        st.info("✅ No fraudulent transactions found.")

    # Download buttons
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
                label="⬇️ Download Fraudulent Only",
                data=frauds.to_csv(index=False),
                file_name="fraud_alerts.csv",
                mime="text/csv"
            )
