import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

print("📦 Loading dataset...")
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Optional: Sample smaller for demo speed
print("🔍 Sampling dataset (optional for speed)...")
df = df.sample(n=50000, random_state=42)

print("⚖️ Balancing fraud and non-fraud...")
fraud = df[df['isFraud'] == 1]
nonfraud = df[df['isFraud'] == 0].sample(n=len(fraud) * 3, random_state=42)
df = pd.concat([fraud, nonfraud]).sample(frac=1, random_state=42)

print("🧹 Preprocessing features...")
features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest']
df = pd.get_dummies(df[features + ['isFraud']], columns=['type'])

X = df.drop('isFraud', axis=1)
y = df['isFraud']

print("🔀 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🧠 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("💾 Saving model to model.pkl...")
joblib.dump(model, 'model.pkl')

print("📊 Evaluation results:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("🧾 Trained with features:")
print(model.feature_names_in_)

print("✅ Done!")
