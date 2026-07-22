import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

print("Loading dataset...")
# Reading from the data folder
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- PREPROCESSING ---
print("Preprocessing dataset...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)

le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
print("Training models...")
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(df[num_cols])

# --- EVALUATION ---
y_pred = dt_classifier.predict(X_test)
print("\n--- MODEL METRICS ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred), 2))
print("Recall:", round(recall_score(y_test, y_pred), 2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- EXPORT MODEL ARTIFACTS ---
# Saving to the models folder
joblib.dump(dt_classifier, 'models/decision_tree_model.pkl')
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("\n✅ All models saved successfully to the models/ folder!")