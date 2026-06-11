import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_excel("TestData.xlsx")

# Remove patient ID (not useful for prediction)
df = df.drop("id_patient", axis=1)

# Handle missing values
df = df.dropna()

# Target column
target_column = "diagnostic_principal"

# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

# Find target column after encoding
target_cols = [col for col in df.columns if "diagnostic_principal" in col]

# Keep original target before encoding
df_original = pd.read_excel("TestData.xlsx").dropna()
y = df_original[target_column]

# Features
X = df.drop(target_cols, axis=1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Plot
# Top 10 features
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_features["Feature"], top_features["Importance"])
plt.title("Top 10 Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
