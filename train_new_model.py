import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load your final product list
df = pd.read_csv("products.csv")

# Clean column names
df.columns = ["product", "price", "image"]

# Create a fake label (category) for ML
# We will group products by price range (simple but useful)
df["label"] = pd.cut(
    df["price"],
    bins=[0, 50, 150, 300, 5000],
    labels=[0, 1, 2, 3]
)

# Encode product text
encoder = LabelEncoder()
df["encoded"] = encoder.fit_transform(df["product"])

# Features & labels
X = df[["encoded"]]
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ML model
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# Save files
joblib.dump(model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoder, "encoder.joblib")

print("MODEL TRAINED SUCCESSFULLY 🎉")
