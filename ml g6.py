# export_svc_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("water_potability.csv")
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# Define model and pipeline
svc = SVC(probability=True, class_weight='balanced', C=1, kernel='rbf')
calibrated_svc = CalibratedClassifierCV(estimator=svc, cv=5)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', calibrated_svc)
])

# Train
pipeline.fit(X_train, y_train)

# Get best threshold on validation set
val_probs = pipeline.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thr = thresholds[np.argmax(f1_scores)]

# Evaluate
test_probs = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (test_probs >= best_thr).astype(int)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"✅ Best Threshold: {best_thr:.4f}")
print(f"✅ Test Accuracy (at threshold): {test_acc:.4f}")

# Save
joblib.dump({
    "pipeline": pipeline,
    "threshold": best_thr
}, "best_model_svc_calibrated.pkl")

print("\n🎉 Model and threshold saved as 'best_model_svc_calibrated.pkl'")
