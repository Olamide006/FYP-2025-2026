# ============================================================
# DIABETES PREDICTION - MODEL TRAINING
# This script trains and compares multiple ML models
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("=" * 70)
print("DIABETES PREDICTION PROJECT - MODEL TRAINING")
print("=" * 70)

# ============================================================
# STEP 1: LOAD PREPARED DATA
# ============================================================
print("\n[STEP 1] Loading prepared data...")

try:
    df = pd.read_csv('diabetes_prepared.csv')
    print(f"[SUCCESS] Loaded {len(df)} records")
    print(f"[SUCCESS] Features: {len(df.columns) - 1}")
except FileNotFoundError:
    print("[ERROR] diabetes_prepared.csv not found!")
    print("Please run 1_prepare_data.py first")
    exit()

# Separate features and target
X = df.drop('at_risk', axis=1)
y = df['at_risk']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())
print(f"\nPercentages:")
print(y.value_counts(normalize=True).mul(100).round(2))

# ============================================================
# STEP 2: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 70)
print("[STEP 2] Encoding categorical variables...")
print("=" * 70)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  Encoded: {col}")

print(f"[SUCCESS] Encoded {len(categorical_cols)} categorical columns")

# ============================================================
# STEP 3: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 3] Splitting data...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTraining set distribution:")
print(y_train.value_counts())
print(f"\nTesting set distribution:")
print(y_test.value_counts())

# ============================================================
# STEP 4: SCALE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[STEP 4] Scaling features...")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[SUCCESS] Features scaled (mean=0, std=1)")

# ============================================================
# STEP 5: TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 5] Training models...")
print("=" * 70)

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Store results
results = []
trained_models = {}

print("\nTraining models (this may take a few minutes)...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC-ROC if probability predictions available
    if y_pred_proba is not None:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_roc = None
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    })
    
    # Store trained model
    trained_models[name] = model
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if auc_roc:
        print(f"  AUC-ROC: {auc_roc:.4f}")
    print()

print("[SUCCESS] All models trained!")

# ============================================================
# STEP 6: COMPARE MODELS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 6] Model Comparison")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n[BEST MODEL] {best_model_name}")
print(f"[ACCURACY] {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================
# STEP 7: DETAILED EVALUATION OF BEST MODEL
# ============================================================
print("\n" + "=" * 70)
print(f"[STEP 7] Detailed Evaluation - {best_model_name}")
print("=" * 70)

best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, 
                          target_names=['Normal', 'At Risk']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(f"True Negatives (Normal correctly predicted): {cm[0][0]}")
print(f"False Positives (Normal wrongly predicted as At Risk): {cm[0][1]}")
print(f"False Negatives (At Risk wrongly predicted as Normal): {cm[1][0]}")
print(f"True Positives (At Risk correctly predicted): {cm[1][1]}")

# ============================================================
# STEP 8: VISUALIZE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 8] Creating visualizations...")
print("=" * 70)

# Plot 1: Model Comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: model_comparison.png")
plt.close()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'At Risk'],
            yticklabels=['Normal', 'At Risk'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: confusion_matrix.png")
plt.close()

# Plot 3: Feature Importance (for tree-based models)
if best_model_name in ['Decision Tree', 'Random Forest']:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'].head(10), 
             feature_importance['Importance'].head(10))
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: feature_importance.png")
    plt.close()
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# ============================================================
# STEP 9: SAVE MODELS AND RESULTS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 9] Saving models and results...")
print("=" * 70)

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"[SUCCESS] Saved best model: best_model.pkl ({best_model_name})")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("[SUCCESS] Saved scaler: scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("[SUCCESS] Saved label encoders: label_encoders.pkl")

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("[SUCCESS] Saved results: model_results.csv")

# Save feature names
feature_names = X.columns.tolist()
pd.DataFrame({'features': feature_names}).to_csv('feature_names.csv', index=False)
print("[SUCCESS] Saved feature names: feature_names.csv")

# ============================================================
# STEP 10: SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy*100:.2f}%")
print(f"Total models trained: {len(models)}")

print("\n" + "=" * 70)
print("FILES SAVED:")
print("=" * 70)
print("1. best_model.pkl - Best trained model")
print("2. scaler.pkl - Feature scaler")
print("3. label_encoders.pkl - Categorical encoders")
print("4. model_results.csv - All model performances")
print("5. feature_names.csv - Feature names")
print("6. model_comparison.png - Performance comparison chart")
print("7. confusion_matrix.png - Confusion matrix visualization")
if best_model_name in ['Decision Tree', 'Random Forest']:
    print("8. feature_importance.png - Feature importance chart")

print("\n" + "=" * 70)
print("[COMPLETE] Model training finished!")
print("=" * 70)
print("\nNext step: Create web interface (3_web_app.py)")