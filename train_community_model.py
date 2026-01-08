# ============================================================
# DIABETES PREDICTION - COMMUNITY SCREENING MODEL
# Train model WITHOUT lab tests for resource-limited settings
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
                             roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("=" * 70)
print("COMMUNITY SCREENING MODEL - WITHOUT LAB TESTS")
print("For Resource-Limited Settings")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[STEP 1] Loading prepared data...")

try:
    df = pd.read_csv('diabetes_prepared.csv')
    print(f"[SUCCESS] Loaded {len(df)} records")
except FileNotFoundError:
    print("[ERROR] diabetes_prepared.csv not found!")
    print("Please run 1_prepare_data.py first")
    exit()

# ============================================================
# STEP 2: SELECT FEATURES (NO LAB TESTS)
# ============================================================
print("\n" + "=" * 70)
print("[STEP 2] Selecting features WITHOUT lab tests...")
print("=" * 70)

# Features available without lab tests
community_features = [
    'age',
    'sex',
    'bmi',
    'family_history_diabetes',
    'physically_active',
    'has_hypertension',
    'residence',
    'is_pregnant',
    'pcos',
    'hiv_positive'
]

print(f"\nSelected {len(community_features)} community-accessible features:")
for i, feature in enumerate(community_features, 1):
    print(f"  {i:2d}. {feature}")

print("\nEXCLUDED lab test features:")
excluded = ['fasting_glucose_mg_dl', 'hba1c_percent', 'total_cholesterol_mg_dl', 
            'ldl_mg_dl', 'hdl_mg_dl', 'triglycerides_mg_dl']
for feature in excluded:
    print(f"  - {feature}")

# Create feature matrix
X = df[community_features].copy()
y = df['at_risk']

print(f"\n[SUCCESS] Features shape: {X.shape}")
print(f"[SUCCESS] Target shape: {y.shape}")

# ============================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 70)
print("[STEP 3] Encoding categorical variables...")
print("=" * 70)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_cols}")

label_encoders_community = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders_community[col] = le
    print(f"  Encoded: {col}")

print(f"[SUCCESS] Encoded {len(categorical_cols)} columns")

# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("[STEP 4] Splitting data...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# STEP 5: SCALE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[STEP 5] Scaling features...")
print("=" * 70)

scaler_community = StandardScaler()
X_train_scaled = scaler_community.fit_transform(X_train)
X_test_scaled = scaler_community.transform(X_test)

print("[SUCCESS] Features scaled")

# ============================================================
# STEP 6: TRAIN MODELS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 6] Training models...")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
trained_models = {}

print("\nTraining models (this may take a few minutes)...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if y_pred_proba is not None:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_roc = None
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    })
    
    trained_models[name] = model
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if auc_roc:
        print(f"  AUC-ROC: {auc_roc:.4f}")
    print()

print("[SUCCESS] All models trained!")

# ============================================================
# STEP 7: COMPARE MODELS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 7] Model Comparison")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\nCommunity Screening Model Performance:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n[BEST MODEL] {best_model_name}")
print(f"[ACCURACY] {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================
# STEP 8: DETAILED EVALUATION
# ============================================================
print("\n" + "=" * 70)
print(f"[STEP 8] Detailed Evaluation - {best_model_name}")
print("=" * 70)

best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, 
                          target_names=['Normal', 'At Risk']))

cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")

# ============================================================
# STEP 9: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("[STEP 9] Creating visualizations...")
print("=" * 70)

# Model Comparison Chart
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Community Screening Model Performance (Without Lab Tests)')
plt.xticks(x + width*1.5, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('community_model_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: community_model_comparison.png")
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'At Risk'],
            yticklabels=['Normal', 'At Risk'])
plt.title(f'Confusion Matrix - Community Model\n{best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('community_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: community_confusion_matrix.png")
plt.close()

# Feature Importance (if applicable)
if best_model_name in ['Decision Tree', 'Random Forest']:
    feature_importance = pd.DataFrame({
        'Feature': community_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - Community Model\n{best_model_name}')
    plt.tight_layout()
    plt.savefig('community_feature_importance.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: community_feature_importance.png")
    plt.close()
    
    print("\nTop Features for Community Screening:")
    print(feature_importance.to_string(index=False))

# ============================================================
# STEP 10: SAVE COMMUNITY MODEL
# ============================================================
print("\n" + "=" * 70)
print("[STEP 10] Saving community screening model...")
print("=" * 70)

joblib.dump(best_model, 'community_model.pkl')
print(f"[SUCCESS] Saved: community_model.pkl ({best_model_name})")

joblib.dump(scaler_community, 'community_scaler.pkl')
print("[SUCCESS] Saved: community_scaler.pkl")

joblib.dump(label_encoders_community, 'community_label_encoders.pkl')
print("[SUCCESS] Saved: community_label_encoders.pkl")

pd.DataFrame({'features': community_features}).to_csv('community_feature_names.csv', index=False)
print("[SUCCESS] Saved: community_feature_names.csv")

results_df.to_csv('community_model_results.csv', index=False)
print("[SUCCESS] Saved: community_model_results.csv")

# ============================================================
# STEP 11: COMPARISON WITH CLINICAL MODEL
# ============================================================
print("\n" + "=" * 70)
print("[STEP 11] Comparison: Clinical vs Community Models")
print("=" * 70)

try:
    clinical_results = pd.read_csv('model_results.csv')
    clinical_best = clinical_results.iloc[0]
    
    print("\nCLINICAL MODEL (With Lab Tests):")
    print(f"  Best Model: {clinical_best['Model']}")
    print(f"  Accuracy: {clinical_best['Accuracy']*100:.2f}%")
    print(f"  Features: 16 (includes lab tests)")
    
    print("\nCOMMUNITY MODEL (Without Lab Tests):")
    print(f"  Best Model: {best_model_name}")
    print(f"  Accuracy: {best_accuracy*100:.2f}%")
    print(f"  Features: {len(community_features)} (no lab tests required)")
    
    accuracy_drop = (clinical_best['Accuracy'] - best_accuracy) * 100
    print(f"\nAccuracy Trade-off: {accuracy_drop:.2f}% lower")
    print("Benefit: Accessible without expensive lab tests")
    
except FileNotFoundError:
    print("\n[NOTE] Clinical model results not found for comparison")

# ============================================================
# STEP 12: SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY - COMMUNITY SCREENING MODEL")
print("=" * 70)

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy*100:.2f}%")
print(f"Features Used: {len(community_features)} (no lab tests)")

print("\n" + "=" * 70)
print("FILES SAVED:")
print("=" * 70)
print("1. community_model.pkl - Trained community model")
print("2. community_scaler.pkl - Feature scaler")
print("3. community_label_encoders.pkl - Encoders")
print("4. community_feature_names.csv - Feature list")
print("5. community_model_results.csv - Performance metrics")
print("6. community_model_comparison.png - Performance chart")
print("7. community_confusion_matrix.png - Confusion matrix")
if best_model_name in ['Decision Tree', 'Random Forest']:
    print("8. community_feature_importance.png - Feature importance")

print("\n" + "=" * 70)
print("[COMPLETE] Community screening model ready!")
print("=" * 70)
print("\nNext: Update web app to support both models")