# ============================================================
# DIABETES PREDICTION - DATA PREPARATION
# This script loads, cleans, and prepares the data
# ============================================================

import pandas as pd
import warnings
import sys

warnings.filterwarnings('ignore')

print("=" * 70)
print("DIABETES PREDICTION PROJECT - DATA PREPARATION")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("\n[STEP 1] Loading African Diabetes Dataset...")
sys.stdout.flush()

from datasets import load_dataset

dataset = load_dataset("electricsheepafrica/african-diabetes-dataset")
print("[SUCCESS] Dataset downloaded!")

# Convert to DataFrame
df = pd.DataFrame(dataset['train'])
print(f"[SUCCESS] Loaded {len(df)} patient records with {len(df.columns)} columns")

# ============================================================
# STEP 2: EXPLORE THE DATA
# ============================================================
print("\n" + "=" * 70)
print("[STEP 2] EXPLORING THE DATA")
print("=" * 70)

print("\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "-" * 70)
print("Target Variable Distribution:")
print("-" * 70)
print(df['diabetes_status'].value_counts())
print("\nPercentages:")
print(df['diabetes_status'].value_counts(normalize=True).mul(100).round(2))

print("\n" + "-" * 70)
print("Missing Values:")
print("-" * 70)
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("  No missing values!")

# ============================================================
# STEP 3: SELECT FEATURES (NO DATA LEAKAGE)
# ============================================================
print("\n" + "=" * 70)
print("[STEP 3] SELECTING FEATURES")
print("=" * 70)

# Features to use (available BEFORE diabetes diagnosis)
features = [
    'age',
    'sex',
    'bmi',
    'family_history_diabetes',
    'physically_active',
    'has_hypertension',
    'fasting_glucose_mg_dl',
    'hba1c_percent',
    'total_cholesterol_mg_dl',
    'ldl_mg_dl',
    'hdl_mg_dl',
    'triglycerides_mg_dl',
    'residence',
    'is_pregnant',
    'pcos',
    'hiv_positive'
]

print(f"\nSelected {len(features)} features:")
for i, feature in enumerate(features, 1):
    print(f"  {i:2d}. {feature}")

# Create feature matrix (X) and target (y)
X = df[features].copy()
y = df['diabetes_status'].copy()

print(f"\n[SUCCESS] Features shape: {X.shape}")
print(f"[SUCCESS] Target shape: {y.shape}")

# ============================================================
# STEP 4: CREATE BINARY TARGET (AT RISK vs NORMAL)
# ============================================================
print("\n" + "=" * 70)
print("[STEP 4] CREATING BINARY TARGET")
print("=" * 70)

# Convert to binary: 1 = At Risk (Prediabetes or Diabetic), 0 = Normal
y_binary = y.apply(lambda x: 1 if x in ['Diabetic', 'Prediabetes'] else 0)

print("\nOriginal categories:")
print(y.value_counts())

print("\nBinary categories:")
print(y_binary.value_counts())
print("\nBinary percentages:")
print(y_binary.value_counts(normalize=True).mul(100).round(2))

# ============================================================
# STEP 5: CHECK FOR MISSING VALUES IN FEATURES
# ============================================================
print("\n" + "=" * 70)
print("[STEP 5] CHECKING FEATURE MISSING VALUES")
print("=" * 70)

feature_missing = X.isnull().sum()
if feature_missing.sum() > 0:
    print("\nMissing values in features:")
    print(feature_missing[feature_missing > 0])
    print("\n[ACTION] No missing values in selected features!")
else:
    print("\n[SUCCESS] No missing values in selected features!")

# ============================================================
# STEP 6: DISPLAY SAMPLE DATA
# ============================================================
print("\n" + "=" * 70)
print("[STEP 6] SAMPLE DATA")
print("=" * 70)

print("\nFirst 5 rows (key features):")
print(df[['age', 'bmi', 'fasting_glucose_mg_dl', 'hba1c_percent', 'diabetes_status']].head())

print("\nBasic statistics:")
print(X[['age', 'bmi', 'fasting_glucose_mg_dl', 'hba1c_percent']].describe())

# ============================================================
# STEP 7: SAVE PREPARED DATA
# ============================================================
print("\n" + "=" * 70)
print("[STEP 7] SAVING PREPARED DATA")
print("=" * 70)

# Save original data
df.to_csv('diabetes_original.csv', index=False)
print("[SUCCESS] Saved: diabetes_original.csv (all original data)")

# Save features
X.to_csv('diabetes_features.csv', index=False)
print("[SUCCESS] Saved: diabetes_features.csv (16 selected features)")

# Save binary target
y_binary_df = pd.DataFrame({
    'diabetes_status_original': y,
    'at_risk': y_binary
})
y_binary_df.to_csv('diabetes_target.csv', index=False)
print("[SUCCESS] Saved: diabetes_target.csv (target variable)")

# Save combined data (features + target)
combined = X.copy()
combined['at_risk'] = y_binary
combined.to_csv('diabetes_prepared.csv', index=False)
print("[SUCCESS] Saved: diabetes_prepared.csv (features + target ready for modeling)")

# ============================================================
# STEP 8: SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal patients: {len(df)}")
print(f"Total features selected: {len(features)}")
print(f"At Risk (Diabetic + Prediabetes): {y_binary.sum()} ({y_binary.sum()/len(y_binary)*100:.1f}%)")
print(f"Normal: {len(y_binary) - y_binary.sum()} ({(len(y_binary) - y_binary.sum())/len(y_binary)*100:.1f}%)")

print("\n" + "=" * 70)
print("FILES SAVED:")
print("=" * 70)
print("1. diabetes_original.csv - Full original dataset")
print("2. diabetes_features.csv - Selected 16 features")
print("3. diabetes_target.csv - Target variable (binary)")
print("4. diabetes_prepared.csv - Ready for model training")

print("\n" + "=" * 70)
print("[COMPLETE] Data preparation finished!")
print("=" * 70)
print("\nNext step: Train machine learning models")
print("Files are saved in: C:\\Users\\user\\Desktop\\HuggingFace\\")