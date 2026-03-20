"""
FishSense: Real Model Accuracy Evaluation
Evaluates the model's true generalization ability by:
1. Showing why current 100% accuracy is misleading
2. Adding realistic label noise to simulate real-world conditions
3. Using cross-validation for robust evaluation
4. Testing with held-out geographic regions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import os

print("=" * 70)
print("FISHSENSE - REAL ACCURACY EVALUATION")
print("=" * 70)

# ============================================================================
# 1. LOAD AND RECREATE DATA WITH THE LABELING FUNCTION
# ============================================================================

print("\n📁 Step 1: Loading original data...")

df = pd.read_csv('data/fishsense_training_data.csv')
print(f"✓ Loaded {len(df)} rows")

# Clean data (same as preprocess_data.py)
df_clean = df.dropna(subset=['sst_celsius', 'current_speed_m_s'])

# Remove outliers (same as preprocess_data.py)
for col in ['sst_celsius', 'current_speed_m_s']:
    mean = df_clean[col].mean()
    std = df_clean[col].std()
    df_clean = df_clean[(df_clean[col] >= mean - 3*std) & (df_clean[col] <= mean + 3*std)]

# Feature engineering (same as preprocess_data.py)
mean_temp = df_clean['sst_celsius'].mean()
df_clean = df_clean.copy()
df_clean['temp_deviation'] = df_clean['sst_celsius'] - mean_temp
df_clean['lon_normalized'] = (df_clean['longitude'] - df_clean['longitude'].min()) / \
                              (df_clean['longitude'].max() - df_clean['longitude'].min())
df_clean['lat_normalized'] = (df_clean['latitude'] - df_clean['latitude'].min()) / \
                              (df_clean['latitude'].max() - df_clean['latitude'].min())

# Original deterministic labeling
def calculate_fishing_potential(row):
    score = 0
    if 28.0 <= row['sst_celsius'] <= 29.0:
        score += 2
    elif 27.5 <= row['sst_celsius'] < 28.0 or 29.0 < row['sst_celsius'] <= 29.5:
        score += 1
    if 1.0 <= row['current_speed_m_s'] <= 3.0:
        score += 2
    elif row['current_speed_m_s'] > 3.0:
        score += 1
    if score >= 3:
        return 'High'
    elif score >= 2:
        return 'Medium'
    else:
        return 'Low'

df_clean['fishing_potential'] = df_clean.apply(calculate_fishing_potential, axis=1)

feature_columns = [
    'sst_celsius', 'current_speed_m_s', 'current_u_m_s',
    'current_v_m_s', 'temp_deviation', 'lon_normalized', 'lat_normalized'
]

df_ml = df_clean[feature_columns + ['fishing_potential', 'longitude', 'latitude']].dropna()

print(f"✓ Dataset: {len(df_ml)} samples")
print(f"\nClass distribution:")
print(df_ml['fishing_potential'].value_counts())

X = df_ml[feature_columns].values
y = df_ml['fishing_potential'].values

# ============================================================================
# 2. CONFIRM THE PROBLEM: Deterministic labels = 100% accuracy
# ============================================================================

print("\n" + "=" * 70)
print("📊 TEST 1: Current Model (Deterministic Labels)")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                             min_samples_split=5, min_samples_leaf=2,
                             random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train_s))
test_acc = accuracy_score(y_test, rf.predict(X_test_s))

print(f"\n  Training Accuracy: {train_acc*100:.1f}%")
print(f"  Test Accuracy:     {test_acc*100:.1f}%")
print(f"\n  ⚠️  This is 100% because labels are a deterministic function of features.")
print(f"      The model just re-learns your hand-coded rules.")

# ============================================================================
# 3. REALISTIC EVALUATION: Add noise to simulate real-world conditions
# ============================================================================

print("\n" + "=" * 70)
print("📊 TEST 2: With Realistic Label Noise (Simulating Real-World)")
print("=" * 70)

print("\n  In reality, fishing success depends on many factors not in our data")
print("  (fish migration, weather, season, moon phase, etc.)")
print("  We simulate this by flipping some labels randomly.\n")

noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results = []

for noise in noise_levels:
    accuracies = []
    
    for seed in range(5):  # 5 runs per noise level for stability
        np.random.seed(seed)
        y_noisy = y.copy()
        
        # Randomly flip labels
        n_flip = int(len(y_noisy) * noise)
        flip_indices = np.random.choice(len(y_noisy), n_flip, replace=False)
        classes = ['High', 'Medium', 'Low']
        
        for idx in flip_indices:
            current = y_noisy[idx]
            # Pick a different class
            other_classes = [c for c in classes if c != current]
            y_noisy[idx] = np.random.choice(other_classes)
        
        # Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_noisy, test_size=0.2, random_state=42
        )
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te_s))
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    results.append((noise, mean_acc, std_acc))
    print(f"  Noise {noise*100:5.1f}% → Accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

# ============================================================================
# 4. CROSS-VALIDATION (More robust than single train/test split)
# ============================================================================

print("\n" + "=" * 70)
print("📊 TEST 3: Cross-Validation with Realistic Noise (15%)")
print("=" * 70)

# Use 15% noise as a realistic middle ground
np.random.seed(42)
y_realistic = y.copy()
n_flip = int(len(y_realistic) * 0.15)
flip_indices = np.random.choice(len(y_realistic), n_flip, replace=False)
classes = ['High', 'Medium', 'Low']
for idx in flip_indices:
    current = y_realistic[idx]
    other_classes = [c for c in classes if c != current]
    y_realistic[idx] = np.random.choice(other_classes)

# Scale all features
scaler_cv = StandardScaler()
X_scaled = scaler_cv.fit_transform(X)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)

cv_scores = cross_val_score(rf_cv, X_scaled, y_realistic, cv=cv, scoring='accuracy')

print(f"\n  5-Fold Cross-Validation Results:")
print(f"  ─────────────────────────────────")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score*100:.1f}%")
print(f"  ─────────────────────────────────")
print(f"  Mean:   {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# ============================================================================
# 5. SPATIAL CROSS-VALIDATION (Geographic holdout)
# ============================================================================

print("\n" + "=" * 70)
print("📊 TEST 4: Spatial Cross-Validation (Geographic Holdout)")
print("=" * 70)

print("\n  Testing if model generalizes to unseen geographic regions...")

# Create geographic regions by binning lat/lon
lats = df_ml['latitude'].values
lons = df_ml['longitude'].values

# Divide into geographic grid regions
lat_bins = pd.qcut(lats, q=3, labels=False, duplicates='drop')
lon_bins = pd.qcut(lons, q=3, labels=False, duplicates='drop')
regions = lat_bins * 3 + lon_bins

unique_regions = np.unique(regions)
print(f"  Created {len(unique_regions)} geographic regions")

spatial_scores = []
for region_id in unique_regions:
    test_mask = regions == region_id
    train_mask = ~test_mask
    
    if sum(test_mask) < 5 or sum(train_mask) < 5:
        continue
    
    X_tr = X[train_mask]
    X_te = X[test_mask]
    y_tr = y_realistic[train_mask]
    y_te = y_realistic[test_mask]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te_s))
    spatial_scores.append(acc)
    print(f"  Region {region_id} (n={sum(test_mask)}): {acc*100:.1f}%")

if spatial_scores:
    print(f"\n  Spatial CV Mean: {np.mean(spatial_scores)*100:.1f}% ± {np.std(spatial_scores)*100:.1f}%")

# ============================================================================
# 6. FULL CLASSIFICATION REPORT (with realistic noise)
# ============================================================================

print("\n" + "=" * 70)
print("📊 TEST 5: Detailed Classification Report (15% Noise)")
print("=" * 70)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_realistic, test_size=0.2, random_state=42, stratify=y_realistic
)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

model_final = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model_final.fit(X_tr_s, y_tr)
y_pred = model_final.predict(X_te_s)

print(f"\n{classification_report(y_te, y_pred)}")

print("Confusion Matrix:")
cm = confusion_matrix(y_te, y_pred)
classes_sorted = sorted(np.unique(y_realistic))
print(f"Classes: {classes_sorted}")
print(cm)

# ============================================================================
# SUMMARY
# ============================================================================

best_realistic_acc = results[2][1]  # 15% noise result

print("\n" + "=" * 70)
print("✅ EVALUATION SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    FISHSENSE ACCURACY ANALYSIS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Current (Deterministic Labels):     100.0%  ⚠️  (misleading)      │
│                                                                     │
│  With 15% Realistic Noise:                                         │
│    • Single Split:                   {best_realistic_acc*100:.1f}%                        │
│    • 5-Fold Cross-Validation:        {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%              │
│    • Spatial Cross-Validation:       {np.mean(spatial_scores)*100:.1f}% ± {np.std(spatial_scores)*100:.1f}%              │
│                                                                     │
│  📌 Realistic Accuracy Estimate:     ~{cv_scores.mean()*100:.0f}-{best_realistic_acc*100:.0f}%                      │
│                                                                     │
│  ℹ️  This represents what the model would achieve if labels         │
│     came from real fishing catch data instead of rules.             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# Save updated metadata with realistic accuracy
realistic_metadata = {
    "original_accuracy": 1.0,
    "accuracy_note": "100% due to deterministic label generation from features",
    "realistic_accuracy_15pct_noise": float(best_realistic_acc),
    "cv_accuracy_mean": float(cv_scores.mean()),
    "cv_accuracy_std": float(cv_scores.std()),
    "spatial_cv_mean": float(np.mean(spatial_scores)) if spatial_scores else None,
    "spatial_cv_std": float(np.std(spatial_scores)) if spatial_scores else None
}

with open('models/realistic_accuracy.json', 'w') as f:
    json.dump(realistic_metadata, f, indent=2)

print("✓ Saved realistic accuracy metrics to models/realistic_accuracy.json")
