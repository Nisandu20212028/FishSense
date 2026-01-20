"""
FishSense: Data Preprocessing
Cleans and prepares oceanographic data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 70)
print("FISHSENSE DATA PREPROCESSING")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n📁 Step 1: Loading data...")

# Load the CSV file
data_file = "data/fishsense_training_data.csv"

if not os.path.exists(data_file):
    print(f"❌ Error: {data_file} not found!")
    print("   Please make sure your CSV file is in the data/ folder")
    exit()

df = pd.read_csv(data_file)

print(f"✓ Loaded {len(df)} rows")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================================

print("\n📊 Step 2: Initial data exploration...")

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nBasic statistics:")
print(df.describe())

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================

print("\n🧹 Step 3: Handling missing values...")

# Count missing values
missing_before = df.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_before}")

# Strategy: Remove rows with missing SST or current data
# (These are our key features)
df_clean = df.dropna(subset=['sst_celsius', 'current_speed_m_s'])

missing_after = df_clean.isnull().sum().sum()
rows_removed = len(df) - len(df_clean)

print(f"Rows removed: {rows_removed}")
print(f"Remaining rows: {len(df_clean)}")
print(f"Missing values after cleaning: {missing_after}")

# ============================================================================
# 4. REMOVE OUTLIERS
# ============================================================================

print("\n🎯 Step 4: Removing outliers...")

def remove_outliers(df, column, n_std=3):
    """Remove outliers using standard deviation method"""
    mean = df[column].mean()
    std = df[column].std()
    lower = mean - (n_std * std)
    upper = mean + (n_std * std)
    
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    after = len(df)
    
    print(f"  {column}: removed {before - after} outliers")
    return df

# Remove outliers from key features
df_clean = remove_outliers(df_clean, 'sst_celsius')
df_clean = remove_outliers(df_clean, 'current_speed_m_s')

print(f"✓ Rows after outlier removal: {len(df_clean)}")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

print("\n⚙️ Step 5: Feature engineering...")

# Create additional features

# 1. Temperature deviation from mean
mean_temp = df_clean['sst_celsius'].mean()
df_clean['temp_deviation'] = df_clean['sst_celsius'] - mean_temp

# 2. Current strength category
df_clean['current_strength'] = pd.cut(
    df_clean['current_speed_m_s'],
    bins=[0, 1, 3, 10],
    labels=['weak', 'moderate', 'strong']
)

# 3. Spatial features (normalized coordinates)
df_clean['lon_normalized'] = (df_clean['longitude'] - df_clean['longitude'].min()) / \
                              (df_clean['longitude'].max() - df_clean['longitude'].min())
df_clean['lat_normalized'] = (df_clean['latitude'] - df_clean['latitude'].min()) / \
                              (df_clean['latitude'].max() - df_clean['latitude'].min())

print("✓ Created features:")
print("  - temp_deviation: how much warmer/cooler than average")
print("  - current_strength: weak/moderate/strong")
print("  - lon_normalized, lat_normalized: spatial features")

# ============================================================================
# 6. CREATE TARGET VARIABLE (FISHING POTENTIAL)
# ============================================================================

print("\n🎯 Step 6: Creating target variable (fishing potential)...")

# Define fishing potential based on oceanographic conditions
# This is a simplified model - in reality you'd use historical catch data

def calculate_fishing_potential(row):
    """
    Calculate fishing potential based on oceanographic parameters
    
    Rules (simplified):
    - Optimal SST: 28-29°C
    - Moderate currents (1-3 m/s) are good
    - Strong currents (>3 m/s) can indicate upwelling (also good)
    """
    score = 0
    
    # SST scoring
    if 28.0 <= row['sst_celsius'] <= 29.0:
        score += 2  # Optimal
    elif 27.5 <= row['sst_celsius'] < 28.0 or 29.0 < row['sst_celsius'] <= 29.5:
        score += 1  # Good
    
    # Current scoring
    if 1.0 <= row['current_speed_m_s'] <= 3.0:
        score += 2  # Moderate - good for feeding
    elif row['current_speed_m_s'] > 3.0:
        score += 1  # Strong - might indicate upwelling
    
    # Classify into categories
    if score >= 3:
        return 'High'
    elif score >= 2:
        return 'Medium'
    else:
        return 'Low'

df_clean['fishing_potential'] = df_clean.apply(calculate_fishing_potential, axis=1)

print("\nFishing potential distribution:")
print(df_clean['fishing_potential'].value_counts())

# ============================================================================
# 7. PREPARE FEATURES FOR ML
# ============================================================================

print("\n🔧 Step 7: Preparing features for ML...")

# Select features for modeling
feature_columns = [
    'sst_celsius',
    'current_speed_m_s',
    'current_u_m_s',
    'current_v_m_s',
    'temp_deviation',
    'lon_normalized',
    'lat_normalized'
]

# Keep coordinates for visualization but not for ML
df_ml = df_clean[feature_columns + ['fishing_potential', 'longitude', 'latitude']].dropna()

print(f"Final dataset size: {len(df_ml)} rows")
print(f"Features: {feature_columns}")

# ============================================================================
# 8. SPLIT DATA (TRAIN/TEST)
# ============================================================================

print("\n📊 Step 8: Splitting data into train/test sets...")

X = df_ml[feature_columns]
y = df_ml['fishing_potential']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 80% train, 20% test
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

print("\nTraining set distribution:")
print(y_train.value_counts())

# ============================================================================
# 9. NORMALIZE FEATURES
# ============================================================================

print("\n📏 Step 9: Normalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalized using StandardScaler")

# ============================================================================
# 10. SAVE PROCESSED DATA
# ============================================================================

print("\n💾 Step 10: Saving processed data...")

# Create processed data directory
os.makedirs('data/processed', exist_ok=True)

# Save processed datasets
np.save('data/processed/X_train.npy', X_train_scaled)
np.save('data/processed/X_test.npy', X_test_scaled)
np.save('data/processed/y_train.npy', y_train.values)
np.save('data/processed/y_test.npy', y_test.values)

# Save feature names
np.save('data/processed/feature_names.npy', feature_columns)

# Save scaler for later use
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

print("✓ Saved to data/processed/:")
print("  - X_train.npy, X_test.npy (features)")
print("  - y_train.npy, y_test.npy (labels)")
print("  - feature_names.npy")
print("✓ Saved scaler to models/scaler.pkl")

# ============================================================================
# 11. CREATE VISUALIZATIONS
# ============================================================================

print("\n📊 Step 11: Creating visualizations...")

# Create plots directory
os.makedirs('data/plots', exist_ok=True)

# Plot 1: Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(df_ml['sst_celsius'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('SST (°C)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Sea Surface Temperature Distribution')
axes[0, 0].axvline(df_ml['sst_celsius'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

axes[0, 1].hist(df_ml['current_speed_m_s'], bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Current Speed (m/s)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Ocean Current Speed Distribution')
axes[0, 1].axvline(df_ml['current_speed_m_s'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].legend()

axes[1, 0].scatter(df_ml['longitude'], df_ml['latitude'], 
                   c=df_ml['sst_celsius'], cmap='coolwarm', alpha=0.6, s=20)
axes[1, 0].set_xlabel('Longitude', fontweight='bold')
axes[1, 0].set_ylabel('Latitude', fontweight='bold')
axes[1, 0].set_title('SST Spatial Distribution')
plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='SST (°C)')

fishing_counts = df_ml['fishing_potential'].value_counts()
axes[1, 1].bar(fishing_counts.index, fishing_counts.values, 
               color=['red', 'yellow', 'green'], alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Fishing Potential', fontweight='bold')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Fishing Potential Distribution')

plt.tight_layout()
plt.savefig('data/plots/preprocessing_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/preprocessing_summary.png")

# Plot 2: Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df_ml[feature_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('data/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/correlation_matrix.png")

plt.close('all')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 70)

summary = f"""
📊 DATA SUMMARY:
   - Original rows: {len(df)}
   - After cleaning: {len(df_ml)}
   - Features: {len(feature_columns)}
   - Classes: {len(df_ml['fishing_potential'].unique())}

📈 CLASS DISTRIBUTION:
{df_ml['fishing_potential'].value_counts()}

🎯 READY FOR MACHINE LEARNING:
   ✓ Training set: {len(X_train)} samples
   ✓ Test set: {len(X_test)} samples
   ✓ Features normalized
   ✓ Data saved to data/processed/
   
📊 VISUALIZATIONS CREATED:
   ✓ preprocessing_summary.png
   ✓ correlation_matrix.png
"""

print(summary)
print("=" * 70)
print("\n🎯 Next step: Train Random Forest model!")
print("   Run: python scripts/train_model.py")
print("=" * 70)