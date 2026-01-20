"""
FishSense: Random Forest Model Training
Trains a Random Forest classifier to predict fishing zones
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=" * 70)
print("FISHSENSE - RANDOM FOREST TRAINING")
print("=" * 70)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

print("\n📁 Step 1: Loading processed data...")

try:
    X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
    X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
    y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
    y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
    feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Features: {list(feature_names)}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("   Make sure you ran preprocess_data.py first!")
    exit()

# ============================================================================
# 2. TRAIN RANDOM FOREST MODEL
# ============================================================================

print("\n🌲 Step 2: Training Random Forest model...")

# Initialize Random Forest with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Maximum depth of trees
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in a leaf
    random_state=42,         # For reproducibility
    n_jobs=-1,               # Use all CPU cores
    verbose=1                # Show progress
)

print("\nModel parameters:")
print(f"  - Number of trees: 100")
print(f"  - Max depth: 10")
print(f"  - Min samples split: 5")
print(f"  - Min samples leaf: 2")

print("\nTraining in progress...")
rf_model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# ============================================================================
# 3. EVALUATE MODEL ON TRAINING SET
# ============================================================================

print("\n📊 Step 3: Evaluating on training set...")

y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# ============================================================================
# 4. EVALUATE MODEL ON TEST SET
# ============================================================================

print("\n📊 Step 4: Evaluating on test set...")

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

print("\n🎯 Step 5: Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ============================================================================
# 6. SAVE MODEL
# ============================================================================

print("\n💾 Step 6: Saving model...")

# Save the trained model
model_path = 'models/fishsense_rf_model.pkl'
joblib.dump(rf_model, model_path)
print(f"✓ Model saved to: {model_path}")

# Save model metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100,
    'max_depth': 10,
    'features': list(feature_names),
    'classes': list(rf_model.classes_),
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test)
}

import json
metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================

print("\n📊 Step 7: Creating visualizations...")

os.makedirs('data/plots', exist_ok=True)

# Plot 1: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Random Forest', fontweight='bold', fontsize=14)
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('data/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/confusion_matrix.png")

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
plt.barh(feature_importance['feature'], feature_importance['importance'], 
         color=colors, edgecolor='black', linewidth=1)
plt.xlabel('Importance', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')
plt.title('Feature Importance - Random Forest', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('data/plots/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/feature_importance.png")

# Plot 3: Model Performance Comparison
plt.figure(figsize=(8, 6))
metrics = ['Train Accuracy', 'Test Accuracy']
values = [train_accuracy * 100, test_accuracy * 100]
colors_bar = ['#2ecc71', '#3498db']

bars = plt.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy (%)', fontweight='bold')
plt.title('Model Performance', fontweight='bold', fontsize=14)
plt.ylim([0, 100])

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('data/plots/model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/model_performance.png")

plt.close('all')

# ============================================================================
# 8. TEST PREDICTION FUNCTION
# ============================================================================

print("\n🧪 Step 8: Testing prediction function...")

# Test with a sample
print("\nTesting with sample data point:")
sample_features = X_test[0:1]  # Take first test sample
sample_true = y_test[0]
sample_pred = rf_model.predict(sample_features)[0]
sample_proba = rf_model.predict_proba(sample_features)[0]

print(f"True label: {sample_true}")
print(f"Predicted label: {sample_pred}")
print(f"Prediction probabilities:")
for class_label, prob in zip(rf_model.classes_, sample_proba):
    print(f"  {class_label}: {prob:.4f} ({prob*100:.2f}%)")

if sample_pred == sample_true:
    print("✓ Prediction correct!")
else:
    print("✗ Prediction incorrect (this is normal - no model is 100% accurate)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)

summary = f"""
🌲 RANDOM FOREST MODEL SUMMARY:

📊 PERFORMANCE METRICS:
   - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
   
🎯 MODEL DETAILS:
   - Algorithm: Random Forest Classifier
   - Number of trees: 100
   - Max depth: 10
   - Features: {len(feature_names)}
   - Classes: {len(rf_model.classes_)} (High, Medium, Low)
   
💾 SAVED FILES:
   ✓ models/fishsense_rf_model.pkl (trained model)
   ✓ models/model_metadata.json (model info)
   
📊 VISUALIZATIONS:
   ✓ confusion_matrix.png
   ✓ feature_importance.png
   ✓ model_performance.png

🎯 TOP 3 MOST IMPORTANT FEATURES:
"""

print(summary)
for idx, row in feature_importance.head(3).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 70)
print("🎯 Next step: Create Streamlit dashboard!")
print("=" * 70)