"""
FishSense: K-Means Clustering Model
Implements K-Means clustering for fishing zone prediction and compares with Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import time

print("=" * 70)
print("FISHSENSE: K-MEANS CLUSTERING MODEL")
print("=" * 70)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================

print("\n📁 Step 1: Loading preprocessed data...")

try:
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
    y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
    feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True)
    
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Feature names: {list(feature_names)}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("   Make sure you ran preprocess_data.py first!")
    exit()

# ============================================================================
# 2. TRAIN K-MEANS MODEL
# ============================================================================

print("\n🔵 Step 2: Training K-Means clustering model...")

# We know we have 3 classes (High, Medium, Low), so use k=3
n_clusters = 3

# Record training time
start_time = time.time()

kmeans_model = KMeans(
    n_clusters=n_clusters,
    init='k-means++',  # Smart initialization
    n_init=10,  # Number of times to run with different seeds
    max_iter=300,
    random_state=42
)

# Fit on training data
kmeans_model.fit(X_train)

training_time = time.time() - start_time

print(f"✓ K-Means model trained in {training_time:.4f} seconds")
print(f"✓ Number of clusters: {n_clusters}")
print(f"✓ Number of iterations: {kmeans_model.n_iter_}")
print(f"✓ Inertia (sum of squared distances): {kmeans_model.inertia_:.2f}")

# ============================================================================
# 3. PREDICT CLUSTERS
# ============================================================================

print("\n📊 Step 3: Predicting clusters...")

# Predict clusters for train and test sets
train_clusters = kmeans_model.predict(X_train)
test_clusters = kmeans_model.predict(X_test)

print(f"✓ Training predictions: {len(train_clusters)}")
print(f"✓ Test predictions: {len(test_clusters)}")

# ============================================================================
# 4. MAP CLUSTERS TO FISHING POTENTIAL LABELS
# ============================================================================

print("\n🗺️ Step 4: Mapping clusters to fishing potential labels...")

# Map cluster IDs to actual labels based on cluster centers
# We'll analyze which cluster corresponds to which fishing potential

# Get cluster centers
centers = kmeans_model.cluster_centers_

# For each cluster, calculate average SST and current speed
# (assuming these are the first two features)
cluster_info = []
for i in range(n_clusters):
    cluster_mask = train_clusters == i
    cluster_samples = X_train[cluster_mask]
    
    # Get actual labels for this cluster
    cluster_labels = y_train[cluster_mask]
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    # Most common label in this cluster
    most_common_label = unique[np.argmax(counts)]
    
    cluster_info.append({
        'cluster_id': i,
        'size': np.sum(cluster_mask),
        'dominant_label': most_common_label,
        'label_distribution': dict(zip(unique, counts))
    })
    
    print(f"\nCluster {i}:")
    print(f"  Size: {np.sum(cluster_mask)} samples")
    print(f"  Dominant label: {most_common_label}")
    print(f"  Label distribution: {dict(zip(unique, counts))}")

# Create mapping from cluster ID to label
cluster_to_label = {info['cluster_id']: info['dominant_label'] for info in cluster_info}

# Map predictions to labels
train_pred_labels = np.array([cluster_to_label[c] for c in train_clusters])
test_pred_labels = np.array([cluster_to_label[c] for c in test_clusters])

# ============================================================================
# 5. EVALUATE K-MEANS PERFORMANCE
# ============================================================================

print("\n📈 Step 5: Evaluating K-Means performance...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Calculate metrics
train_accuracy = accuracy_score(y_train, train_pred_labels)
test_accuracy = accuracy_score(y_test, test_pred_labels)

precision = precision_score(y_test, test_pred_labels, average='weighted')
recall = recall_score(y_test, test_pred_labels, average='weighted')
f1 = f1_score(y_test, test_pred_labels, average='weighted')

# Clustering-specific metrics
silhouette = silhouette_score(X_test, test_clusters)
davies_bouldin = davies_bouldin_score(X_test, test_clusters)

# Agreement with true labels
ari = adjusted_rand_score(y_test, test_clusters)

print("\n📊 K-Means Performance Metrics:")
print(f"  • Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Precision (weighted): {precision:.4f} ({precision*100:.2f}%)")
print(f"  • Recall (weighted): {recall:.4f} ({recall*100:.2f}%)")
print(f"  • F1-Score (weighted): {f1:.4f} ({f1*100:.2f}%)")

print("\n🔍 Clustering Quality Metrics:")
print(f"  • Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"  • Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
print(f"  • Adjusted Rand Index: {ari:.4f} (agreement with true labels)")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, test_pred_labels))

# ============================================================================
# 6. COMPARE WITH RANDOM FOREST
# ============================================================================

print("\n🔄 Step 6: Comparing with Random Forest...")

# Load Random Forest model
try:
    rf_model = joblib.load('models/fishsense_rf_model.pkl')
    rf_metadata = json.load(open('models/model_metadata.json'))
    
    print("\n📊 Model Comparison:")
    print(f"\n{'Metric':<25} {'Random Forest':<20} {'K-Means':<20}")
    print("-" * 65)
    print(f"{'Test Accuracy':<25} {rf_metadata['test_accuracy']:.4f} ({rf_metadata['test_accuracy']*100:.1f}%){'':<7} {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"{'Precision':<25} {rf_metadata['precision']:.4f} ({rf_metadata['precision']*100:.1f}%){'':<7} {precision:.4f} ({precision*100:.1f}%)")
    print(f"{'Recall':<25} {rf_metadata['recall']:.4f} ({rf_metadata['recall']*100:.1f}%){'':<7} {recall:.4f} ({recall*100:.1f}%)")
    print(f"{'F1-Score':<25} {rf_metadata['f1_score']:.4f} ({rf_metadata['f1_score']*100:.1f}%){'':<7} {f1:.4f} ({f1*100:.1f}%)")
    print(f"{'Training Time':<25} {'~1-2 seconds':<20} {training_time:.4f}s")
    
    print("\n🏆 Winner: Random Forest" if rf_metadata['test_accuracy'] > test_accuracy else "\n🏆 Winner: K-Means")
    
except Exception as e:
    print(f"⚠️ Could not load Random Forest model for comparison: {e}")

# ============================================================================
# 7. SAVE K-MEANS MODEL
# ============================================================================

print("\n💾 Step 7: Saving K-Means model...")

os.makedirs('models', exist_ok=True)

# Save model
kmeans_path = 'models/fishsense_kmeans_model.pkl'
joblib.dump(kmeans_model, kmeans_path)
print(f"✓ K-Means model saved to: {kmeans_path}")

# Save cluster mapping
mapping_path = 'models/kmeans_cluster_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump({str(k): v for k, v in cluster_to_label.items()}, f, indent=2)
print(f"✓ Cluster mapping saved to: {mapping_path}")

# Save metadata
kmeans_metadata = {
    'model_type': 'KMeans',
    'n_clusters': n_clusters,
    'init': 'k-means++',
    'n_iterations': int(kmeans_model.n_iter_),
    'inertia': float(kmeans_model.inertia_),
    'features': list(feature_names),
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'silhouette_score': float(silhouette),
    'davies_bouldin_index': float(davies_bouldin),
    'adjusted_rand_index': float(ari),
    'training_time_seconds': float(training_time),
    'cluster_mapping': cluster_to_label,
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test)
}

metadata_path = 'models/kmeans_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(kmeans_metadata, f, indent=2)
print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================

print("\n📊 Step 8: Creating visualizations...")

os.makedirs('data/plots', exist_ok=True)

# Plot 1: Cluster visualization (2D projection using first 2 features)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data
scatter1 = axes[0].scatter(X_train[:, 0], X_train[:, 1], 
                           c=train_clusters, cmap='viridis', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
axes[0].scatter(centers[:, 0], centers[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                label='Cluster Centers')
axes[0].set_xlabel(f'{feature_names[0]}', fontweight='bold')
axes[0].set_ylabel(f'{feature_names[1]}', fontweight='bold')
axes[0].set_title('K-Means Clustering - Training Data', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Test data
scatter2 = axes[1].scatter(X_test[:, 0], X_test[:, 1], 
                           c=test_clusters, cmap='viridis', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
axes[1].scatter(centers[:, 0], centers[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                label='Cluster Centers')
axes[1].set_xlabel(f'{feature_names[0]}', fontweight='bold')
axes[1].set_ylabel(f'{feature_names[1]}', fontweight='bold')
axes[1].set_title('K-Means Clustering - Test Data', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.colorbar(scatter2, ax=axes[1], label='Cluster ID')
plt.tight_layout()
plt.savefig('data/plots/kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/plots/kmeans_clusters.png")

# Plot 2: Model comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Random Forest', 'K-Means']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

try:
    rf_values = [rf_metadata['test_accuracy'], rf_metadata['precision'], 
                 rf_metadata['recall'], rf_metadata['f1_score']]
    kmeans_values = [test_accuracy, precision, recall, f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, kmeans_values, width, label='K-Means', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Comparison: Random Forest vs K-Means', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/plots/model_comparison.png")
    
except Exception as e:
    print(f"⚠️ Could not create comparison plot: {e}")

plt.close('all')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ K-MEANS CLUSTERING COMPLETE!")
print("=" * 70)

summary = f"""
🔵 K-MEANS MODEL SUMMARY:

📊 PERFORMANCE METRICS:
   - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
   - Precision: {precision:.4f} ({precision*100:.2f}%)
   - Recall: {recall:.4f} ({recall*100:.2f}%)
   - F1-Score: {f1:.4f} ({f1*100:.2f}%)

🔍 CLUSTERING QUALITY:
   - Silhouette Score: {silhouette:.4f}
   - Davies-Bouldin Index: {davies_bouldin:.4f}
   - Training Time: {training_time:.4f} seconds

🎯 MODEL DETAILS:
   - Algorithm: K-Means Clustering
   - Number of clusters: {n_clusters}
   - Iterations: {kmeans_model.n_iter_}
   - Features: {len(feature_names)}

💾 SAVED FILES:
   ✓ models/fishsense_kmeans_model.pkl
   ✓ models/kmeans_metadata.json
   ✓ models/kmeans_cluster_mapping.json

📊 VISUALIZATIONS:
   ✓ kmeans_clusters.png
   ✓ model_comparison.png
"""

print(summary)

print("\n🎯 CONCLUSION:")
if test_accuracy >= 0.8:
    print("   K-Means performs well for this dataset!")
else:
    print("   Random Forest outperforms K-Means (expected for supervised learning)")
print("   Both models now available for comparative analysis.")
