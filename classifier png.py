
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, cohen_kappa_score)
import matplotlib.pyplot as plt
import os
import joblib
from scipy import stats

FIG="results/FIG"
out_dir = "results/feature_importance"
os.makedirs(out_dir, exist_ok=True)
out_dir = "results/feature_importance"
os.makedirs(FIG, exist_ok=True)

warnings.filterwarnings('ignore')

print("=" * 80)
print("Small Sample Classification - PCA + Multiple Models + Training Accuracy")
print("=" * 80)
plt.rcParams["font.family"] = "Arial"


# 1.Data Loading and Preprocessing
print("\n[Step 1] Data Loading and Preprocessing...")

# Read data, skip first row comment


top5_models = joblib.load("results/top5_models.pkl")

results = joblib.load("results/results.pkl")

df = pd.read_csv('brain.csv', skiprows=0)
results_df = pd.read_csv("results/classification_results.csv")
top5_models_df = pd.read_csv("results/classification_top5_models.csv")

# Separate features and target
X = df.iloc[:, 1:]  # Features start from 2nd column (skip classification target)
y = df.iloc[:, 0]   # First column is classification target

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

REGISTERED_FIGURES = []
best_row = top5_models[0]
best_preds = best_row['predictions']
best_model = best_row['model']
best_pca   = best_row['pca']
best_acc   = best_row['accuracy']

# Find best model
best_result = results_df.iloc[0]
print("\n" + "=" * 80)
print(" Best Model")
print("=" * 80)
print(f"Model: {best_result['Model']}")
print(f"PCA Dimensions: {int(best_result['PCA_Components'])}")
print(f"Cumulative variance explained: {best_result['Variance_Explained']:.4f}")
print(f"Training Accuracy: {best_result['Training_Accuracy']:.4f}")
print(f"Precision: {best_result['Precision']:.4f}")
print(f"Recall: {best_result['Recall']:.4f}")
print(f"F1-score: {best_result['F1_Score']:.4f}")
print(f"Kappa coefficient: {best_result['Kappa']:.4f}")
print(f"Best parameters: {best_result['Best_Params']}")

# 2.Feature Standardization
print("\n[Step 2] Feature Standardization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

# 3.Visualization
print("\n[Step 3] Generating visualizations...")

print("\n" + "=" * 80)
print("Top4 Models Confusion Matrices (Prediction Results)")
print("=" * 80)

for idx, item in enumerate(top5_models, start=1):
    preds = item['predictions']
    acc = item['accuracy']
    model_key = item['model_key']
    model_name = item["model"][:3]
    if model_name == 'SVM':
        model_name = item['model'][:5]
    pca = item["pca"]


    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)'],
        yticklabels=['Normal(0)', 'Mild(1)', 'Moderate(2)', 'Severe(3)'],
        cbar_kws={'label': 'Sample Count'},
        annot_kws={'size': 16}
    )
    plt.title(
        f'{model_name}_PAC{pca.n_components_}, Accuracy={acc:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(
        f'{FIG}/confusion_matrix_{model_name}_PAC{pca.n_components_}.png',
        dpi=600,
        bbox_inches='tight'
    )
    plt.close()

    print(f" Saved Global Top {idx} Confusion Matrix")
    
print("\n" + "=" * 80)
print("Performance heatmap for all models")
print("=" * 80)

metrics = ['Training_Accuracy', 'Precision', 'Recall', 'F1_Score']

temp_list = []
for _, row in results_df.iterrows():
    model_s= row['Model'][:3]
    if model_s == 'SVM':
        model_s = row['Model'][:5]

    label = f"{model_s}_PCA{int(row['PCA_Components'])}"
    data_row = [row[m] for m in metrics]

    temp_list.append({
        "label": label,
        "model": row["Model"],
        "pca": row["PCA_Components"],
        "acc": row["Training_Accuracy"], 
        "data": data_row
    })

sorted_list = sorted(temp_list,key=lambda x: (-x["acc"], x["model"], x["pca"]))

seen = set()
unique_list = []
for item in sorted_list:
    if item["label"] not in seen:
        unique_list.append(item)
        seen.add(item["label"])

labels = [item["label"] for item in unique_list]
heatmap_data = [item["data"] for item in unique_list]

plt.figure(figsize=(5, 15))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=['Accuracy', 'Precision', 'Recall', 'F1-score'],
            yticklabels=labels, vmin=0, vmax=1,
            linecolor='white',
            linewidths= 0.3 ,
            annot_kws={'size': 10 },
            cbar_kws={'label': 'Score'})
plt.title('All Models Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIG}/all_models_heatmap.png', dpi=600, bbox_inches='tight')
print(" Saved figure: all_models_heatmap.png")
# plt.show()
plt.close()

REGISTERED_FIGURES = []
figs = REGISTERED_FIGURES[:4]

print("\n" + "=" * 80)
print("Top4 Models Feature importance  (Prediction Results)")
print("=" * 80)

top_n = 4
for m_idx, item in enumerate(top5_models, start=1):

    model_name = item["model"][:3]
    if model_name == 'SVM':
        model_name = item['model'][:5]
    
    acc = item["accuracy"]
    labels = item["predictions"]
    pca = item["pca"]
    scaler = item["scaler"]

    print("\n" + "=" * 80)
    print(f"[Feature analysis] Top model {m_idx}")
    print(f"Model: {model_name}")
    print(f"Training accuracy: {acc:.4f}")
    print(f"PCA components: {pca.n_components_}")
    print("=" * 80)


    # Compute feature differences between two clusters
    cluster_0_data = X_scaled[labels == 0]
    cluster_1_data = X_scaled[labels == 1]
    cluster_2_data = X_scaled[labels == 2]
    cluster_3_data = X_scaled[labels == 3]

    feature_stats = []
    for i, fname in enumerate(feature_names):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
        plt.rcParams['axes.unicode_minus'] = False
        mean_0 = cluster_0_data[:, i].mean()
        mean_1 = cluster_1_data[:, i].mean()
        mean_2 = cluster_2_data[:, i].mean()
        mean_3 = cluster_3_data[:, i].mean()

        std_0 = cluster_0_data[:, i].std()
        std_1 = cluster_1_data[:, i].std()
        std_2 = cluster_2_data[:, i].std()
        std_3= cluster_3_data[:, i].std()

        all_data = X_scaled[:, i]
        overall_mean = all_data.mean()

        between_var = 0.0
        within_var = 0.0

        for cls_data in [cluster_0_data, cluster_1_data, cluster_2_data, cluster_3_data]:
            n_c = cls_data.shape[0]
            mean_c = cls_data[:, i].mean()
            var_c = cls_data[:, i].var()
            
            between_var += n_c * (mean_c - overall_mean) ** 2
            within_var += n_c * var_c

        eps = 1e-12
        fisher_score = between_var / (within_var + eps)

        f_stat, p_value = stats.f_oneway(cluster_0_data[:, i],cluster_1_data[:, i],cluster_2_data[:, i],cluster_3_data[:, i])

        feature_stats.append({
            'Feature name': fname,
            'Class 0 mean': mean_0,
            'Class 1 mean': mean_1,
            'Class 0 standard deviation': std_0,
            'Class 1 standard deviation': std_1,
            'Class 2 mean': mean_2,
            'Class 3 mean': mean_3,
            'Class 2 standard deviation': std_2,
            'Class 3 standard deviation': std_3,
            'Mean difference': abs(mean_0 - mean_1),
            'f statistic': abs(f_stat),
            'fisher_score' : fisher_score,
            'p value': p_value,
            'Significance': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        })

    feature_importance_df = pd.DataFrame(feature_stats).sort_values('fisher_score', ascending=False)

    # Save feature importance
    feature_importance_df.to_csv('kmeans_feature_importance.csv', index=False, encoding='utf-8-sig')
    print("Saved: kmeans_feature_importance.csv")

    # Print features
    print("\n  important features (to distinguish two classes):")
    for idx, row in feature_importance_df.head(9).iterrows():
        print(f"    {row['Feature name']:20s}: difference={row['fisher_score']:.4f}, p={row['p value']:.4e} {row['Significance']}")
    
    X_pca = pca.transform(X_scaled)
    X_recon_scaled = pca.inverse_transform(X_pca)
    X_recon = scaler.inverse_transform(X_recon_scaled)
    top_df = feature_importance_df.head(top_n)

    plt.figure(figsize=(7.5, 5))

    y_pos = np.arange(len(top_df))
    bars = plt.barh(y_pos, top_df["fisher_score"], alpha=0.7)

    colors = []
    for sig in top_df["Significance"]:
        if sig == "***":
            colors.append("#E74C3C")
        elif sig == "**":
            colors.append("#F39C12")
        elif sig == "*":
            colors.append("#F1C40F")
        else:
            colors.append("#95A5A6")

    for bar, c in zip(bars, colors):
        bar.set_color(c)

    plt.yticks(y_pos, top_df["Feature name"], fontsize=12)
    plt.xlabel("Fisher score", fontsize=12)
    plt.title(f"{model_name}_PAC{pca.n_components_}, Feature importance", fontsize=14, fontweight="bold",pad=15)
    plt.gca().invert_yaxis()  
    plt.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{FIG}/feature_importance_{model_name}_{pca.n_components_}.png", dpi=600)


print("\n" + "=" * 80)
print(" Visualization Task Completed!")
print(f"  - Best model: {best_result['Model']}, Training Accuracy: {best_result['Training_Accuracy']:.2%}")
print(f"  - Using {int(best_result['PCA_Components'])} principal components, explaining {best_result['Variance_Explained']:.2%} of variance")

