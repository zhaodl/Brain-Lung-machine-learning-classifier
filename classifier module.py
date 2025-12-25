import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, cohen_kappa_score)
import matplotlib.pyplot as plt
import joblib

REGISTERED_FIGURES = []

FIG="results/FIG5"
out_dir = "results/feature_importance"
os.makedirs(out_dir, exist_ok=True)
out_dir = "results/feature_importance"
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.family"] = "Arial"

print("=" * 80)
print("Small Sample Classification - PCA + Multiple Models + Training Accuracy")
print("=" * 80)

# ================================
# 1. Data Loading and Preprocessing
# ================================
print("\n[Step 1] Data Loading and Preprocessing...")

# Read data, skip first row comment
df = pd.read_csv('brain.csv', skiprows=0)

# View basic data information
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

# Separate features and target
X = df.iloc[:, 1:]  # Features start from 2nd column (skip classification target)
y = df.iloc[:, 0]   # First column is classification target

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# Convert all features to numeric type
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values (fill with median)
if X.isnull().sum().sum() > 0:
    print(f"Handling {X.isnull().sum().sum()} missing values...")
    X = X.fillna(X.median())

# Remove constant columns (variance = 0)
constant_cols = X.columns[X.std() == 0]
if len(constant_cols) > 0:
    print(f"Removing {len(constant_cols)} constant columns")
    X = X.drop(columns=constant_cols)

print(f"Number of features after preprocessing: {X.shape[1]}")

# ================================
# 2. Feature Standardization
# ================================
print("\n[Step 2] Feature Standardization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

print(f"  Standardization completed: {X_scaled.shape}")
# ================================
# 3. PCA Dimensionality Reduction
# ================================
print("\n[Step 3] PCA Dimensionality Reduction Analysis...")

# First check variance explained ratio for different number of components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot variance explained ratio
plt.figure(figsize=(5, 5))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Variance Explained', fontsize=12)
plt.title('PCA Cumulative Variance Explained', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Threshold')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG}/pca_individual_variance.png', dpi=300, bbox_inches='tight')
print("Cumulative variance plot saved")
# plt.show()
plt.close()

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

print("\nPCA cumulative explained variance:")
for i, v in enumerate(cum_var, start=1):
    print(f"PC{i:02d}: cumulative variance = {v:.4f} ({v*100:.2f}%)")

plt.figure(figsize=(5, 5))
plt.bar(range(1, min(16, len(pca_full.explained_variance_ratio_) + 1)),
        pca_full.explained_variance_ratio_[:15], alpha=0.7, color='steelblue')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Variance Explained Ratio', fontsize=12)
plt.title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{FIG}/Variance_Explained_by_Each_Component.png', dpi=300, bbox_inches='tight')
print("ndividual variance plot saved")
plt.tight_layout()
# plt.show()
plt.close()

print("\nExplained variance ratio by each principal component:")
for i, v in enumerate(pca_full.explained_variance_ratio_, start=1):
    print(f"PC{i:02d}: variance explained = {v:.4f} ({v*100:.2f}%)")


# Find number of components to reach 95% variance
n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
n_components_90 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1

print(f"Components needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")
print(f"Cumulative variance of first 5 components: {np.sum(pca_full.explained_variance_ratio_[:5]):.3f}")
print(f"Cumulative variance of first 10 components: {np.sum(pca_full.explained_variance_ratio_[:10]):.3f}")

# Try multiple PCA dimensions (including very low dimensions for small sample)
pca_dimensions = [2, 3, 4, 5, n_components_90]
pca_dimensions = sorted(list(set([d for d in pca_dimensions if d <= X_scaled.shape[1] and d < X_scaled.shape[0]])))

print(f"\nWill test the following PCA dimensions: {pca_dimensions}")

# ================================
# 4. Model Training and Evaluation (Training Accuracy)
# ================================
print("\n[Step 4] Model Training and Evaluation (using Training Accuracy)...")

results = []
best_models = {}
top5_models = []
for n_comp in pca_dimensions:
    print(f"\n{'=' * 60}")
    print(f"Testing PCA Dimension: {n_comp}")
    print(f"{'=' * 60}")
    
    # PCA dimensionality reduction
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    var_explained = np.sum(pca.explained_variance_ratio_)
    print(f"Cumulative variance explained: {var_explained:.4f}")
    
    # Define all models to test with optimized parameter ranges for small samples
    models_config = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(random_state=42, multi_class='ovr'),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [5000, 10000],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        {
            'name': 'SVM-RBF',
            'model': SVC(random_state=42, kernel='rbf'),
            'params': {
                'C': [0.1, 1, 10, 100, 1000, 10000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100]
            }
        },
        {
            'name': 'SVM-Linear',
            'model': SVC(random_state=42, kernel='linear'),
            'params': {
                'C': [0.1, 1, 10, 100, 1000, 10000]
            }
        },
        {
            'name': 'SVM-Poly',
            'model': SVC(random_state=42, kernel='poly'),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4]
            }
        },
        {
            'name': 'KNN',
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [2, 3, 4, 5, 6, 7, 8, None],
                'min_samples_split': [2, 3, 4, 5],
                'min_samples_leaf': [1, 2, 3, 4],
                'criterion': ['gini', 'entropy']
            }
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'subsample': [0.8, 1.0]
            }
        },
        {
            'name': 'AdaBoost',
            'model': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
            'params': {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0]
            }
        },
        {
            'name': 'LDA',
            'model': LinearDiscriminantAnalysis(),
            'params': {
                'solver': ['svd', 'lsqr', 'eigen']
            }
        },
        {
            'name': 'QDA',
            'model': QuadraticDiscriminantAnalysis(),
            'params': {
                'reg_param': [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        {
            'name': 'Naive Bayes',
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
            }
        },
        {
            'name': 'Ridge Classifier',
            'model': RidgeClassifier(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                'max_iter': [2000, 5000, 10000]
            }
        },
        {
            'name': 'Passive Aggressive',
            'model': PassiveAggressiveClassifier(random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'max_iter': [1000, 2000, 5000],
                'loss': ['hinge', 'squared_hinge']
            }
        },
    ]
    
    # Train and evaluate all models
    for idx, config in enumerate(models_config, 1):
        print(f"\n[4.{idx}] {config['name']}...")
        
        try:
            # Grid search for hyperparameters (using 5-fold CV for parameter selection)
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            grid = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=kfold, 
                scoring='accuracy', 
                n_jobs=-1,
                error_score='raise'
            )
            grid.fit(X_pca, y)
            
            best_model = grid.best_estimator_
            # Use training set accuracy instead of LOOCV
            pred = best_model.predict(X_pca)
            acc = accuracy_score(y, pred)
            
            print(f"  Best parameters: {grid.best_params_}")
            print(f"  Training accuracy: {acc:.4f}")
            
            # Calculate other metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, pred, average='weighted', zero_division=0
            )
            kappa = cohen_kappa_score(y, pred)
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  Kappa coefficient: {kappa:.4f}")
            
            results.append({
                'Model': config['name'],
                'PCA_Components': n_comp,
                'Variance_Explained': var_explained,
                'Training_Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Kappa': kappa,
                'Best_Params': str(grid.best_params_)
            })

            top5_models.append({
                "model": config["name"],
                "model_key": f"{config['name']}_PCA{n_comp}",
                "pca": pca,
                "scaler": scaler,
                #"best_model": best_model,
                "predictions": pred,
                "accuracy": acc
            })

            top5_models= sorted(
                top5_models,
                key=lambda x: x["accuracy"],
                reverse=True
            )[:5]
            
        except Exception as e:
            print(f"  Warning: {config['name']} failed with error: {str(e)}")
            continue

# ================================
# 5. Results Save
# ================================
print("\n" + "=" * 80)
print("[Step 5] Results Save")
print("=" * 80)

top5_models_df= pd.DataFrame(top5_models)
top5_models_df = top5_models_df.sort_values('accuracy', ascending=False)

print("\nComplete Results Table:")
print(top5_models_df.to_string(index=False))

joblib.dump(top5_models,"results/top5_models.pkl")

top5_models_df.to_csv('results/classification_top5_models.csv', index=False, encoding='utf-8-sig')
print("\n Saved results: classification_top5_models.csv")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Training_Accuracy', ascending=False)

print("\nComplete Results Table:")
print(results_df.to_string(index=False))

joblib.dump(results,"results/results.pkl")

results_df.to_csv('results/classification_results.csv', index=False, encoding='utf-8-sig')
print("\n Saved results: classification_results.csv")

print("\n" + "=" * 80)
print(" Classification Task Completed!")
print("\n" + "=" * 80)