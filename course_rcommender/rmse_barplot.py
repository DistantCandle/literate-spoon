import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Data (all RMSE values) ---
data = {
    'Model': [
        'KNN', 
        'NMF', 
        'RecommenderNet', 
        'Linear Regression (Embed)', 
        'Ridge Regression (Embed)', 
        'Random Forest (Embed)',
        'Logistic Regression (Classifier)', 
        'Decision Tree (Classifier)', 
        'SVM (Classifier)', 
        'Bagging (Classifier)', 
        'Gradient Boosting (Classifier)'
    ],
    'RMSE': [
        1.0205, 1.2998, 0.1505, 
        0.8138, 0.138, 0.8141,
        1.1783, 1.2666, 1.2901, 
        1.2214, 1.1531
    ],
    'Category': [
        'Recommender', 'Recommender', 'Recommender', 
        'Recommender', 'Recommender', 'Recommender',
        'Classifier', 'Classifier', 'Classifier', 
        'Classifier', 'Classifier'
    ]
}

# --- Create DataFrame ---
df = pd.DataFrame(data)

# Sort by RMSE ascending (best at top)
df = df.sort_values('RMSE', ascending=True)

# --- Plot Style ---
sns.set(style="whitegrid", palette="Set2", font_scale=1.1)

# --- Plot ---
plt.figure(figsize=(10, 7))
sns.barplot(
    data=df,
    y='Model',
    x='RMSE',
    hue='Category',
    palette="Set2",
    edgecolor='black'
)

plt.title("Model Performance Comparison (RMSE - lower is better)", fontsize=16, fontweight='bold')
plt.xlabel("RMSE")
plt.ylabel("Model")
plt.legend(title="Model Type")
plt.tight_layout()
plt.show()
