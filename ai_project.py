import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#need to uncompress zip file
folderpath = "steam_dataset_2025_csv"

#list of all csv files
all_files = [
    os.path.join(folderpath, f)
    for f in os.listdir(folderpath)
    if f.endswith(".csv")
]

print("All the CSV files used for this project")
for f in all_files:
    print(f)

#applications is the main file for this project, need to merge other important datasets into it
main_file = os.path.join(folderpath, "applications.csv")
df = pd.read_csv(main_file, low_memory=False)
print("Dataset (rows, columns):", df.shape)



#merging reviews file
reviews_df = pd.read_csv(os.path.join(folderpath, "reviews.csv"), low_memory=False)
reviews_df['voted_up'] = reviews_df['voted_up'].astype(bool)
review_counts = reviews_df.groupby('appid')['voted_up'].agg(
    total_positive_reviews = 'sum',
    total_reviews = 'count'
).reset_index()

review_counts['total_negative_reviews'] = (
    review_counts['total_reviews'] - review_counts['total_positive_reviews']
)

df = df.merge(review_counts, on='appid', how='left')



print("Merging Reviews", review_counts.shape)
print("Dataset after:", df.shape)


#merging genres
genres_df = pd.read_csv(os.path.join(folderpath, "application_genres.csv"), low_memory=False)
genre_count = genres_df.groupby("appid").size().reset_index(name='genre_count')
df = df.merge(genre_count, on='appid', how='left') 

print("Merging Genres", genres_df.shape)

#merging categories
cat_df = pd.read_csv(os.path.join(folderpath, "application_categories.csv"), low_memory=False)
cat_count = cat_df.groupby("appid").size().reset_index(name='category_count')
df = df.merge(cat_count, on='appid', how='left')

print("Merging Categories", cat_df.shape)


#merging platforms
plat_df = pd.read_csv(os.path.join(folderpath, "application_platforms.csv"), low_memory=False)
plat_count = plat_df.groupby("appid").size().reset_index(name='platform_count')
df = df.merge(plat_count, on='appid', how='left')

print("Merging Platforms", plat_df.shape)

#all data so far
print("After all joins:", df.shape)
print(df.columns)


#cleaning
df_clean = df.copy()
init_shape = df_clean.shape
print("Cleaning up data")

#removing duplicates
duplicates = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print("Duplicates removed:", duplicates)


print("Before removal of points that are missing critical data:", df_clean.shape)
#remove points that are missing too much critical data
#these are my critical data points, if they dont have at least one of them, i dont need the data
df_clean = df_clean.dropna(
    subset=[
        'total_positive_reviews',
        'metacritic_score',
        'recommendations_total'
    ],
    how='all'
)
print("After:", df_clean.shape)

#fill missing for rest of data
for col in df_clean.columns:
    missing = df_clean[col].isnull().sum()
    if missing > 0:
        if df_clean[col].dtype == 'object':
            df_clean.fillna({col: 'No Data'}, inplace=True)
        elif df_clean[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            df_clean.fillna({col: 0}, inplace=True)

print("Missing values have been filled with'No Data'and 0")


df = df_clean
print("After cleaning")
print(df.shape)



#defining success
df['success'] = np.where(
    ((df['total_positive_reviews'] - df['total_negative_reviews']) > 100) |
    (df['metacritic_score'] >= 60) |
    (df['recommendations_total'] > 1000),
    1, 0
)

#plotting success with class distribution
sns.countplot(x=df['success'])
plt.title("Class Distribution")
plt.show()

#plot showing success with votes up
sns.histplot(df, x='total_positive_reviews', hue='success', bins=50)
plt.title("Positive Reviews Distribution")
plt.show()

#plot showing success with recomendations
sns.histplot(df, x='recommendations_total', hue='success', bins=50)
plt.title("Recomendations Distribution")
plt.show()

#plot showing success with metacritic score
sns.histplot(df, x='metacritic_score', hue='success', bins=50)
plt.title("Metacritic Score Distribution")
plt.show()

#feature engineering
#price brackets on how expensive a game is, 1000 is $10, 100000 is $100 etc
df['price_tier'] = pd.cut(df['mat_final_price'], bins=[-1, 0, 1000, 3000, 6000, 100000], labels=['Free', '$', '$$', '$$$', '$$$$'])
df['price_tier'] = df['price_tier'].astype('category').cat.codes

#if a game is past 2020 then it is considered a recent game
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['recent_game'] = (df['release_year'] >= 2020).astype(int)

#finding the ratio of positive and negative reviews 
df['positive_review_ratio'] = (
    df['total_positive_reviews'] /
    (df['total_positive_reviews'] + df['total_negative_reviews'] + 1)
)

#choosing features
features = [
    'price_tier',
    'is_free',
    'release_year',
    'recent_game',
    'required_age',
    'category_count',
    'platform_count',
    'positive_review_ratio'
]

print(features)

print(df.columns)

X = df[features]
X = X.apply(pd.to_numeric, errors='coerce')
y = df['success']

#80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify= y)

#FIRST MODEL WILL USE DECISION TREE
print("Model 1: Decision Tree")
dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [3, 10, 15, 25, 30],  #since dataset is very large, 100000 datapoints, we need a large tree
    'min_samples_split': [2, 5, 10, 50, 100]  #lots of data to split
}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1')
grid_dt.fit(X_train, y_train)
print("Chosen parameters:", grid_dt.best_params_)

results_dt = pd.DataFrame(grid_dt.cv_results_)
#plot from hypertuning from decision tree
plt.plot(results_dt['param_max_depth'], results_dt['mean_test_score'], marker='o')
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.title("DECISION TREE")
plt.show()

plt.plot(results_dt['param_min_samples_split'], results_dt['mean_test_score'], marker='o')
plt.xlabel("Min Samples Split")
plt.ylabel("F1 Score")
plt.title("DECISION TREE")
plt.show()

#RANDOM FOREST FOR SECOND MODEL
print("Model 2: Random Forest")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'max_depth':[15, 25, 30], #same as before, large dataset so need a large forest
    'n_estimators': [100, 200]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)
print("Chosen parameters:", grid_rf.best_params_)

results_rf = pd.DataFrame(grid_rf.cv_results_)

# plot with hyperparameters from random forest
plt.plot(results_rf['param_max_depth'], results_rf['mean_test_score'], marker = 'o')
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.title("RANDOM FOREST")
plt.show()

plt.plot(results_rf['param_n_estimators'], results_rf['mean_test_score'], marker = 'o')
plt.xlabel("Num Trees")
plt.ylabel("F1 Score")
plt.title("RANDOM FOREST")
plt.show()


#results of model 1
print("Stats from Decision Tree")
y_pred_dt = grid_dt.predict(X_test)
print(classification_report(y_test, y_pred_dt))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt)
plt.title("Decision Tree Confusion Matrix")
plt.show()

y_pred_dt = grid_dt.best_estimator_.predict(X_test)
metrics_dt = {
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'Precision': precision_score(y_test, y_pred_dt),
    'Recall': recall_score(y_test, y_pred_dt),
    'F1-score': f1_score(y_test, y_pred_dt)
}

#feature importance for decision tree
dt_best = grid_dt.best_estimator_

dt_importance = pd.DataFrame({
    'Feature': features,
    'Importance': dt_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nDecision Tree Feature Importance:")
print(dt_importance)



#results of model 2
print("Stats from Random Forest")
y_pred_rf=grid_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Random Forest Confusion Matrix")
plt.show()

y_pred_rf = grid_rf.best_estimator_.predict(X_test)

metrics_rf = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-score': f1_score(y_test, y_pred_rf)
}

#feature importance for random forest
rf_best = grid_rf.best_estimator_

rf_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

plt.figure()
plt.barh(rf_importance['Feature'], rf_importance['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

# both model comparison
metrics_df = pd.DataFrame({
    'Decision Tree': metrics_dt,
    'Random Forest': metrics_rf
})
metrics_df.plot(kind='bar')
plt.title("Decision Tree & Random Forest Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
