import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('train_set.csv')
labels_df = pd.read_csv('training_set_labels.csv')
test_df = pd.read_csv('test_set.csv')
#MERGE LABELS
train_df = train_df.merge(labels_df, on='respondent_id')
X = train_df.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y = train_df[['xyz_vaccine', 'seasonal_vaccine']]
X_test = test_df.drop(columns=['respondent_id'])
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
X_preprocessed = preprocessor.fit_transform(X)
X_test_preprocessed = preprocessor.transform(X_test)

# MODEL
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))

#TRAINING
model.fit(X_preprocessed, y)
def multi_label_roc_auc_score(y_true, y_pred, average="macro"):
    return roc_auc_score(y_true, y_pred, average=average)
scorer = make_scorer(multi_label_roc_auc_score, needs_proba=True)
cv_scores = cross_val_score(model, X_preprocessed, y, cv=5, scoring=scorer)
print(f'Cross-validated ROC AUC score: {np.mean(cv_scores)}')
print(f'Cross-validated ROC AUC score: {np.mean(cv_scores)}')
predictions = model.predict_proba(X_test_preprocessed)
xyz_vaccine_prob = predictions[0][:, 1]
seasonal_vaccine_prob = predictions[1][:, 1]

#FILE CREATION
submission_df = pd.DataFrame({
    'respondent_id': test_df['respondent_id'],
    'xyz_vaccine': xyz_vaccine_prob,
    'seasonal_vaccine': seasonal_vaccine_prob
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created.")