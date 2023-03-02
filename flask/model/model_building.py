import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv('model_trainset.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # 1: 'bad' client, 0: 'good' client

numeric_features = ['annual_income', 'birthday', 'days_employment', 'family_size']
scaler = ColumnTransformer(
    transformers=[('num_scaler', StandardScaler(), numeric_features)],
    remainder='passthrough'
)

# define pipeline
steps = [('over', SMOTE()), 
         ('scaler', scaler),
         ('model', XGBClassifier(objective='binary:logistic', 
                                 learning_rate=0.5, max_depth=8,
                                 n_estimators=250, min_child_weight=4))]

model = Pipeline(steps).fit(X, y)


with open('../flaskapi/model.pkl', 'wb') as f:
    pickle.dump(model, f)

