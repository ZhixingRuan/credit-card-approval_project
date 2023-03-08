# prepare dataset for modelling

import pandas as pd
import numpy as np

df = pd.read_csv('./data/data_clean.csv')
df.dropna(inplace=True)

# Just learning from the users who have long than 3 months of records.
df_3m = df.loc[df.window >= 3]

features = list(df_3m.columns)
target = 'label'
features.remove('ID')
features.remove('window')
features.remove(target)
features

X = df_3m[features]
y = df_3m[target]
output = pd.concat([X, y], axis=1)
output.to_csv('./flask/model/model_trainset.csv', index=False)
print('Data output ---> ./flask/model/model_trainset.csv')
