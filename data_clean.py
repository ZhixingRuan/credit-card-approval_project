import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import util

# load data
df1 = pd.read_csv('./data/application_record.csv')
df2 = pd.read_csv('./data/credit_record.csv')

# clean flag type data
# 'car', 'property', 'phone', 'mobile', 'work_phone', 'email'
ordinal = OrdinalEncoder(categories=[['N', 'Y']])
df1['car'] = ordinal.fit_transform(df1[['FLAG_OWN_CAR']])
df1['property'] = ordinal.fit_transform(df1[['FLAG_OWN_REALTY']])
df1 = df1.rename(columns={'FLAG_MOBIL': 'mobile',
                    'FLAG_WORK_PHONE': 'work_phone',
                    'FLAG_PHONE': 'phone',
                    'FLAG_EMAIL': 'email'})

# clean categorical type data
# 'gender', 'education', 'income', 'housing', 'occupation', 'marital_status'
df1 = pd.concat([df1, util.Dummy_Transformer('gender_').fit_transform(df1['CODE_GENDER'])], axis=1)
df1 = df1.rename(columns={'gender_F': 'gender'})

education_ordinal = OrdinalEncoder(
    categories=[['Lower secondary', 'Secondary / secondary special', 'Incomplete higher',
                 'Higher education', 'Academic degree']])
df1['education'] = education_ordinal.fit_transform(df1[['NAME_EDUCATION_TYPE']])

df1 = pd.concat([df1, util.Dummy_Transformer('income_').fit_transform(df1['NAME_INCOME_TYPE'])], axis=1)

df1 = pd.concat([df1, util.Dummy_Transformer('housing_').fit_transform(df1['NAME_HOUSING_TYPE'])], axis=1)

df1['OCCUPATION_TYPE'].fillna('others', inplace=True)
df1 = pd.concat([df1, util.Dummy_Transformer('occupation_').fit_transform(df1['OCCUPATION_TYPE'])], axis=1)

df1 = pd.concat([df1, util.Dummy_Transformer('marital_').fit_transform(df1['NAME_FAMILY_STATUS'])], axis=1)

# clean numerical type data
# 'num_children', 'annual_income', 'birthday', 'days_employment', 'family_size'
df1 = df1.rename(columns={'CNT_CHILDREN': 'num_children', 'AMT_INCOME_TOTAL': 'annual_income',
                          'DAYS_BIRTH': 'birthday', 'DAYS_EMPLOYED': 'days_employment',
                          'CNT_FAM_MEMBERS': 'family_size'})
df1.loc[df1.days_employment > 0, 'days_employment'] = 0


# all features for modelling
# drop 'num_children' due to collinearity
# drop 'mobile' due to value problem
numeric_features = ['annual_income', 'birthday', 'days_employment', 'family_size']
cat_ordinal_features = ['work_phone', 'phone', 'email', 'car', 'property', 'education']
cat_dummy_features = ['gender',
                      'income_Student', 'income_Pensioner', 'income_State servant',
                      'income_Commercial associate', 'income_Working',
                      'housing_Municipal apartment', 'housing_With parents',
                      'housing_Office apartment', 'housing_Rented apartment',
                      'housing_House / apartment', 'housing_Co-op apartment',
                      'occupation_Cleaning staff', 'occupation_others',
                      'occupation_Security staff', 'occupation_Medicine staff',
                      'occupation_Managers', 'occupation_Sales staff',
                      'occupation_Cooking staff', 'occupation_Core staff',
                      'occupation_Laborers', 'occupation_Low-skill Laborers',
                      'occupation_Waiters/barmen staff', 'occupation_IT staff',
                      'occupation_Accountants', 'occupation_Secretaries',
                      'occupation_High skill tech staff', 'occupation_Drivers',
                      'occupation_Realty agents', 'occupation_Private service staff',
                      'marital_Single / not married',
                      'marital_Married', 'marital_Civil marriage', 'marital_Separated', 'marital_Widow']

features = numeric_features + cat_ordinal_features + cat_dummy_features

# define target
# Choose users who overdue for more than 60 days as the risk users -- Target, labeled as 1.
df2['status_label'] = None
df2.loc[df2.STATUS=='2', ['status_label']] = 1
df2.loc[df2.STATUS=='3', ['status_label']] = 1
df2.loc[df2.STATUS=='4', ['status_label']] = 1
df2.loc[df2.STATUS=='5', ['status_label']] = 1

df2_label = df2.groupby('ID').count()
df2_label['ID'] = df2_label.index
df2_label['window'] = df2.groupby('ID').MONTHS_BALANCE.max() - df2.groupby('ID').MONTHS_BALANCE.min()
df2_label.reset_index(drop=True, inplace=True)
df2_label.loc[df2_label.status_label>0, ['label']] = 1
df2_label.loc[df2_label.status_label==0, ['label']] = 0

df = pd.merge(df1, df2_label, how='left', on='ID')
df = df[features + ['label'] + ['ID'] + ['window']]
df.to_csv('./data/data_clean.csv', index=False)
print('Data cleaned ---> output directory ./data/data_clean.csv')
