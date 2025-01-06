# %%
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()
# %%
df = pd.read_csv(cwd+'/fis.csv')

# %%
# Sepsis is the target variable, so we don't need sirs and qsofa variables.
df = df.drop(columns=['sirs', 'qsofa'])

# %%
# Drop rows with missing age
df = df.dropna(subset=['age'])


# %%
na_counts = df.groupby('patient_id')['heart_rate'].apply(lambda x: x.isna().sum())
filtered_patient_ids = na_counts[na_counts == 1].index
filtered_df = df[df['patient_id'].isin(filtered_patient_ids)]

# %%
# fill missing heart rate values with the before value or after value for patient_id is im filtered_df.patient_id.unique().tolist()

for patient_id in filtered_df.patient_id.unique().tolist():
    patient_df = df[df['patient_id'] == patient_id]
    patient_df.loc[:, 'heart_rate'] = patient_df['heart_rate'].ffill().bfill()
    df.loc[df['patient_id'] == patient_id, :] = patient_df


# %%
# Fill missing heart rate values with the previous non-missing value
df['heart_rate'] = df.groupby('patient_id')['heart_rate'].ffill().bfill()

df = df.drop(columns=['fio2'])
df = df.drop(columns=['bilirubin'])

# %%
df.shape

# %%
# Correlation between age and sepsis_icd
df['age'].corr(df['sepsis_icd'])


# %%
# if temp is greater than 60 or less than 30, change nan
df.loc[df.temp > 60, 'temp'] = np.nan
df.loc[df.temp < 30, 'temp'] = np.nan

# %%
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

# %%
df.temp = imputer.fit_transform(df[['temp']])

# %%
df.gcs = imputer.fit_transform(df[['gcs']])

# %%
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

# %%
filling_columns = ['bp_systolic', 'bp_diastolic', 'map', 'resp', 'spo2']


# %%
for column in filling_columns:
    df[column] = imputer.fit_transform(df[[column]])

# %%
high_missing_columns = ['lactate', 'ph', 'pco2', 'po2']

# %%
df = df.drop(columns=['lactate', 'pco2'])

# %%
df.ph = df.ph.ffill().bfill()
df.po2 = df.po2.ffill().bfill()

# %%
missing_columns = ['wbc', 'bun', 'creatinine', 'platelets',  'bicarbonate', 'hemoglobin', 'hematocrit', 'potassium', 'chloride']


# %%
for column in missing_columns:
    df[column] = df[column].ffill().bfill()


columns_to_normalize = df.drop(columns=['patient_id']).columns


# %%
df.loc[df.resp > 60, 'resp'] = np.nan

# %%
df.resp = imputer.fit_transform(df[['resp']])


# %%
df = df.drop(columns=['bp_diastolic',
                      'po2',
                      'hemoglobin',
                      'hematocrit',
                      'potassium',
                      'gcs',
                      'age'])


# %%
df.to_csv('fis_cleaned.csv', index=False)


