import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Financials.csv", delimiter=",")
df.columns = df.columns.str.strip()
df.describe()
df.info()
df.isnull().sum()
df.duplicated().sum()
def convert_to_f(s):
    s = s.strip().replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    if s == "-":
        return float(0)
    else:
        return float(s)
convert_to_f("$12")
money = df.columns.values.tolist()[4:12]
money
for col in money:
    df[col] = df[col].apply(convert_to_f)

df.head()
df["Date"] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Days_Since_Epoch'] = (df['Date'] - pd.Timestamp('2014-01-01')).dt.days
df.head()
df = df.drop(['Month Name', 'Date', ''], axis=1)
df.head()
yes = df.columns.to_list()[0:4]
yes
df_encoded = pd.get_dummies(df, columns=yes, drop_first=True)
df_encoded.head()
df_encoded.info()
plt.figure(figsize=(12, 10))  
matrix_corr = df_encoded.corr()
sns.heatmap(matrix_corr, cmap='coolwarm', fmt='.2f', cbar=True)
sns.pairplot(df_encoded[df_encoded.columns.to_list()[0:11]])
plt.figure(figsize=(12, 10))
plt.xticks(rotation=90)
sns.boxplot(df_encoded[df_encoded.columns.to_list()[0:11]])
num_cols = df_encoded.columns.to_list()[3:8]
num_cols
df_cleaned = df_encoded
Q1 = df_cleaned[num_cols].quantile(0.25)
Q3 = df_cleaned[num_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[~((df_cleaned[num_cols] < lower_bound) | (df_cleaned[num_cols] > upper_bound)).any(axis=1)]

plt.figure(figsize=(12, 10))
plt.xticks(rotation=90)
sns.boxplot(df_cleaned[df_cleaned.columns.to_list()[0:11]])
