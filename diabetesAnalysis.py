
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/diabetes.csv")

def check_df(dataFrame, head=5):
    print("####### Shape #######")
    print(dataFrame.shape)
    print("####### Type ########")
    print(dataFrame.dtypes)
    print("####### Head ########")
    print(dataFrame.head(head))
    print("####### Tail ########")
    print(dataFrame.tail(head))
    print("####### NA ########")
    print(dataFrame.isnull().sum())
    print("####### Quantiles ########")
    print(dataFrame.describe().T)

check_df(df)

#Categorical & Numerical Feature Analysis

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in df:
    print(f'{col} : {df[col].nunique()}')


df[num_cols].plot(kind='box')
plt.xticks(rotation=30, horizontalalignment='right')

df[['Insulin']].plot(kind='box')
df[['DiabetesPedigreeFunction']].plot(kind='box')

df[df.columns[~df.columns.isin(['Insulin','Outcome'])]].plot(kind='box')
plt.xticks(rotation=30, horizontalalignment='right')

# Outliers Functions
def check_outlier(dataframe, col_name, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def replace_with_thresholds(dataframe, variable, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(f'{col} : {check_outlier(df, col, q1= 0.05, q3=0.95)}')

for col in num_cols:
    print(f'{col} : {check_outlier(df, col, q1= 0.1, q3=0.9)}')

print('Insulin : ', check_outlier(df, 'Insulin', q1=0.25, q3=0.75))

df.describe().T

for col in ['BloodPressure', 'DiabetesPedigreeFunction']:
    if check_outlier(df, col, q1=0.1, q3=0.9):
        replace_with_thresholds(df, col, q1=0.1, q3=0.9)

for col in ['Insulin']:
    if check_outlier(df, col, q1=0.1, q3=0.9):
        replace_with_thresholds(df, col, q1=0.1, q3=0.9)

df.describe().T

# Local Outlier Factor Analysis

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim= [0, 50], style='.-', use_index=True)
plt.show()

th = np.sort(df_scores)[8]

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores > th].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df.shape

# aykırı değerlerin bu kadar yüksek çıkması?

# Missing Values

df.isnull().values.any()

zero_columns = [col for col in df.columns if (df[col] == 0).sum() > 0]


def zero_values_table(dataframe, zero_name=False):
    zero_columns = [col for col in df.columns if (df[col] == 0).sum() > 0]
    n_zeros = (dataframe[zero_columns] == 0).sum()
    ratio = ((dataframe[zero_columns] == 0).sum() / dataframe.shape[0] * 100)
    zeros_df = pd.concat([n_zeros, np.round(ratio, 2)], axis=1, keys=['n_zeros', 'ratio'])

    print(zeros_df, end="\n")

    if zero_name:
        return zero_columns

    zero_values_table(df, zero_name=False)

# Imputation - Zero Values

unwanted = {'Pregnancies', 'Outcome'}

zero_columns = [e for e in zero_columns if e not in unwanted]

df[zero_columns] = df[zero_columns].replace({0: np.nan})

#standartlaştırma

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(df[zero_columns]), columns=zero_columns)
dff.head()

#eksik değerlerin knn algoritması ile doldurulması

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)


dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

for col in zero_columns:
    df[col] = dff[[col]]

# Correlation Analysis

df.corr()
sns.heatmap(df.corr(), annot=True)

# Target Variable Analysis

def target_vs_numeric(dataframe, target, num_cols):
    temp_df = dataframe.copy()
    for col in num_cols:
        print(pd.DataFrame({f"{col}_mean": temp_df.groupby(target)[col].mean(),
                            f"{col}_max": temp_df.groupby(target)[col].max(),
                            f"{col}_min": temp_df.groupby(target)[col].min(),
                            f"{col}_count": temp_df.groupby(target)[col].count()}), end="\n\n\n")

target_vs_numeric(df, "Outcome", num_cols)

#Does blood pressure have an impact on diabetes?

target_vs_numeric(df, "Outcome", num_cols)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_columns, rare_perc):
    temp_df = dataframe.copy()
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 999, temp_df[var])

    return temp_df

rare_analyser(df, "Outcome", ['Pregnancies'])

new_df = rare_encoder(df, ['Pregnancies'], 0.06)
rare_analyser(new_df, "Outcome", ['Pregnancies'])


#Rare values for pregnancies are going to be replaced with 7 and represent 7+ pregnancies
df['Pregnancies']=np.where((df['Pregnancies']>=7), 7, df['Pregnancies'])
df.groupby('Pregnancies').agg({'Outcome': 'mean'})

cat_cols, num_cols, cat_but_car = grab_col_names(df.iloc[:,0:8], cat_th=10, car_th=20)

df.Pregnancies = df.Pregnancies.astype('category')
df.dtypes

############
# Feature Engineering

df['PregnancyFlag'] = np.where(df['Pregnancies'].astype(int) > 0, 1, 0)
df.groupby("PregnancyFlag").agg({"Outcome": "mean"})

df['Pregnancy3Flag'] = np.where(df['Pregnancies'].astype(int) > 3, 1, 0)
df.groupby("Pregnancy3Flag").agg({"Outcome": "mean"})

#pairplot


sns.pairplot(data=df, vars=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction','Age'], hue='Outcome')

#Glucose and Insulin : positive corr

df['GluInsPair']=df['Glucose']*df['Insulin']
df.head()

df.groupby("Outcome").agg({"GluInsPair": "mean"})

#BMI and SkinThickness : positive corr

df['BmiSkinPair']=df['BMI']*df['SkinThickness']
df.head()

df.groupby("Outcome").agg({"BmiSkinPair": "mean"})

#Age Groups

df["AgeCat"] = pd.qcut(df['Age'], 4)
df.head()

df["AgeCat"].value_counts()
df.groupby("AgeCat").agg({"Outcome": "mean"})


#BMI Groups

df["BmiCat"] = pd.qcut(df['BMI'], 2)
df.head()
df["BmiCat"].value_counts()
df.groupby("BmiCat").agg({"Outcome": "mean"})

#BMI can be flagged as above 32

df.drop('BmiCat', axis=1, inplace=True)
df['HighBmiFlag'] = np.where(df['BMI']> 32, 1, 0)
df.head()

# Encoding

df.dtypes

#OHE Encoding For Categorical Features
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first= True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
df.head()

df.dtypes

def bool_to_int(dataframe):
    bool_cols = dataframe.select_dtypes(include='bool').columns
    for col in bool_cols:
        dataframe[col] = dataframe[col].astype(int)
    return dataframe

df = bool_to_int(df)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

#Model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show();
    if save:
        plt.savefig('importances.png')

rf_model = RandomForestClassifier(random_state=555).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


plot_importance(rf_model, X_train)