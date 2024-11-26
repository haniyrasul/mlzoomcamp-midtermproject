import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore") 

output_file = f'model.bin'
dv_file = f'dv.bin'

df = pd.read_csv('laptop_prices.csv')


def category(inches, weight):
    if inches <= 14 and weight <= 1.54:
        return 'portable'
    elif inches > 15.6 and weight > 2.3:
        return 'heavy duty'
    elif inches > 14 and inches <= 15.6 and weight > 1.54 and weight <= 2.3:
        return 'standard'
    else:
        return 'Other'
    
def OS(os):
    if 'Windows' in os:
        return 'windows'
    elif 'Mac' in os or 'mac' in os:
        return 'mac'
    else:
        return 'other'

def resolution(res):
    if res < 1920*1080:
        return 'HD'
    elif 1920*1080 <= res < 2560*1440:
        return 'FullHD'
    elif 2560*1440 <= res < 3840*2160:
        return 'QuadHD'
    elif 3840*2160 <= res < 7680*2160:
        return 'UltraHD'
    else:
        return '8K'
    
def processor_category(model):
    if 'i3' in model:
        return 'i3'
    elif 'i5' in model:
        return 'i5'
    elif 'i7' in model:
        return 'i7'
    else:
        return 'AMD'
    
def rmse(y, y_pred):
    error = y-y_pred
    sr = error ** 2
    mse = sr.mean()
    return np.sqrt(mse)

#EDA
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(['typename', 'product', 'touchscreen', 'ipspanel', 'retinadisplay', 'secondarystorage', 'secondarystoragetype'], axis=1)
remv_brands = ['Chuwi', 'MSI', 'Microsoft', 'Huawei', 'Xiaomi', 'Vero', 'Razer','Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']
df = df[~df['company'].isin(remv_brands)]
df = df.reset_index(drop=True)
df['category'] = df.apply(lambda row: category(row['inches'], row['weight']), axis=1)
df = df.drop(['inches', 'weight'], axis=1)
df['OS'] = df['os'].apply(OS)
df = df.drop('os', axis=1)
df['res'] = df['screenw'] * df['screenh']
df['resolution'] = df['res'].apply(resolution)
df = df.drop(['screen', 'screenw', 'screenh', 'res'], axis=1)
df['gen'] = df['cpu_model'].apply(processor_category)
df = df.drop(['gpu_model', 'cpu_model'], axis=1)

strings = list(df.dtypes[df.dtypes == 'object'].index)
df[strings] = df[strings].apply(lambda x: x.str.lower().str.replace(' ', '_'))
df.columns = df.columns.str.lower()


df['price_lkr'] = round(df['price_euros']*307.15/3, 2)
df = df.drop('price_euros', axis=1)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

numerical = ['ram', 'cpu_freq', 'primarystorage']

categorical = [
    'company',
    'cpu_company',
    'primarystoragetype',
    'gpu_company',
    'category',
    'os',
    'resolution',
    'gen'
]

# training 

def train(df_train, y_train, n_estimators=300, max_depth=10, min_samples_split=4):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

# validation

print('doing validation: .....')

Kfold = KFold(n_splits=10, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in Kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log1p(df_train.price_lkr.values)
    y_val = np.log1p(df_val.price_lkr.values)

    dv, model = train(df_train, y_train, n_estimators=300, max_depth=10, min_samples_split=4)
    y_pred = predict(df_val, dv, model)

    rmse_val = rmse(y_val, y_pred)
    scores.append(rmse_val)

    print(f'rmse on fold {fold} is {round(rmse_val,3)}')
    fold+=1

print('validation result')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# training the final model

print('training the final model')

dv, model = train(df_full_train, np.log1p(df_full_train.price_lkr.values))
y_pred = predict(df_test, dv, model)

y_test = np.log1p(df_test.price_lkr.values)
rmse_value = rmse(y_test, y_pred)
rmse_value

print(f'rmse={rmse_value}')

with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

with open(dv_file, 'wb') as f_out_dv:
    pickle.dump((dv), f_out_dv)


print(f'the mode is saved to {output_file}')
print(f'the dv is saved to {dv_file}')



