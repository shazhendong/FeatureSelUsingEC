import os
import pandas as pd
from sklearn.model_selection import train_test_split

def attribute_tran(y):
    if isinstance(y[0], str):
        for index in range(len(y)):
            if y[index] == 'normal':
                y[index] = 0
            else:
                y[index] = 1
    y = pd.Series(y, name='Labels')
    return y

def train_test(file, x, y, train_pct, seed):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1-train_pct, random_state=seed, stratify=y)
    print(file + ': succeed')
    train = pd.concat([train_x, train_y], axis=1)
    test = pd.concat([test_x, test_y], axis=1)
    return train, test

def save_data(train,test,file,train_path):
    save_path = train_path + '/' + file + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train.to_csv(save_path + '/' + file + '_train.csv', index=False)
    test.to_csv(save_path + '/' + file + '_test.csv', index=False)




