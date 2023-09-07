import pandas as pd
import numpy as np
import random as rnd
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


lab_score = ['удовлетворительно', 'хорошо', 'отлично']


# функция генерации рандомных значений и создание датасета
def createdataset():
    global lab_score
    tmp = {
        'student': [],
        'subject1': [],
        'subject2': [],
        'subject3': [],
        'subject4': [],
        'subject5': [],
        'subject6': [],
        'lab_work_time': [],
        'lab_work_score': []
    }
    for i in range(2010):
        tmp['student'].append('Student_'+str(i+1))
        tmp['subject1'].append(rnd.randint(0, 99))
        tmp['subject2'].append(rnd.randint(0, 99))
        tmp['subject3'].append(rnd.randint(0, 99))
        tmp['subject4'].append(rnd.randint(0, 99))
        tmp['subject5'].append(rnd.randint(0, 99))
        tmp['subject6'].append(rnd.randint(0, 99))
        tmp['lab_work_time'].append(rnd.randint(40, 120))
        tmp['lab_work_score'].append(rnd.choice(lab_score))
    df = pd.DataFrame(tmp)
    df.to_csv(r'ds_files\cb_dataset.csv', index=False)


# функция обечения модели
def train_model(ds):
    global lab_score
    df = pd.read_csv(ds)
    df['lab_work_score'] = df['lab_work_score'].apply(lambda x: lab_score.index(x))
    # признаки и целевой показатель
    df = df.drop(columns=['student'])
    X = df.drop(columns=['lab_work_score'])
    Y = df['lab_work_score']
    selector = SelectKBest(chi2, k=3)
    X_new = selector.fit_transform(X, Y)
    X_new = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])

    ###### сравнение выбора признаков
    # print(X_new)
    # matrix = np.triu(X.corr())
    # plt.figure(figsize=(16, 6))
    # sns.heatmap(X.corr(), annot=True, mask=matrix, center=0, cmap='PiYG', linewidths=1, linecolor='white')
    # plt.show()
    ######

    # разделим выборку на тестовую и обучающую
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    # обучение модели
    model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=list(X_new.keys()))
    model.fit(X_train, y_train, eval_set=(X_test, y_test),  verbose=10)
    # сохраним модель
    model.save_model(r'ds_files\stu_model.cbm')
    y_pred = model.predict(X_test)
    # оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    s = f'Модель обучилась: {str(model.is_fitted())}\n' \
        f'Параметры модели: {model.get_params()}\n' \
        f'Точность модели: {accuracy}'
    with open(r'ds_files\model_param.txt', 'w') as mp:
        mp.write(s)
    print(f"Модель обучилась: {str(model.is_fitted())}")
    print(f"Параметры модели: {model.get_params()}")
    print(f'Точность модели: {accuracy}')


# функция прогноза оценки за итоговую лабораторную работу
def predict_lab(m_file, sub_score):
    global lab_score
    model = CatBoostClassifier()
    predict_df = pd.DataFrame(sub_score)
    if os.path.isfile(m_file):
        model.load_model(m_file)
        predict = model.predict(predict_df)[0]
        print('Прогнозная оценка за итоговую лабораторную работу: ' + lab_score[predict[0]])
    else:
        print('Модель для выполнения прогноза не найдена')


if __name__ == '__main__':
    if not os.path.isfile(r'ds_files\cb_dataset.csv'):
        print('# Датасет не найден. Генерация датасета #')
        createdataset()
    elif not os.path.isfile(r'ds_files\stu_model.cbm'):
        print('# Старт обучения модели #')
        train_model(r'ds_files\cb_dataset.csv')
    else:
        predict_lab(r'ds_files\stu_model.cbm',
                    {'subject1': [60],
                     'subject2': [50],
                     'subject3': [40],
                     'subject4': [30],
                     'subject5': [20],
                     'subject6': [10],
                     'lab_work_time': [40]})

