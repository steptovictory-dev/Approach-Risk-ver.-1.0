from pathlib import Path

import pandas as pd


data_path = (
    Path(__file__).resolve().parent
    / 'data'
    / 'raw'
    / 'go_arounds_augmented.csv.gz'
)

df = pd.read_csv(data_path, nrows=10_000, low_memory=False)

print('Размер прочитанной части:')
print(df.shape)

print('\nСтолбцы и типы:')
print(df.dtypes.to_string())

print('\nПервые пять записей:')
print(df.head().to_string())

print('\nКоличество пропусков:')
print(df.isna().sum().to_string())

print('\nРаспределение метки has_ga:')
print(df['has_ga'].value_counts(dropna=False))