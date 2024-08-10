import pandas as pd

datasets = {
    'iris': {
        'path': 'datasets/iris/iris.data',
        'column_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
        'target': 'species'
    },
    'wine': {
        'path': 'datasets/wine/wine.data',
        'column_names': ['class'] + [f'feature_{i}' for i in range(1, 14)],
        'target': 'class'
    },
    'breast_cancer': {
        'path': 'datasets/breast_cancer/wdbc.data',
        'column_names': ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)],
        'target': 'diagnosis'
    }
}

for name, info in datasets.items():
    df = pd.read_csv(info['path'], header=None)
    df.columns = info['column_names']
    df.rename(columns={info['target']: 'target'}, inplace=True)
    df.to_csv(f'csv/{name}.csv', index=False)