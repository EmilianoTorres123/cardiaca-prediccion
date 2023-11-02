from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('data/heart_failure_clinical_records_dataset.csv'))
y = df.pop('DEATH_EVENT')
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajuste de hiperparámetros con Grid Search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 5, 10]
}

clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5)
clf.fit(X_train, y_train)

# Imprime el mejor conjunto de hiperparámetros
print('Mejor conjunto de hiperparámetros:')
print(clf.best_params_)

# Guarda el modelo entrenado
dump(clf, pathlib.Path('model/heart_failure_clinical_records_dataset-v1.joblib'))

# Evalúa el modelo en el conjunto de prueba
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
