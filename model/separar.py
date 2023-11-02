import pandas as pd

data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

# Guardar el 100% de los datos
data.to_csv('data/heart_failure_clinical_records_dataset_100.csv', index=False)

# Guardar el 50% de los datos
half_size = int(len(data) / 2)
half_data = data.iloc[:half_size]
half_data.to_csv('data/heart_failure_clinical_records_dataset_50.csv', index=False)

# Guardar el 20% de los datos
twenty_percent_size = int(len(data) * 0.2)
twenty_percent_data = data.iloc[:twenty_percent_size]
twenty_percent_data.to_csv('data/heart_failure_clinical_records_dataset_20.csv', index=False)