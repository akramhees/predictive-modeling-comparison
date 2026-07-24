import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading california housing dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

print(f"Dataset shape: {df.shape}")
print(f"Target mean: {df['target'].mean():.2f}")
print(f"Target std: {df['target'].std():.2f}")

# Save raw data
df.to_csv('data/housing_data.csv', index=False)

# Split and scale
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
np.save('data/X_train.npy', X_train_scaled)
np.save('data/X_test.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("Data prep compelte")
