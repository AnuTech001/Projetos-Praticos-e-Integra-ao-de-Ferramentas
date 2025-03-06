from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

data = {
    'Combustivel': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel'],
    'Idade': [5, 10, 3, 7, 4],
    'Quilometragem': [50000, 120000, 30000, 70000, 45000],
    'Preco': [20000, 15000, 22000, 18000, 16000]
}

df = pd.DataFrame(data)

categorical_features = ['Combustivel']
categorical_transformer = OneHotEncoder()

numeric_features = ['Idade', 'Quilometragem']
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X = df.drop('Preco', axis=1)
y = df['Preco']

pipeline.fit(X, y)

predictions = pipeline.predict(X)
mse = mean_squared_error(y, predictions)

print(f'MSE: {mse}')
