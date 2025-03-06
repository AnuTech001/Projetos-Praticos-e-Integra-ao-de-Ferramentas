import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
params = {
    'vs_currency': 'usd',
    'days': '30'
}

response = requests.get(url, params=params)
data = response.json()

prices = data['prices']

df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['price_change'] = df['price'].diff().fillna(0)
df['label'] = (df['price_change'] > 0).astype(int)
df.to_csv('bitcoin_prices.csv', index=False)


df = pd.read_csv('bitcoin_prices.csv')
X = df[['price']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'Acurácia: {accuracy}')

feature_importances = model.feature_importances_
for i, feature in enumerate(X.columns):
    print(f'Importância de {feature}: {feature_importances[i]}')
