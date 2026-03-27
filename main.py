from src.load import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import explain_model

df = load_data()

print(df.head())
print(df.shape)

X, y = preprocess(df)

model, X_test, y_test = train_model(X, y)
evaluate_model(model, X_test, y_test)

explain_model(model, X_test)