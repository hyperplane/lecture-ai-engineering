import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_data(path, model):
    df = pd.read_csv(path).dropna()
    return df


def measure_time(model_path):
    model = joblib.load(model_path)
    X_input = load_data(test_path, model)
    start = time.time()
    model.predict(X_input)
    end = time.time()
    return end - start


test_path = "day5/演習3/data/Titanic.csv"
model_current_path = "day5/演習3/models/titanic_model.pkl"
model_original_path = "day5/演習3/models/original_titanic_model.pkl"

time_current = measure_time(model_current_path)
time_original = measure_time(model_original_path)

print(f"現行モデル推論時間: {time_current:.4f} 秒")
print(f"過去モデル推論時間: {time_original:.4f} 秒")

assert time_current <= time_original * 1.2, "現行モデルの推論が遅すぎる"
