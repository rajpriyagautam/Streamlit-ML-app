
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Create a sample dataset if dataset.csv is empty/missing
dataset_path = "data/dataset.csv"

if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
    data = {
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "feature_2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "feature_3": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                      6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        "feature_4": [5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                      5, 4, 5, 4, 5, 4, 5, 4, 5, 4],
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print("Sample dataset created.")

else:
    df = pd.read_csv(dataset_path)

FEATURES = ["feature_1", "feature_2", "feature_3", "feature_4"]
TARGET = "target"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.2%}")

joblib.dump(model, "model/model.joblib")

print("Model saved successfully!")
print("Location: model/model.joblib")

