import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import processing


Data = pd.read_csv('..\datasets\Stars.csv')
X = Data.iloc[:, :-1]
y = Data.iloc[:, -1]
y = y.astype('int')  # changing data type for y

X = processing.processing(X)

# Load the model from the file
loaded_knn_model = joblib.load('..\models\knn.pkl')
predictions = loaded_knn_model.predict(X)
accuracy = accuracy_score(y, predictions)

print(f"Validation Accuracy: {accuracy:.2f}")
