import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import processing


def run():
    Data = pd.read_csv('datasets\Stars.csv')
    X = Data.iloc[:, :-1]
    y = Data.iloc[:, -1]
    y=y.astype('int') #changing data type for y

    X = processing.processing(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

    neighbors = np.arange(2, 20)
    accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        accuracy[i] = knn.score(X, y)

    best_index = np.argmax(accuracy)
    best_k = neighbors[best_index]

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    val_predictions = knn.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print('best k is  ', best_k)
    print(f"Validation Accuracy: {accuracy:.2f}")

    joblib.dump(knn, 'models\knn.pkl')
    print('saved model in models directory')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
