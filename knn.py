import numpy as np
from collections import Counter

class KNN:

    def __init__(self, K: int, X, Y):
        self.K = K
        self.X_train = X
        self.Y_train = Y

    def eculidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    

    def predict(self, X):

        X = np.atleast_2d(X)
        predictions = []

        for test_point in X:

            distances = []

            for i in range(len(self.X_train)):

                dist = self.eculidean_distance(test_point, self.X_train[i])
                distances.append((dist, self.Y_train[i]))

            distances.sort(key=lambda x: x[0])

            k = min(self.K, len(self.X_train))
            k_labels = [distances[i][1] for i in range(k)]

            pred = Counter(k_labels).most_common(1)[0][0]
            predictions.append(pred)

        return predictions


# using KNN 

X_train = np.array([[1,2],[2,3],[3,3],[6,7]])
y_train = np.array([0,0,0,1])

knn = KNN(K=3, X=X_train, Y=y_train)

X_test = np.array([[2,2.5],[5,6]])

print(knn.predict(X_test))