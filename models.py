from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, train_x, train_y):
        knn = KNeighborsClassifier(n_neighbors=5)
        self.model = knn.fit(train_x, train_y)

    def predict(self, test_x):
        self.predictions = self.model.predict(test_x)

    def accuracy_score(self, test_y):
        self.accuracy = accuracy_score(test_y, self.predictions)
        return self.accuracy

class mlp:
    def __init__(self, X_train, y_train):
        clf = MLPClassifier(random_state=1, max_iter=300)
        self.model = clf.fit(X_train, y_train)

    def predict(self, x_test):
        self.predictions = self.model.predict(x_test)
        #print(self.predictions)

    def accuracy_score(self, test_y):
        self.accuracy = 0.0
        for i in range(len(test_y)):
            if (self.predictions[i] == test_y[i]):
                self.accuracy+=1
        return self.accuracy / len(test_y)


class xgb:
    def __init__(self, X_train, y_train):
        xgb = XGBClassifier(n_estimators=1000)
        self.model = xgb.fit(X_train, y_train)

    def predict(self, x_test):
        self.predictions = self.model.predict(x_test)
        print(self.predictions)

    def accuracy_score(self, test_y):
        #self.accuracy =  self.model.score(test_y, self.predictions)
        #print(test_y)
        self.accuracy = round(accuracy_score(test_y, self.predictions), 5 )
        return self.accuracy
