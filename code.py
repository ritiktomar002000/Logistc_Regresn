import numpy as np
import pickle


class LogisticRegressionModel:
    def __init__(self, learning_rate=0.005, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def fit(self, X, Y):
        n_features = X.shape[0]
        m = X.shape[1]

        self.initialize_parameters(n_features)

        for i in range(self.num_iterations):
            Z = np.dot(self.weights.T, X) + self.bias
            A = self.sigmoid(Z)

            dw = (1 / m) * np.dot(X, (A - Y).T)
            db = (1 / m) * np.sum(A - Y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                cost = self.compute_cost(A, Y)
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        Z = np.dot(self.weights.T, X) + self.bias
        A = self.sigmoid(Z)
        return (A > 0.5).astype(int)


def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    data = load_data("data/data_dog_nondog.pickle")

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    model = LogisticRegressionModel(learning_rate=0.005, num_iterations=2000)
    model.fit(X_train, Y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = 100 - np.mean(np.abs(train_preds - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_preds - Y_test)) * 100

    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
