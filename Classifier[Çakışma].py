import numpy as np
import random as rn
import csv

class LogisticRegression:#standart logistic regression classifier. uses lr and niter to determine learning rate and epocs
    def __init__(self, lr=0.02, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter

    def predict(self, X):
        X = (X - self.x_mean) / self.x_stddev
        linear = self.weight_mult(X)
        preds = self.sigmoid(linear)
        return (preds >= 0.5).astype('int')

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def weight_mult(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X_train, Y_train):
        self.weights = np.random.rand(X_train.shape[1], 1)
        self.bias = np.zeros((1,))
        self.x_mean = X_train.mean(axis=0).T
        self.x_stddev = X_train.std(axis=0).T

        X_train = (X_train - self.x_mean) / self.x_stddev

        for i in range(self.n_iter):
            probs = self.sigmoid(self.weight_mult(X_train))
            diff = probs - Y_train

            delta_weights = np.mean(diff * X_train, axis=0, keepdims=True).T
            delta_bias = np.mean(diff)

            self.weights = self.weights - self.lr * delta_weights
            self.bias = self.bias - self.lr * delta_bias
        return self
    
    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def loss(self, X, y):
        probs = self.sigmoid(self.weight_mult(X))


        pos_log = y * np.log(probs + 1e-15)

        neg_log = (1 - y) * np.log((1 - probs) + 1e-15)

        l = -np.mean(pos_log + neg_log)
        return l

def dsopen(owo):
    X = []
    Y = []

    with open("data_"+owo+".csv") as csv_file:  #datasets :data_orb, data_hum, data_kaze
        csv_reader = csv.reader(csv_file, delimiter=',') #this section reads data we extracted from the images from their corresponding csv files
        for row in csv_reader:
            X.append([float(x) for x in row[2:]])
            Y.append(row[1])

    
    X = np.array(X)
    Y = (np.array(Y) == 'M').astype('float')
    Y = np.expand_dims(Y, -1)#expand the dimention of array -1
    return(X,Y)

def train_test_split(X, Y, split=0.2): #this section splits our data into test and train parts acccording to the cutoff value
    rn.shuffle(X)
    rn.shuffle(Y)

    split_x = int(1-split * len(X))
    split_y = int(1-split * len(Y))

    x_train, x_test = X[:split_x], X[split_x:]
    y_train, y_test = Y[:split_y], Y[split_y:]

    return x_train, y_train, x_test, y_test

def run(owo):#function to run each database
    X,Y = dsopen(owo)
    x_train, y_train, x_test, y_test = train_test_split(X, Y)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print(owo+"--------------------------------------")
    print('Accuracy on test set: {:.2f}%'.format(lr.accuracy(x_test, y_test) * 100))
    print('Loss on test set: {:.2f}'.format(lr.loss(x_test, y_test)))
    print("-----------------------------------------")

def main():
    run("orb")
    #run("hum")
    #run("kaze")

main()
