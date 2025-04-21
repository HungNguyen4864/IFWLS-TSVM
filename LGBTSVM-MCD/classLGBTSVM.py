import numpy as np
import time
from scipy.linalg import solve
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from itertools import combinations

class LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1,d3=0.1, d4=0.1, eps1=0.05, eps2=0.05):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.eps1 = eps1
        self.eps2 = eps2
        self.models = {}
    def fit(self, X_train):
        C1 = X_train[X_train[:, -1] == 1, :-1]
        C2 = X_train[X_train[:, -1] != 1, :-1]
        A = C1[:,:-1]
        B=C2[:,:-1]
        R1=C1[:,-1]
        R2=C2[:,-1]
        p = A.shape[0]
        q = B.shape[0]
        lb1 = np.concatenate((np.zeros(p), np.zeros(q)))
        ub1 = np.concatenate((np.zeros(p), self.d1 * np.ones(q)))
        f1 = -(self.d3) * np.concatenate((np.zeros(p), np.ones(q)+R2))

        Q1=np.vstack((np.hstack((np.dot(A, A.T) + self.d3 * np.eye(p), np.dot(A, B.T))), 
                      np.hstack((np.dot(B, A.T), np.dot(B, B.T))))) + np.ones((p + q, p + q))
        Q1 = (Q1 + Q1.T) / 2

        if np.linalg.matrix_rank(Q1) < Q1.shape[1]:
            Q1 = Q1 + 1e-4 * np.eye(Q1.shape[0])

        G=np.concatenate((np.eye(q+p), -np.eye(p+q)))
        h=np.concatenate((ub1,-lb1))
        solvers.options['show_progress'] = False
        alpha2= solvers.qp(matrix(Q1,tc='d'),matrix(f1,tc='d'),matrix(G,tc='d'),matrix(h,tc='d'))
        x1 = np.array(alpha2['x'])

        lb2 = np.concatenate((np.zeros(q), np.zeros(p)))
        ub2 = np.concatenate((np.zeros(q), self.d2 * np.ones(p)))
        f2 = -(self.d4) * np.concatenate((np.zeros(q), np.ones(p)+R1))

        Q2 = np.vstack((np.hstack((np.dot(B, B.T) + self.d4 * np.eye(q), np.dot(B, A.T))), np.hstack((np.dot(A, B.T), np.dot(A, A.T))))) + np.ones((p + q, p + q))
        Q2 = (Q2 + Q2.T) / 2

        if np.linalg.matrix_rank(Q2) < Q2.shape[1]:
            Q2 = Q2 + 1e-4 * np.eye(Q2.shape[0])

        cd=np.concatenate((np.eye(q+p), -np.eye(p+q)))
        vcd=np.concatenate((ub2,-lb2))
        alpha2= solvers.qp(matrix(Q2,tc='d'),matrix(f2,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
        x2 = np.array(alpha2['x'])

        self.b1=-(1 / self.d3) *np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x1)
        self.b2 = (1 / self.d4) * np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x2)

        self.w1 = -(1 / self.d3) * np.dot(np.hstack((A.T,B.T)),x1)
        self.w2 = (1 / self.d4) * np.dot(np.hstack((B.T,A.T)),x2)
    def predict(self, X_test):
        m = X_test.shape[0]
        test_data = X_test[:, :-1]
        y1 = np.dot(test_data, self.w1) + self.b1 * np.ones(m)
        y2 = np.dot(test_data, self.w2) + self.b2 * np.ones(m)
        y_pred = np.sign(np.abs(y2) - np.abs(y1))
        y_pred = y_pred.T
        return y_pred
    def score(self, X_test):
        y_pred = self.predict(X_test)
        no_test, no_col = X_test.shape
        err = 0.0
        obs1 = X_test[:, no_col-1]
        for i in range(no_test):
            if np.sign(y_pred[0, i]) != np.sign(obs1[i]):
                err += 1
        acc = ((X_test.shape[0] - err) / X_test.shape[0]) * 100
        return acc

class OvO_LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.models = {}

    def fit_binary(self, X_train):
        C1 = X_train[X_train[:, -1] == 1, :-1]
        C2 = X_train[X_train[:, -1] != 1, :-1]
        A = C1[:, :-1]
        B = C2[:, :-1]
        R1 = C1[:, -1]
        R2 = C2[:, -1]

        p, q = A.shape[0], B.shape[0]
        lb1 = np.concatenate((np.zeros(p), np.zeros(q)))
        ub1 = np.concatenate((np.zeros(p), self.d1 * np.ones(q)))
        f1 = -(self.d3) * np.concatenate((np.zeros(p), np.ones(q) + R2))

        Q1 = np.vstack((np.hstack((np.dot(A,A.T) + self.d3 * np.eye(p), np.dot(A,B.T))),
                        np.hstack((np.dot(B,A.T), np.dot(B,B.T))))) + np.ones((p + q, p + q))
        Q1 = (Q1 + Q1.T) / 2

        G = np.concatenate((np.eye(q + p), -np.eye(p + q)))
        h = np.concatenate((ub1, -lb1))
        solvers.options['show_progress'] = False
        alpha1 = solvers.qp(matrix(Q1, tc='d'), matrix(f1, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))
        x1 = np.array(alpha1['x'])

        lb2 = np.concatenate((np.zeros(q), np.zeros(p)))
        ub2 = np.concatenate((np.zeros(q), self.d2 * np.ones(p)))
        f2 = -(self.d4) * np.concatenate((np.zeros(q), np.ones(p) + R1))

        Q2 = np.vstack((np.hstack((np.dot(B, B.T) + self.d4 * np.eye(q), np.dot(B, A.T))),
                        np.hstack((np.dot(A , B.T), np.dot(A , A.T))))) + np.ones((p + q, p + q))
        Q2 = (Q2 + Q2.T) / 2

        cd = np.concatenate((np.eye(q + p), -np.eye(p + q)))
        vcd = np.concatenate((ub2, -lb2))
        alpha2 = solvers.qp(matrix(Q2, tc='d'), matrix(f2, tc='d'), matrix(cd, tc='d'), matrix(vcd, tc='d'))
        x2 = np.array(alpha2['x'])

        b1 = -(1 / self.d3) * np.dot(np.hstack((np.ones(p), np.ones(q))), x1)
        b2 = (1 / self.d4) * np.dot(np.hstack((np.ones(p), np.ones(q))), x2)

        w1 = -(1 / self.d3) * np.dot(np.hstack((A.T, B.T)), x1)
        w2 = (1 / self.d4) * np.dot(np.hstack((B.T, A.T)), x2)

        return w1, b1, w2, b2

    def fit(self, X, y):
        self.labels_ = np.unique(y)
        for class1, class2 in combinations(self.labels_, 2):
            idx = np.where((y == class1) | (y == class2))[0]
            X_binary = X[idx]
            y_binary = y[idx]
            y_binary = np.where(y_binary == class1, 1, -1)
            X_binary = np.hstack((X_binary, y_binary.reshape(-1, 1)))
            w1, b1, w2, b2 = self.fit_binary(X_binary)
            self.models[(class1, class2)] = (w1, b1, w2, b2)

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.labels_)))
        for (class1, class2), (w1, b1, w2, b2) in self.models.items():
            y1 = np.dot(X, w1) + b1
            y2 = np.dot(X ,w2) + b2
            predictions = np.where(np.abs(y1) < np.abs(y2), class1, class2)
            for i, pred in enumerate(predictions):
                votes[i, np.where(self.labels_ == pred)[0][0]] += 1

        return self.labels_[np.argmax(votes, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class OvR_LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.classifiers = {}

    def fit_binary(self, X_train, y_train_binary):
        C1 = X_train[y_train_binary == 1]
        C2 = X_train[y_train_binary != 1]

        A = C1[:, :-1]
        B = C2[:, :-1]
        R1 = C1[:, -1]
        R2 = C2[:, -1]

        p = A.shape[0]
        q = B.shape[0]

        lb1 = np.concatenate((np.zeros(p), np.zeros(q)))
        ub1 = np.concatenate((np.zeros(p), self.d1 * np.ones(q)))
        f1 = -(self.d3) * np.concatenate((np.zeros(p), np.ones(q) + R2))

        Q1 = np.vstack((np.hstack((np.dot(A, A.T) + self.d3 * np.eye(p), np.dot(A, B.T))),
                        np.hstack((np.dot(B, A.T), np.dot(B, B.T))))) + np.ones((p + q, p + q))
        Q1 = (Q1 + Q1.T) / 2

        if np.linalg.matrix_rank(Q1) < Q1.shape[1]:
            Q1 += 1e-4 * np.eye(Q1.shape[0])

        G = np.concatenate((np.eye(q + p), -np.eye(p + q)))
        h = np.concatenate((ub1, -lb1))

        solvers.options['show_progress'] = False
        alpha2 = solvers.qp(matrix(Q1, tc='d'), matrix(f1, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))
        x1 = np.array(alpha2['x'])

        lb2 = np.concatenate((np.zeros(q), np.zeros(p)))
        ub2 = np.concatenate((np.zeros(q), self.d2 * np.ones(p)))
        f2 = -(self.d4) * np.concatenate((np.zeros(q), np.ones(p) + R1))

        Q2 = np.vstack((np.hstack((np.dot(B, B.T) + self.d4 * np.eye(q), np.dot(B, A.T))),
                        np.hstack((np.dot(A, B.T), np.dot(A, A.T))))) + np.ones((p + q, p + q))
        Q2 = (Q2 + Q2.T) / 2

        if np.linalg.matrix_rank(Q2) < Q2.shape[1]:
            Q2 += 1e-4 * np.eye(Q2.shape[0])

        cd = np.concatenate((np.eye(q + p), -np.eye(p + q)))
        vcd = np.concatenate((ub2, -lb2))
        alpha2 = solvers.qp(matrix(Q2, tc='d'), matrix(f2, tc='d'), matrix(cd, tc='d'), matrix(vcd, tc='d'))
        x2 = np.array(alpha2['x'])

        b1 = -(1 / self.d3) * np.dot(np.hstack((np.ones(p), np.ones(q))), x1)
        b2 = (1 / self.d4) * np.dot(np.hstack((np.ones(p), np.ones(q))), x2)

        w1 = -(1 / self.d3) * np.dot(np.hstack((A.T, B.T)), x1)
        w2 = (1 / self.d4) * np.dot(np.hstack((B.T, A.T)), x2)

        return (w1, b1), (w2, b2)

    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.classifiers = {}

        for cls in self.classes_:
            y_train_binary = np.where(y_train == cls, 1, -1)
            self.classifiers[cls] = self.fit_binary(X_train, y_train_binary)

    def predict(self, X_test):
        scores = np.zeros((X_test.shape[0], len(self.classes_)))

        for idx, cls in enumerate(self.classes_):
            (w1, b1), (w2, b2) = self.classifiers[cls]
            y1 = np.dot(X_test, w1) + b1
            y2 = np.dot(X_test, w2) + b2
            scores[:, idx] = (np.abs(y2) - np.abs(y1)).flatten()

        predictions = self.classes_[np.argmax(scores, axis=1)]
        return predictions

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
