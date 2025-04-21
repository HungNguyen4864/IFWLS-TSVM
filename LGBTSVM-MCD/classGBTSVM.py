import numpy as np
import time
from scipy.linalg import solve
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from itertools import combinations

class OvO_GBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, eps1=0.05, eps2=0.05):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.models = {}
        
    def fit(self, X_train, y_train):
        solvers.options['show_progress'] = False
        self.labels = np.unique(y_train)
        for (class1, class2) in combinations(self.labels, 2):
            idx = np.where((y_train == class1) | (y_train == class2))[0]
            X_binary = X_train[idx]
            y_binary = y_train[idx]

            y_binary = np.where(y_binary == class1, 1, -1)

            A = X_binary[y_binary == 1]
            B = X_binary[y_binary == -1]
            C1 = A[:,:-1]
            C2 = B[:,:-1]
            R1 = A[:,-1]
            R2 = B[:,-1]
            e1 = np.ones((A.shape[0], 1))
            e2 = np.ones((B.shape[0], 1))

            H1 = np.hstack((C1, e1))
            G1 = np.hstack((C2, e2))

            HH1 = np.dot(H1.T, H1) + self.eps1 * np.eye(H1.shape[1])
            HHG = linalg.solve(HH1, G1.T)

            kerH1 = np.dot(G1, HHG)
            kerH1 = (kerH1 + kerH1.T) / 2
            m1=kerH1.shape[0]
            e3= np.ones(m1)
            R11=-(e3+R2)

            vlb = np.zeros((m1, 1))
            vub = self.d1 * np.ones((m1, 1))
            G = np.vstack((np.eye(m1), -np.eye(m1)))
            h = np.vstack((vub, -vlb))

            alpha1 = solvers.qp(matrix(kerH1,tc='d'),matrix(R11,tc='d'),matrix(G,tc='d'),matrix(h,tc='d'))
            alphasol1 = np.array(alpha1['x'])
            z1 = -np.dot(HHG,alphasol1)
            w1, b1 = z1[:-1], z1[-1]


            # Solve second QP
            QQ = np.dot(G1.T, G1)
            QQ = QQ + self.eps2 * np.eye(QQ.shape[1])
            QQP = linalg.solve(QQ, H1.T)

            kerH2 = np.dot(H1, QQP)
            kerH2 = (kerH2 + kerH2.T) / 2
            m2=kerH2.shape[0]
            e4 = np.ones(m2)
            R22=-(e4+R1)


            vlb = np.zeros((m2,1))
            vub = self.d2*(np.ones((m2,1)))
            cd = np.vstack((np.identity(m2),-np.identity(m2)))
            vcd = np.vstack((vub,-vlb))
            alpha2= solvers.qp(matrix(kerH2,tc='d'),matrix(R22,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
            alphasol2 = np.array(alpha2['x'])
            z2 = np.dot(QQP,alphasol2)
            w2, b2 = z2[:-1], z2[-1]

            self.models[(class1, class2)] = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    def predict(self, X_test):
        votes = np.zeros((X_test.shape[0], len(self.labels)))

        for (class1, class2), params in self.models.items():
            y1 = np.dot(X_test, params['w1']) + params['b1']
            y2 = np.dot(X_test, params['w2']) + params['b2']

            pred = np.where(np.abs(y1) <= np.abs(y2), class1, class2)

            for idx, p in enumerate(pred):
                votes[idx, np.where(self.labels == p)[0][0]] += 1

        final_preds = self.labels[np.argmax(votes, axis=1)]
        return final_preds

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        return accuracy


class GBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, eps1=0.05, eps2=0.05):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.models = {}
    def fit(self, X_train):
        mew=1
        kerfPara = {}
        kerfPara['type'] = 'lin'
        kerfPara['pars'] = mew
        A = X_train[X_train[:, -1] == 1, :-1]
        B = X_train[X_train[:, -1] != 1, :-1]
        C1 = A[:,:-1]
        C2=B[:,:-1]
        R1=A[:,-1]
        R2=B[:,-1]

        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        
        H1 = np.hstack((C1, e1))
        G1 = np.hstack((C2, e2))
        
        HH1 = np.dot(H1.T, H1) + self.eps1 * np.eye(H1.shape[1])

        HHG = linalg.solve(HH1, G1.T)
        kerH1 = np.dot(G1, HHG)
        kerH1 = (kerH1 + kerH1.T) / 2
        m1=kerH1.shape[0]
        e3= np.ones(m1)
        R11=-(e3+R2)
        solvers.options['show_progress'] = False
        vlb = np.zeros((m1,1))
        vub = self.d1*(np.ones((m1,1)))
        G=np.vstack((np.eye(m1), -np.eye(m1)))
        h=np.vstack((vub,-vlb))
        alpha1 = solvers.qp(matrix(kerH1,tc='d'),matrix(R11,tc='d'),matrix(G,tc='d'),matrix(h,tc='d'))
        alphasol1 = np.array(alpha1['x'])
        z = -np.dot(HHG,alphasol1)
        self.w1 = z[:len(z)-1]
        self.b1 = z[len(z)-1]

        # 2nd QPP

        QQ = np.dot(G1.T, G1)
        QQ = QQ + self.eps2 * np.eye(QQ.shape[1])
        QQP = linalg.solve(QQ, H1.T)
        kerH2 = np.dot(H1, QQP)
        kerH2 = (kerH2 + kerH2.T) / 2
        m2=kerH2.shape[0]
        e4 = np.ones(m2)
        R22=-(e4+R1)
        solvers.options['show_progress'] = False
        vlb = np.zeros((m2,1))
        vub = self.d2*(np.ones((m2,1)))
        cd = np.vstack((np.identity(m2),-np.identity(m2)))
        vcd = np.vstack((vub,-vlb))
        alpha2= solvers.qp(matrix(kerH2,tc='d'),matrix(R22,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
        alphasol2 = np.array(alpha2['x'])
        z = np.dot(QQP,alphasol2)
        self.w2 = z[:len(z)-1]
        self.b2 = z[len(z)-1]

    def predict(self, X_test):
        P_1 = X_test[:, :-1]
        y1 = np.dot(P_1, self.w1) + self.b1
        y2 = np.dot(P_1, self.w2) + self.b2
        y_pred = np.zeros((y1.shape[0], 1))
        for i in range(y1.shape[0]):
            if np.min([np.abs(y1[i]), np.abs(y2[i])]) == np.abs(y1[i]):
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        y_pred = y_pred.T
        return  y_pred
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