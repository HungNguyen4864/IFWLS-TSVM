import numpy as np
import time
from scipy.linalg import solve
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers




def Pin_GBTSVM(DataTrain, TestX, d1, d2, tau):
    mew=1
    start = time.time()
    eps1 = 0.05
    eps2 = 0.05
    kerfPara = {}
    kerfPara['type'] = 'lin'
    kerfPara['pars'] = mew
    A = DataTrain[DataTrain[:, -1] == 1, :-1]
    B = DataTrain[DataTrain[:, -1] != 1, :-1]
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
    
    HH1 = np.dot(H1.T, H1) + eps1 * np.eye(H1.shape[1])

    HHG = linalg.solve(HH1, G1.T)
    kerH1 = np.dot(G1, HHG)
    kerH1 = (kerH1 + kerH1.T) / 2
    m1=kerH1.shape[0]
    e3= np.ones(m1)
    R11=-(e3+R2)
    solvers.options['show_progress'] = False
    vlb = tau*d1*(np.ones((m1,1)))
    
    G=-np.eye(m1)
    
    alpha1 = solvers.qp(matrix(kerH1,tc='d'),matrix(R11,tc='d'),matrix(G,tc='d'),matrix(vlb,tc='d'))
    alphasol1 = np.array(alpha1['x'])
    z = -np.dot(HHG,alphasol1)
    w1 = z[:len(z)-1]
    b1 = z[len(z)-1]

    # 2nd QPP

    QQ = np.dot(G1.T, G1)
    QQ = QQ + eps2 * np.eye(QQ.shape[1])
    QQP = linalg.solve(QQ, H1.T)
    kerH2 = np.dot(H1, QQP)
    kerH2 = (kerH2 + kerH2.T) / 2
    m2=kerH2.shape[0]
    e4 = np.ones(m2)
    R22=-(e4+R1)
    solvers.options['show_progress'] = False
    vlb = tau*d2*(np.ones((m2,1)))
   
    G=-np.eye(m2)
    alpha2= solvers.qp(matrix(kerH2,tc='d'),matrix(R22,tc='d'),matrix(G,tc='d'),matrix(vlb,tc='d'))
    alphasol2 = np.array(alpha2['x'])
    z = np.dot(QQP,alphasol2)
    w2 = z[:len(z)-1]
    b2 = z[len(z)-1]

    
    P_1 = TestX[:, :-1]
    y1 = np.dot(P_1, w1) + b1
    y2 = np.dot(P_1, w2) + b2
    

    Predict_Y = np.zeros((y1.shape[0], 1))
    for i in range(y1.shape[0]):
        if np.min([np.abs(y1[i]), np.abs(y2[i])]) == np.abs(y1[i]):
            Predict_Y[i] = 1
        else:
            Predict_Y[i] = -1
        
    no_test, no_col = TestX.shape
    err = 0.0
    Predict_Y = Predict_Y.T
    obs1 = TestX[:, no_col-1]
    for i in range(no_test):
        if np.sign(Predict_Y[0, i]) != np.sign(obs1[i]):
            err += 1
    acc = ((TestX.shape[0] - err) / TestX.shape[0]) * 100
    end = time.time()
    Time=end - start
    return acc, Time

# url=r"C:\Users\HP\Music\abalone9-18.csv"
# df=pd.read_csv(url,header=None)
# DataTrain=df.values

# TWSVM_main(DataTrain, DataTrain, 0.5, 0.5, 0.5)