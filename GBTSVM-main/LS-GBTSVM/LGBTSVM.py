import numpy as np
import time
from cvxopt import matrix, solvers

def LGBTSVM(Data,TestX,d1, d2, d3, d4):
    C1 = Data[Data[:, -1] == 1, :-1]
    C2 = Data[Data[:, -1] != 1, :-1]
    A = C1[:,:-1]
    B=C2[:,:-1]
    R1=C1[:,-1]
    R2=C2[:,-1]

    start = time.time()
    p = A.shape[0]
    q = B.shape[0]
    lb1 = np.concatenate((np.zeros(p), np.zeros(q)))
    ub1 = np.concatenate((np.zeros(p), d1 * np.ones(q)))
    f1 = -d3 * np.concatenate((np.zeros(p), np.ones(q)+R2))

    Q1=np.vstack((np.hstack((np.dot(A, A.T) + d3 * np.eye(p), np.dot(A, B.T))), np.hstack((np.dot(B, A.T), np.dot(B, B.T))))) + np.ones((p + q, p + q))
    Q1 = (Q1 + Q1.T) / 2

    if np.linalg.matrix_rank(Q1) < Q1.shape[1]:
        Q1 = Q1 + 1e-4 * np.eye(Q1.shape[0])

    G=np.concatenate((np.eye(q+p), -np.eye(p+q)))
    h=np.concatenate((ub1,-lb1))
    solvers.options['show_progress'] = False
    alpha2= solvers.qp(matrix(Q1,tc='d'),matrix(f1,tc='d'),matrix(G,tc='d'),matrix(h,tc='d'))
    x1 = np.array(alpha2['x'])

    lb2 = np.concatenate((np.zeros(q), np.zeros(p)))
    ub2 = np.concatenate((np.zeros(q), d2 * np.ones(p)))
    f2 = -d4 * np.concatenate((np.zeros(q), np.ones(p)+R1))

    Q2 = np.vstack((np.hstack((np.dot(B, B.T) + d4 * np.eye(q), np.dot(B, A.T))), np.hstack((np.dot(A, B.T), np.dot(A, A.T))))) + np.ones((p + q, p + q))
    Q2 = (Q2 + Q2.T) / 2

    if np.linalg.matrix_rank(Q2) < Q2.shape[1]:
        Q2 = Q2 + 1e-4 * np.eye(Q2.shape[0])

    cd=np.concatenate((np.eye(q+p), -np.eye(p+q)))
    vcd=np.concatenate((ub2,-lb2))
    alpha2= solvers.qp(matrix(Q2,tc='d'),matrix(f2,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
    x2 = np.array(alpha2['x'])

    m = TestX.shape[0]
    test_data = TestX[:, :-1]

    b1=-(1 / d3) *np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x1)
    b2 = (1 / d4) * np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x2)

    w1 = -(1 / d3) * np.dot(np.hstack((A.T,B.T)),x1)
    w2 = (1 / d4) * np.dot(np.hstack((B.T,A.T)),x2)

    y1 = np.dot(test_data, w1) + b1 * np.ones(m)
    y2 = np.dot(test_data, w2) + b2 * np.ones(m)

    Predict_Y = np.sign(np.abs(y2) - np.abs(y1))

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