from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
class TSVM_MCD_new(BaseEstimator):
    def __init__(self, kernel = None,polyconst =1,degree=2,gamma = 1,c1=None,c2=None,c3=None,c4=None):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.degree = degree
        self.gamma = float(gamma)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3      
        self.c4 = c4
        if self.c1 is not None: self.c1 = float(self.c1)        
        if self.c2 is not None: self.c2 = float(self.c2)       
        if self.c3 is not None: self.c3 = float(self.c3)        
        if self.c4 is not None: self.c4 = float(self.c4)       
        self.kf = {'linear':self.linear, 'polynomial':self.polynomial, 'rbf':self.rbf}
        self.k = None
        self.l = None
    def linear(self, x, y):
        return np.dot(x.T, y)
    def polynomial(self, x, y):       
        return (self.polyconst + np.dot(x.T, y))**self.degree      
    def rbf(self,x,y):         
        return np.exp(-1.0*self.gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))      
    def transform(self, X, C):         
        K = np.zeros((X.shape[0],C.shape[0]))         
        for i in range(X.shape[0]):             
            for j in range(C.shape[0]):                 
                K[i,j] = self.kf[self.kernel](X[i],C[j])         
        return K      
    # def cluster(self, X, y):
    #     ### Clustering class A, B.
    #     A = X[np.where(y!=-1)]
    #     B = X[np.where(y==-1)]
    #     # generate the linkage matrix
    #     L_A = linkage(A, 'ward')
    #     L_B = linkage(B, 'ward')
    #     # number of clusters
    #     last_A = L_A[-10:, 2]
    #     last_B = L_B[-10:, 2]
    #     last_rev = last_A[::-1]
    #     idxs = np.arange(1, len(last_A) + 1)
    #     # plt.plot(idxs, last_rev)
    #     acceleration_A = np.diff(last_A, 2)  # 2nd derivative of the distances
    #     acceleration_rev_A = acceleration_A[::-1]
    #     acceleration_B = np.diff(last_B, 2)
    #     acceleration_rev_B = acceleration_B[::-1]
    #     # plt.plot(idxs[:-2] + 1, acceleration_rev_A)
    #     # plt.show()
    #     k = acceleration_rev_A.argmax() + 1  # if idx 0 is the max of this we want 1 clusters
    #     l = acceleration_rev_B.argmax() + 1
    #     print ("clusters_A:", k) 
    #     print("Clusters_B:", l)
    #     # Retrieve the clusters_A, clusters_B
    #     clusters_A = fcluster(L_A, k, criterion='maxclust')
    #     clusters_B = fcluster(L_B, l, criterion='maxclust')
    #     print(clusters_A, clusters_B)
    #     # Visualizing clusters_A
    #     # plt.figure(figsize=(10, 8))
    #     # plt.scatter(A[:,0], A[:,1], c=clusters_A, cmap='prism')  # plot points with cluster dependent colors
    #     # plt.show()
    #     self.labels_A = np.unique(clusters_A)
    #     Z_A = []
    #     if k != 1:
    #         for i in range(k):
    #             Ai = A[np.where(clusters_A == self.labels_A[i])]
    #             Z_A.append(Ai)
    #     else:
    #         Z_A.append(A)

    #     self.labels_B = np.unique(clusters_B)
    #     Z_B = []
    #     if l != 1:
    #         for j in range(l):
    #             Bj = B[np.where(clusters_B == self.labels_B[j])]
    #             Z_B.append(Bj)
    #     else:
    #         Z_B.append(B)
    #     return k, l, Z_A, Z_B
    def cluster(self, X, y):
    # Tách hai nhóm A, B
        A = X[np.where(y != -1)]
        B = X[np.where(y == -1)]

        # ----- Xử lý nhóm A -----
        if len(A) < 2:
            # Nếu A chỉ có 0 hoặc 1 mẫu, coi như có 1 cụm duy nhất
            k = 1
            print("clusters_A:", k)

            # Tự gán tất cả mẫu vào cùng 1 cụm (nếu có)
            Z_A = [A] if len(A) > 0 else []
            # (hoặc chỉ gán mảng rỗng nếu len(A) == 0)
            clusters_A = np.zeros(len(A), dtype=int)

        else:
            # Gọi linkage
            L_A = linkage(A, 'ward')
            # Xác định số cụm bằng cách dựa trên “khoảng cách dừng”
            last_A = L_A[-10:, 2]
            acceleration_A = np.diff(last_A, 2)
            acceleration_rev_A = acceleration_A[::-1]
            # Tìm vị trí mà độ tăng tốc (2nd derivative) lớn nhất
            k = acceleration_rev_A.argmax() + 1
            print("clusters_A:", k)

            # Lấy label cụm
            clusters_A = fcluster(L_A, k, criterion='maxclust')

            # Gom dữ liệu các cụm
            unique_labels_A = np.unique(clusters_A)
            Z_A = []
            if k != 1:
                for lab in unique_labels_A:
                    Ai = A[np.where(clusters_A == lab)]
                    Z_A.append(Ai)
            else:
                Z_A.append(A)

        # ----- Xử lý nhóm B -----
        if len(B) < 2:
            # Tương tự cho B
            l = 1
            print("clusters_B:", l)

            Z_B = [B] if len(B) > 0 else []
            clusters_B = np.zeros(len(B), dtype=int)

        else:
            L_B = linkage(B, 'ward')
            last_B = L_B[-10:, 2]
            acceleration_B = np.diff(last_B, 2)
            acceleration_rev_B = acceleration_B[::-1]
            l = acceleration_rev_B.argmax() + 1
            print("clusters_B:", l)

            clusters_B = fcluster(L_B, l, criterion='maxclust')

            unique_labels_B = np.unique(clusters_B)
            Z_B = []
            if l != 1:
                for lab in unique_labels_B:
                    Bj = B[np.where(clusters_B == lab)]
                    Z_B.append(Bj)
            else:
                Z_B.append(B)

        # Hàm trả về:
        # k: số cụm của A
        # l: số cụm của B
        # Z_A: danh sách mảng các điểm mỗi cụm của A
        # Z_B: danh sách mảng các điểm mỗi cụm của B
        return k, l, Z_A, Z_B

    def fit(self, X, y):         
        self.k, self.l, self.Z_A, self.Z_B = self.cluster(X, y)         
        A = X[np.where(y!=-1)]         
        B = X[np.where(y==-1)]         
        self.C = np.vstack((A,B))          
        n = A.shape[1]         
        m = self.C.shape[0]          
        self.m_A = A.shape[0]         
        e_A = np.ones((self.m_A, 1))         
        self.m_B = B.shape[0]         
        e_B = np.ones((self.m_B, 1))          
        if self.kernel == None:             
            HA = np.hstack((A, e_A))             
            GB = np.hstack((B, e_B))             
            I = np.identity(n+1)         
        else:             
            HA = np.hstack((self.transform(A,self.C),e_A))             
            GB = np.hstack((self.transform(B,self.C),e_B))             
            I = np.identity(m+1)          
        ### class A         
        self.WA = []         
        self.bA = []         
        for i in range(self.k):             
            mAi = self.Z_A[i].shape[0]             
            eAi = np.ones((mAi, 1))             
            if self.kernel == None:                
                H_i = np.hstack((self.Z_A[i], eAi))               
                K = matrix(GB.dot(np.linalg.inv((H_i.T).dot(H_i) + self.c2*I)).dot(GB.T))                 
                q = matrix((-e_B))                 
                G = matrix(np.vstack((-np.eye(self.m_B), np.eye(self.m_B))))     
                h = matrix(np.vstack((np.zeros((self.m_B,1)), self.c1*np.ones((self.m_B,1)))))                 
                solvers.options['show_progress'] = False                 
                sol = solvers.qp(K, q, G, h)               
                alpha = np.array(sol['x'])                
                self.zi = np.linalg.inv(((H_i.T).dot(H_i) + self.c2*I)).dot(GB.T).dot(alpha)            
            else:                 
                H_i = np.hstack((self.transform(self.Z_A[i],self.C), eAi))          
                K = matrix(GB.dot(np.linalg.inv((H_i.T).dot(H_i) + self.c2*I)).dot(GB.T))              
                q = matrix((-e_B))              
                G = matrix(np.vstack((-np.eye(self.m_B), np.eye(self.m_B))))           
                h = matrix(np.vstack((np.zeros((self.m_B,1)), self.c1*np.ones((self.m_B,1)))))             
                solvers.options['show_progress'] = False      
                sol = solvers.qp(K, q, G, h)          
                alpha = np.array(sol['x'])       
                self.zi = np.linalg.inv(((H_i.T).dot(H_i) + self.c2*I)).dot(GB.T).dot(alpha)    
            wi = self.zi[:-1].ravel()  # => shape (n,)
            bi = float(self.zi[-1])    # => bias      
            self.WA.append(wi)      
            self.bA.append(bi)          
        ### class B         
        self.WB = []         
        self.bB = []         
        for j in range(self.l):             
            mBj = self.Z_B[j].shape[0]             
            eBj = np.ones((mBj, 1))           
            if self.kernel == None:                 
                G_j = np.hstack((self.Z_B[j], eBj))                 
                K = matrix(HA.dot(np.linalg.inv((G_j.T).dot(G_j) + self.c4*I)).dot(HA.T))     
                q = matrix(-e_A)         
                G = matrix(np.vstack((-np.eye(self.m_A),np.eye(self.m_A))))     
                h = matrix(np.vstack((np.zeros((self.m_A,1)), self.c3*np.ones((self.m_A,1)))))    
                solvers.options['show_progress'] = False       
                sol = solvers.qp(K, q, G, h)          
                gam = np.array(sol['x'])              
                self.zj = np.linalg.inv((G_j.T).dot(G_j) + self.c4*I).dot(HA.T).dot(gam) 
            else:           
                G_j = np.hstack((self.transform(self.Z_B[j],self.C), eBj))            
                K = matrix(HA.dot(np.linalg.inv((G_j.T).dot(G_j) + self.c4*I)).dot(HA.T))       
                q = matrix(-e_A)           
                G = matrix(np.vstack((-np.eye(self.m_A),np.eye(self.m_A))))        
                h = matrix(np.vstack((np.zeros((self.m_A,1)), self.c3*np.ones((self.m_A,1)))))      
                solvers.options['show_progress'] = False         
                sol = solvers.qp(K, q, G, h)           
                gam = np.array(sol['x'])          
                self.zj = np.linalg.inv((G_j.T).dot(G_j) + self.c4*I).dot(HA.T).dot(gam)     
            wj = self.zj[:-1].ravel()
            bj = float(self.zj[-1])   # => bias         
            self.WB.append(wj)         
            self.bB.append(bj)    
    def signum(self,X):        
         return np.ravel(np.where(X>=0,1,-1))
    # def project(self,X):
    #     scoreA = np.zeros(X.shape[0])
    #     scoreB = np.zeros(X.shape[0])
    #     score_arrayA = np.zeros((self.k,X.shape[0]))
    #     score_arrayB = np.zeros((self.l,X.shape[0]))
    #     if self.kernel== None:
    #         for i in range(self.k):
    #             scoreAi = ((self.Z_A[i].shape[0])/(self.m_A))*(np.dot(X,self.WA[i]) + self.bA[i]).ravel()
    #             score_arrayA[i] = scoreAi
    #         scoreA = np.sum(score_arrayA, axis = 0)
    #         for j in range(self.l):
    #             scoreBj = ((self.Z_B[j].shape[0])/(self.m_B))*(np.dot(X, self.WB[j]) + self.bB[j]).ravel()
    #             score_arrayB[j] = scoreBj
    #         scoreB = np.sum(score_arrayB, axis = 0)
    #     else:
    #         for i in range(self.k):
    #             scoreAi = np.zeros(X.shape[0])
    #             for j in range(X.shape[0]):
    #                 sA=0
    #                 for uj, ct in zip(self.WA[i], self.C):
    #                     sA += self.kf[self.kernel](X[j],ct)*uj
    #                 scoreAi[j] = sA + self.bA[i]
    #             scoreA += ((self.Z_A[i].shape[0])/(self.m_A))*scoreAi
    #         for i in range(self.l):
    #             scoreBi = np.zeros(X.shape[0])
    #             for j in range(X.shape[0]):
    #                 sB=0
    #                 for vj, ct in zip(self.WB[i], self.C):
    #                     sB += self.kf[self.kernel](X[j],ct)*vj
    #                 scoreBi[j] = sB + self.bB[i]
    #             scoreB += ((self.Z_B[i].shape[0])/(self.m_B))*scoreBi

    #     score = scoreB - scoreA
    #     return score

    # def predict(self,X):
    #     return self.signum(self.project(X))
    # def score(self, X, y):
    #     return 100*np.mean(self.predict(X)==y)      
    
    ###########################################
    def project(self, X):
        """
        Tính giá trị quyết định (score) cho từng mẫu trong X.
        Dữ liệu X có shape (num_samples, n), 
        trong khi W và b đã được tách riêng (W shape (n,), b là vô hướng).
        """
        scoreA = np.zeros(X.shape[0])  # Tổng điểm 'nghiêng về A' cho mỗi mẫu
        scoreB = np.zeros(X.shape[0])  # Tổng điểm 'nghiêng về B' cho mỗi mẫu

        # Mảng tạm để lưu đóng góp của từng cụm A/B
        score_arrayA = np.zeros((self.k, X.shape[0]))
        score_arrayB = np.zeros((self.l, X.shape[0]))

        if self.kernel is None:
            # ---- Trường hợp không dùng kernel ----
            # => Tính score = X*w_i + b_i
            for i in range(self.k):
                # self.WA[i] và self.bA[i] đã được tách bias; 
                # WA[i] có shape (n,), bA[i] vô hướng
                scoreAi = ( 
                    (self.Z_A[i].shape[0] / self.m_A) * 
                    (np.dot(X, self.WA[i]) + self.bA[i]) 
                )
                score_arrayA[i] = scoreAi

            scoreA = np.sum(score_arrayA, axis=0)

            for j in range(self.l):
                scoreBj = (
                    (self.Z_B[j].shape[0] / self.m_B) *
                    (np.dot(X, self.WB[j]) + self.bB[j])
                )
                score_arrayB[j] = scoreBj

            scoreB = np.sum(score_arrayB, axis=0)

        else:
            # ---- Trường hợp có kernel (linear, polynomial, rbf) ----
            # Lúc này, WA[i] và bA[i] vẫn tách riêng, nhưng WA[i] là
            # trọng số ứng với 'transform' (kích thước ~ m), 
            # còn bA[i] là bias vô hướng.
            for i in range(self.k):
                scoreAi = np.zeros(X.shape[0])
                for idx in range(X.shape[0]):
                    sA = 0
                    # self.WA[i] ~ vector (m,), 
                    # self.C ~ tập hợp A + B, 
                    # self.kf[self.kernel](X[idx], ct) tính kernel
                    for uj, ct in zip(self.WA[i], self.C):
                        sA += self.kf[self.kernel](X[idx], ct) * uj
                    scoreAi[idx] = sA + self.bA[i]

                # Nhân thêm tỷ lệ cụm
                scoreA += (self.Z_A[i].shape[0] / self.m_A) * scoreAi

            for i in range(self.l):
                scoreBi = np.zeros(X.shape[0])
                for idx in range(X.shape[0]):
                    sB = 0
                    for vj, ct in zip(self.WB[i], self.C):
                        sB += self.kf[self.kernel](X[idx], ct) * vj
                    scoreBi[idx] = sB + self.bB[i]

                scoreB += (self.Z_B[i].shape[0] / self.m_B) * scoreBi

        # "score" = (mức độ nghiêng về B) - (mức độ nghiêng về A)
        score = scoreB - scoreA
        return score
    
    def plot_clusters_and_planes(self):
        """
        Minh họa các cụm và siêu phẳng (đường) cho trường hợp dữ liệu 2D.
        self: đối tượng mô hình TSVM_MCD, đã fit xong.

        Giả sử:
        - self.Z_A[i] chứa mảng các điểm thuộc cụm i của lớp A (shape (mAi, 2))
        - self.WA[i], self.bA[i] chứa w_i, b_i cho cụm A_i.
        - self.Z_B[j], self.WB[j], self.bB[j] tương tự cho lớp B.
        """
        # ===== 1) Vẽ dữ liệu =====
        plt.figure(figsize=(8, 6))

        # Tạo bảng màu/marker tùy ý cho A, B.
        colorsA = ['tab:orange','tab:green','tab:blue','tab:brown','tab:olive']
        colorsB = ['tab:red','tab:purple','tab:gray','tab:pink','tab:cyan']
        
        # Marker: hình vuông cho A, hình tròn cho B.
        # bạn có thể tùy biến
        markerA = 's'
        markerB = 'o'
        
        # Vẽ lần lượt từng cụm A
        for i in range(self.k):
            Xi = self.Z_A[i]              # (mAi, 2)
            plt.scatter(Xi[:,0], Xi[:,1],
                        color=colorsA[i % len(colorsA)],
                        marker=markerA,
                        label=f'Cluster A{i+1}')

        # Vẽ lần lượt từng cụm B
        for j in range(self.l):
            Xj = self.Z_B[j]              # (mBj, 2)
            plt.scatter(Xj[:,0], Xj[:,1],
                        color=colorsB[j % len(colorsB)],
                        marker=markerB,
                        label=f'Cluster B{j+1}')

        # ===== 2) Vẽ các siêu phẳng =====
        # Lấy min-max trục x để vẽ đường
        all_points = np.vstack([np.vstack(self.Z_A), np.vstack(self.Z_B)])
        x_min, x_max = all_points[:,0].min()-1, all_points[:,0].max()+1
        xs = np.linspace(x_min, x_max)

        # 2.1) Siêu phẳng lớp A (nét liền)
        line_colorsA = ['blue','green','orange','brown','olive']  # Tùy chọn
        for i in range(self.k):
            w = self.WA[i]  # w = (w0, w1) giả sử 2D
            b = self.bA[i]
            # Kiểm tra w[1] != 0
            if abs(w[1]) > 1e-9:
                ys = -(b + w[0]*xs)/w[1]
                plt.plot(xs, ys,
                        color=line_colorsA[i % len(line_colorsA)],
                        label=f'f{i+1}+ = 0')  # f{i+1}+ = 0

        # 2.2) Siêu phẳng lớp B (nét đứt)
        line_colorsB = ['red','purple','gray','pink','cyan']  # Tùy chọn
        for j in range(self.l):
            w = self.WB[j]
            b = self.bB[j]
            if abs(w[1]) > 1e-9:
                ys = -(b + w[0]*xs)/w[1]
                plt.plot(xs, ys, '--',
                        color=line_colorsB[j % len(line_colorsB)],
                        label=f'f{j+1}- = 0')  # f{j+1}- = 0

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('TSVM MCD: Clusters & Decision Boundaries')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        scores = self.project(X)    # mảng shape (num_samples,)
        return self.signum(scores)  
    def score(self, X, y):         # Tính phần trăm mẫu dự đoán đúng theo cách mới         
        y_pred = self.predict(X)         
        return 100.0 * np.mean(y_pred == y)
