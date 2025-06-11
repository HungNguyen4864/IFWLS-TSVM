import os
import numpy as np
from Pin_GBTSVM import Pin_GBTSVM
from gen_ball import gen_balls

directory = 'D:/LGBTSVM_MCD/Pin-GBTSVM-main/Pin-GBTSVM/Data'
file_list = os.listdir(directory)
if __name__ == '__main__':
    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
       
            print(file_name)
            file_data = np.loadtxt(file_path, delimiter=',')
        
            m, n = file_data.shape
            for i in range(m):
                if file_data[i, n-1] == 0:
                    file_data[i, n-1] = -1
    
            np.random.seed(0)
            indices = np.random.permutation(m)
            file_data = file_data[indices]
            A_train=file_data[0:int(m*(1-0.30))]
            A_test=file_data[int(m * (1-0.30)):]
        
            pur = 1 - (0.015 * 5)                      
            num = 4
            c1=0.1
            c2=0.1
            tau=1
                     
            A_train = gen_balls(A_train, pur=pur, delbals=num)
        
            Radius=[]
            for i in A_train:
                Radius.append(i[1])
            Center=[]
            for ii in A_train:
                Center.append(ii[0])
            Label=[]
            for iii in A_train:
                Label.append(iii[-1])
            Radius=np.array(Radius)
            Center=np.array(Center)
            Label=np.array(Label)
            Z_train=np.hstack((Center,Radius.reshape(Radius.shape[0], 1)))
            Lab=Label.reshape(Label.shape[0], 1)
            A_train=np.hstack((Z_train,Lab))

            Test_accuracy, Test_time = Pin_GBTSVM(A_train, A_test, c1, c2, tau)
            print(Test_accuracy)
