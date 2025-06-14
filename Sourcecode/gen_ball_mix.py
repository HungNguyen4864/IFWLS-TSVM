import csv
import time
import pandas as pd
import numpy as np
import json
import csv
# import sklearn.cluster.k_means_ as KMeans
from sklearn.cluster import KMeans
import warnings
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

warnings.filterwarnings("ignore")  # ignore warning


class GranularBall:
    def __init__(self, data):
        self.data = data[:, :]
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity ,self.r= self.__get_label_and_purity_and_r()

    def __get_label_and_purity_and_r(self):
        count = Counter(self.data[:, -2])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        arr=np.array(self.data_no_label)-self.center
        ar=np.square(arr)
        a=np.sqrt(np.sum(ar,1))
        r=np.sum(a)/len(self.data_no_label)
        return label, purity,r

    def split_2balls(self):
        clu=KMeans(n_clusters=2).fit(self.data_no_label)
        label_cluster=clu.labels_
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :]) 
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2

class GBList:
    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, purity=1, min_sample=1):
        ll = len(self.granular_balls)
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_r(self):
        return np.array(list(map(lambda x: x.r, self.granular_balls)))

    def get_data(self):
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)
    def del_ball(self,purty=0.,num_data=0):
        T_ball=[]
        for ball in self.granular_balls:
            if ball.purity>=purty and ball.num>=num_data:
                T_ball.append(ball)
        self.granular_balls=T_ball.copy()
        self.data=self.get_data()
    def re_division(self, i):
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        clu = KMeans(n_clusters=k, init=self.get_center()[:, attributes], max_iter=1).fit(self.data[:, attributes])
        label_cluster = clu.labels_
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division
    
def generate_ball_data(data, pur, delbals):
    num, dim = data[:, :-1].shape
    index = np.array(range(num)).reshape(num, 1)  # index column
    data = np.hstack((data, index))  # add index column
    gb = GBList(data)
    gb.init_granular_balls(purity=pur)
    gb.del_ball(num_data=delbals)

    centers = gb.get_center().tolist()
    rs = gb.get_r().tolist()

    ball_data = []
    for i in range(len(gb.granular_balls)):
        ball_info = {
            "center": centers[i],
            "radius": rs[i],
            "label": gb.granular_balls[i].label,
            "purity": gb.granular_balls[i].purity,
            "size": gb.granular_balls[i].num
        }
        ball_data.append(ball_info)
    return ball_data
def gen_balls(data, pur, delbals):
    balls = generate_ball_data(data, pur=pur, delbals=delbals)
    simplified_balls = [
        [ball["center"], ball["radius"], ball["purity"], ball["size"],ball["label"]]
        for ball in balls
    ]
    return simplified_balls
