from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt