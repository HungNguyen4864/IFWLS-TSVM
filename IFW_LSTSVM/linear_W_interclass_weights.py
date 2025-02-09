import numpy as np
from scipy.spatial.distance import pdist, squareform

class LinearWInterclassWeights:
    def __init__(self, A, mew, k):
        self.A = A
        self.mew = mew
        self.k = k

    def compute_weights(self):
        # Separate the data based on the class label
        A1 = self.A[self.A[:, -1] == 1, :-1]
        B1 = self.A[self.A[:, -1] != 1, :-1]

        # Calculate weights for A1
        ro1 = self._calculate_weight(A1)

        # Calculate weights for B1
        ro2 = self._calculate_weight(B1)

        return ro1, ro2

    def _calculate_weight(self, data):
        # Compute pairwise distances and sort them
        D = pdist(data)
        Z = squareform(D)
        sorted_indices = np.argsort(Z, axis=1)
        sorted_distances = np.take_along_axis(Z, sorted_indices, axis=1)

        # Select the k nearest neighbors excluding the first column which is zero
        aak = sorted_distances[:, 1:1 + self.k]
        
        # Compute the exponential weight
        exaak = np.exp(-(aak ** 2) / self.mew)
        weights = np.sum(exaak, axis=1)
        
        return weights

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 1, 1], [2, 2, 1], [3, 3, -1], [4, 4, -1]])
    mew = 2
    k = 1
    liw = LinearWInterclassWeights(A, mew, k)
    ro1, ro2 = liw.compute_weights()
    print("ro1:", ro1)
    print("ro2:", ro2)
