import numpy as np

class NonLinearScoreValues:
    def __init__(self, A, mew):
        self.A = A
        self.mew = mew

    def compute_scores(self):
        no_input, no_col = self.A.shape

        A1 = self.A[self.A[:, -1] == 1, :-1]
        B1 = self.A[self.A[:, -1] != 1, :-1]

        # Compute kernels K1, K2, and K3
        K1 = self._compute_kernel(A1, A1)
        K2 = self._compute_kernel(B1, B1)
        A_temp = self.A[:, :-1]
        K3 = self._compute_kernel(A_temp, A_temp)

        # Compute radii
        radiusxp = np.sqrt(1 - 2 * np.mean(K1, axis=1) + np.mean(K1))
        radiusmaxxp = np.max(radiusxp)
        radiusxn = np.sqrt(1 - 2 * np.mean(K2, axis=1) + np.mean(K2))
        radiusmaxxn = np.max(radiusxn)

        alpha_d = max(radiusmaxxn, radiusmaxxp)

        mem1 = 1 - (radiusxp / (radiusmaxxp + 1e-4))
        mem2 = 1 - (radiusxn / (radiusmaxxn + 1e-4))

        # Compute ro values
        DD = np.sqrt(2 * (1 - K3))
        ro = []

        for i in range(no_input):
            temp = DD[i, :]
            B1 = self.A[temp < alpha_d, :]
            x3, _ = B1.shape
            count = np.sum(self.A[i, -1] * np.ones(B1.shape[0]) != B1[:, -1])
            x5 = count / x3
            ro.append(x5)

        A2 = np.column_stack((self.A[:, -1], ro))
        ro2 = A2[A2[:, 0] == -1, 1]
        ro1 = A2[A2[:, 0] != -1, 1]
        v1 = (1 - mem1) * ro1
        v2 = (1 - mem2) * ro2

        S1, S2 = [], []
        for i in range(len(v1)):
            S1.append(self._compute_score(mem1[i], v1[i]))
        for i in range(len(v2)):
            S2.append(self._compute_score(mem2[i], v2[i]))

        return np.array(S1), np.array(S2), alpha_d, self.mew

    def _compute_kernel(self, X, Y):
        dist_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-dist_sq / (self.mew ** 2))

    def _compute_score(self, mem, v):
        if v == 0:
            return mem
        elif mem <= v:
            return 0
        else:
            return (1 - v) / (2 - mem - v)

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 1, 1], [2, 2, -1], [3, 3, 1], [4, 4, -1]])
    mew = 2
    nlsv = NonLinearScoreValues(A, mew)
    S1, S2, alpha_d, mew_computed = nlsv.compute_scores()
    print("S1:", S1)
    print("S2:", S2)
    print("Alpha_d:", alpha_d)
    print("Mew:", mew_computed)
