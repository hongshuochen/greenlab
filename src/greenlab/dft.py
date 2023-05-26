import numpy as np
import multiprocessing as mp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


class DiscriminantFeatureTest:
    def __init__(self, max_depth=1, n_jobs=-1):
        self.max_depth = max_depth
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.n_jobs == 1:
            self.loss = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(
                    X[:, i : i + 1], y
                )
                prob = tree.predict_proba(X[:, i : i + 1])
                self.loss[i] = log_loss(y, prob)
        else:
            with mp.Pool(self.n_jobs) as pool:
                self.loss = np.array(
                    pool.map(
                        self._fit, [(X[:, i : i + 1], y) for i in range(X.shape[1])]
                    )
                )
        self.rank = np.argsort(self.loss)
        self.sorted_loss = self.loss[self.rank]
        return self

    def _fit(self, args):
        X, y = args
        tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y)
        prob = tree.predict_proba(X)
        return log_loss(y, prob)

    def select(self, X, n):
        return X[:, self.rank[:n]]

    def plot_loss(self, sorted=True, path=None):
        if sorted:
            plt.plot(self.sorted_loss)
        else:
            plt.plot(self.loss)
        plt.xlabel("Feature Index")
        plt.ylabel("Loss")
        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100000, n_features=1000, n_informative=2, n_classes=2
    )
    dft = DiscriminantFeatureTest(max_depth=1)
    dft.fit(X, y)
    dft.plot_loss()
    dft.plot_loss(sorted=False)

    dft = DiscriminantFeatureTest(max_depth=3)
    dft.fit(X, y)
    dft.plot_loss()
    dft.plot_loss(sorted=False)
