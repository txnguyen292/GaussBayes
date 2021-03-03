import numpy as np
from scipy.stats import multivariate_normal as mvn


class GaussNB():
    """Naive Bayes model"""
    def __init__(self):
        pass

    def fit(self, X:np.array, y:np.array, epsilon:int=1e-3) -> None:
        self.likelihoods = dict()
        self.priors = dict()

        self.K = set(y.astype(int)) # list of classes

        for k in self.K:
            X_k = X[y==k,:]
            self.likelihoods[k] = {"mean": X_k.mean(axis=0), "cov": X_k.var(axis=0) + epsilon}
            self.priors[k] = len(X_k) / len(X)

    def predict(self, X: np.array) -> np.array:
        N, D = X.shape

        P_hat = np.zeros((N, len(self.K)))

        for k, l in self.likelihoods.items():
            P_hat[:, k] = mvn.logpdf(X, l["mean"], l["cov"] + np.log(self.priors[k]))
        
        return P_hat.argmax(axis=1)


class GaussBayes:

    def fit(self, X:np.array, y:np.array, epsilon:int = 1e-3) -> None:
        self.likelihoods = dict()
        self.priors = dict()

        self.K = set(y.astype(int))

        for k in self.K:
            X_k = X[y==k,:]
            N_k, D = X_k.shape
            mu_k = X_k.mean(axis=0)
            self.likelihoods[k] = {"mean": X_k.mean(axis=0), "cov": (1 / (N_k - 1)) * np.matmul((X_k-mu_k).T, X_k - mu_k) + epsilon*np.identity(D)} 

            self.priors[k] = len(X_k) / len(X)
    
    def predict(self, X:np.array) -> np.array:
        N, D = X.shape
        P_hat = np.zeros((N, len(self.K)))

        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])

        return P_hat.argmax(axis=1)

