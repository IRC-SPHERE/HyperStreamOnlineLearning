import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.lines import Line2D

np.random.seed(42)

n_samples = 5000
MU = np.array([0.5, 1.5])
COV = np.array([[1., 0.7], [0.7, 2.]])

def get_samples(n):
    return np.random.multivariate_normal(mean=MU, cov=COV, size=n)

class BackgroundCheck(object):
    def __init__(self, model):
        self.model = model

    def fit(self, x):
        self.model.fit(x)

    def prob_foreground(self, x):
        l = self.model.likelihood(x)
        l_max = self.model.max
        return np.true_divide(l, l_max)

    def prob_background(self, x):
        return 1 - self.prob_foreground(x)

    def predict_proba(self, x):
        return self.prob_background(x)


class GaussianEstimation(object):
    def __init__(self):
        self.mu = None
        self.cov = None
        self.N = 0

    def fit(self, x):
        N = x.shape[1]
        mu = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)

        if self.N is 0:
            self.N = N
            self.mu = mu
            self.k = len(mu)
            self.cov = cov
        else:
            self.mu = np.true_divide((self.mu * self.N) + (mu * N), self.N + N)
            self.cov = np.true_divide((self.cov * self.N) + (cov * N), self.N + N)
            self.N += N

    def likelihood(self, x):
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x):
        x_mu = x - self.mu
        # a = np.array([[1, 2]])
        # b = np.array([[1, 2],[3,4]])
        # np.inner(np.inner(a, b.T), a)
        inverse = np.linalg.inv(self.cov)
        exp = np.array([np.inner(np.inner(a, inverse.T), a) for a in x_mu])
        return - 0.5 * (
                    np.log(np.linalg.det(self.cov))
                    + exp \
                    + self.k * np.log(2*np.pi)
                    )

    @property
    def max(self):
        return self.likelihood(self.mu.reshape(1,-1))


model = BackgroundCheck(GaussianEstimation())
for i in range(n_samples/2):
    x = get_samples(2)
    model.fit(x)

x = get_samples(n_samples)

p_foreground = 1 - model.predict_proba(x)
fig = plt.figure('scatter')
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], p_foreground)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('p_foreground')
fig.savefig('p_foreground_x.svg')


X = np.linspace(min(x[:,0]), max(x[:,0]), 30)
Y = np.linspace(min(x[:,1]), max(x[:,1]), 30)
X, Y = np.meshgrid(X, Y)

grid = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis=1)
p_foreground = 1 - model.predict_proba(grid).reshape(X.shape[0], X.shape[1])

fig = plt.figure('surface')
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, p_foreground, cmap=cm.coolwarm)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('p_foreground')
fig.savefig('p_foreground_grid.svg')
