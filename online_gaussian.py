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

mu = []
cov = []

gaussian = GaussianEstimation()
for i in range(n_samples/2):
    x = get_samples(2)
    gaussian.fit(x)
    mu.append(gaussian.mu)
    cov.append(gaussian.cov)

mu = np.array(mu)
cov = np.array(cov)

fig = plt.figure('mu')
fig.clf()
ax = fig.add_subplot(111)
for i in range(2):
    p = ax.plot(mu[:,i], label='$\mu_{}$'.format(i))
    ax.add_line(Line2D([0, n_samples/2], [MU[i], MU[i]], linewidth=0.5,
        color=p[0].get_color()))
for i in range(2):
    for j in range(2):
        p = ax.plot(cov[:,i,j], label='cov[{0},{1}]'.format(i,j))
        ax.add_line(Line2D([0, n_samples/2], [COV[i,j], COV[i,j]],
                           linewidth=0.5, color=p[0].get_color()))
ax.set_xlabel('samples/2')
ax.set_ylabel('estimation')
ax.legend()
fig.savefig('mus.svg')

x = get_samples(n_samples)

l = gaussian.likelihood(x)
fig = plt.figure('scatter')
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], l)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('likelihood')
fig.savefig('likelihood_x.svg')


X = np.linspace(min(x[:,0]), max(x[:,0]), 30)
Y = np.linspace(min(x[:,1]), max(x[:,1]), 30)
X, Y = np.meshgrid(X, Y)

grid = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis=1)
l = gaussian.likelihood(grid).reshape(X.shape[0], X.shape[1])

fig = plt.figure('surface')
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, l, cmap=cm.coolwarm)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('likelihood')
fig.savefig('likelihood_grid.svg')
