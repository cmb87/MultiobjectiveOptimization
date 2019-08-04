import numpy as np
import matplotlib.pyplot as plt
import sys  
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(X):
    return (1-X[:,0])**2 + 100*(X[:,1]-X[:,0]**2)**2

# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

### 2D Test problem ###
Xtrain = np.asarray([-2,-1]) + np.random.rand(50, 2)*(np.asarray([2,3])-np.asarray([-2,-1]))
Ytrain = rosenbrock(Xtrain).reshape(-1,1)

XG,YG = np.meshgrid(np.linspace(-2,2,20),np.linspace(-1,3,20))
Xtest = np.vstack((XG.flatten(), YG.flatten())).T
ZG = rosenbrock(Xtest).reshape(-1,20)

zmin,zmax = ZG.min(),ZG.max()
ZG = (ZG-zmin)/(zmax-zmin)
Ytrain = (Ytrain-zmin)/(zmax-zmin)
### GP parameters ###
param = 0.3

### prior ###
NDRAWS = 3
K_ss = kernel(Xtest, Xtest, param)
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(Xtest.shape[0]))

N = np.random.normal(size=(Xtest.shape[0],NDRAWS))
f_prior = 0.0 + np.dot(L, N)

### Posterior ###
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)

mu = np.dot(Lk.T, np.linalg.solve(L, Ytrain)).reshape((-1,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)

# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(400) - np.dot(Lk.T, Lk))
# xposterior = mu_posterior + sqrt(sigma)*N
N = np.random.normal(size=(400,3))

f_post = mu.reshape(-1,1) + 1.0*np.dot(L, N)


### The plotting ###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(XG, YG, ZG, color='k')
ax.plot(Xtrain[:,0], Xtrain[:,1], Ytrain[:,0],'ro', markersize=12)
ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), mu.reshape(20,20), color='r')


#for n, color in zip(range(NDRAWS),('g','m', 'b')):
#    ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), f_post[:,n].reshape(20,20), color=color)

plt.show()





sys.exit()











# Test data
n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

param = 0.3
K_ss = kernel(Xtest, Xtest, param)

# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample 3 sets of standard normals (rand) for our test points,
# multiply them by the square root of the covariance matrix == Cholesky!
N = np.random.normal(size=(n,3))
f_prior = 0.0 + np.dot(L, N)
# xprior = mu + sqrt(sigma)*N


# Now let's plot the 3 sampled functions.
plt.plot(Xtest, f_prior)
plt.axis([-5, 5, -3, 3])
plt.title('Three samples from the GP prior')
plt.show()



# Noiseless training data
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = np.sin(Xtrain)

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)

# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
# xposterior = mu_posterior + sqrt(sigma)*N
N = np.random.normal(size=(n,3))
f_post = mu.reshape(-1,1) + np.dot(L, N)

plt.plot(Xtrain, ytrain, 'bs', ms=8)
plt.plot(Xtest, f_post)
plt.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
plt.plot(Xtest, mu, 'r--', lw=2)
plt.axis([-5, 5, -3, 3])
plt.title('Three samples from the GP posterior')
plt.show()
