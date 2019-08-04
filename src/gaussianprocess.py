import numpy as np



class GaussianProcess(object):
    ### Constructor ###
    def __init__(self, xtrain, ytrain, param=[0.3, 0.0]):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.param = param
        self.noise = 1e-5

    ### RBF Kernel ###
    @staticmethod
    def kernel(a, b, param):

        c = np.zeros((a.shape[0], b.shape[0]))
        np.fill_diagonal(c, param[1]**2)
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1/param[0]) * sqdist) + c

    ### Predict new values ###
    def predict(self, xtest, param=None):

        param = self.param if param is None else param

        K_ss = GaussianProcess.kernel(xtest, xtest, param)
        K_s  = GaussianProcess.kernel(self.xtrain, xtest, param)
        K    = GaussianProcess.kernel(self.xtrain, self.xtrain, param)

        L = np.linalg.cholesky(K + self.noise*np.eye(self.xtrain.shape[0]))
        Lk = np.linalg.solve(L, K_s)

        z = np.linalg.solve(L, self.ytrain)
        mu = np.dot(Lk.T, z).reshape((-1,))
        var = np.linalg.cholesky(K_ss + self.noise*np.eye(xtest.shape[0]) - np.dot(Lk.T, Lk))

        ### P value ###
        logP = -0.5*np.dot(z.T,z).reshape(1) - np.trace(np.log(L)).reshape(1)

        return mu, var, logP[0]

    ### Hyperparameter fitting with GD ###
    def tuneParameters(self, xtest, p0, max_iters=2000, learn_rate=0.01, delta=0.001, eps=1e-4, plot=False):
        p = np.asarray(p0).reshape(-1)
        logPs = []
        for i in range(max_iters):
            grady = np.zeros_like(p)

            for i in range(p.shape[0]):
                dx = np.zeros_like(p)
                dx[i] = delta
                _, _, yf = self.predict(xtest, param=p+dx)
                _, _, yb = self.predict(xtest, param=p-dx)
                grady[i] = (yf-yb)/(2*delta)

            step = (learn_rate*grady).reshape(-1)
            p += step

            ### Get current value ###
            _, _, y = self.predict(xtest, param=p)
            logPs.append(y)
            print("new logp: {}, p: {}".format(y, p))

            if np.all(step<eps):
                print("No further improvement observed!")
                break
            elif i == max_iters-1:
                print("Maximum iteration number reached!")

        ### Assign optimial parameters to param ###
        self.param = p

        ### Plot ###
        if plot:
            plt.plot(range(len(logPs)), logPs, lw=3)
            plt.grid(True)
            plt.xlabel("Iterations")
            plt.ylabel("log P(y|x,p)")
            plt.show()



### TEST ###
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if True:
        Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
        ytrain = np.sin(Xtrain)
        Xtest = np.linspace(-5, 5, 50).reshape(-1,1)

        gp = GaussianProcess(Xtrain, ytrain, param=[0.8,0.5])
        #gp.tuneParameters(Xtest, p0=[0.8,0.1], plot=True, eps=0.001)
        YtestHat, Yvar, logP = gp.predict(Xtest)

        print(Yvar.shape)

        plt.plot(Xtrain, ytrain, 'bs', ms=8)
       # plt.gca().fill_between(Xtest.flat, YtestHat-2*Yvar, YtestHat+2*Yvar, color="#dddddd")
        plt.plot(Xtest, YtestHat, 'r--', lw=2)
        plt.axis([-5, 5, -3, 3])
        plt.grid(True)
        plt.show()

    if False:
        np.random.seed(42)
        ### 2D Test problem ###
        def rosenbrock(X):
            return (1-X[:,0])**2 + 100*(X[:,1]-X[:,0]**2)**2

        
        Xtrain = np.asarray([-2,-1]) + np.random.rand(50, 2)*(np.asarray([2,3])-np.asarray([-2,-1]))
        Ytrain = rosenbrock(Xtrain).reshape(-1,1)

        XG,YG = np.meshgrid(np.linspace(-2,2,20),np.linspace(-1,3,20))
        Xtest = np.vstack((XG.flatten(), YG.flatten())).T
        ZG = rosenbrock(Xtest).reshape(-1,20)

        zmin,zmax = ZG.min(),ZG.max()
        ZG = (ZG-zmin)/(zmax-zmin)
        Ytrain = (Ytrain-zmin)/(zmax-zmin)

        ### Initialiaze GP ###
        gp = GaussianProcess(Xtrain, Ytrain, 0.8)
        gp.tuneParameters(Xtest, p0=[0.8,1.0], plot=True, eps=0.01)
        YtestHat, Yvar, logP = gp.predict(Xtest)


        ### The plotting ###
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(XG, YG, ZG, color='k')
        ax.plot(Xtrain[:,0], Xtrain[:,1], Ytrain[:,0],'ro', markersize=12)
        ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), YtestHat.reshape(20,20), color='r')


        #for n, color in zip(range(NDRAWS),('g','m', 'b')):
        #    ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), f_post[:,n].reshape(20,20), color=color)

        plt.show()