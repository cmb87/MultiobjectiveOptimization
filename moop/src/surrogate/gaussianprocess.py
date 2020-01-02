import numpy as np



class GaussianProcess(object):
    ### Constructor ###
    def __init__(self, xtrain, ytrain, param=None, noise=0.0):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.param = 0.1*np.ones(xtrain.shape[1]) if param is None else np.asarray(param)
        self.noiseParam = noise


    ### RBF Kernel ###
    @staticmethod
    def kernel(a, b, param, noisevar):
        ### Noise ####
        c = np.zeros((a.shape[0], b.shape[0]))
        np.fill_diagonal(c, noisevar**2)
        ### Kernel fct ###
        a=a/np.abs(param)
        b=b/np.abs(param)
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * sqdist) + c

        #sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        #return np.exp(-.5 * (1/param[0]) * sqdist) + c

    ### Update Training data ###
    def updateTrainingData(self, xtrainNew, ytrainNew):
        self.xtrain = np.vstack((self.xtrain, xtrainNew))
        self.ytrain = np.vstack((self.ytrain, ytrainNew))

    ### Predict new values ###
    def predict(self, xtest, param=None):

        param = self.param if param is None else param

        K_ss = GaussianProcess.kernel(xtest, xtest, param, self.noiseParam)
        K_s  = GaussianProcess.kernel(self.xtrain, xtest, param, self.noiseParam)
        K    = GaussianProcess.kernel(self.xtrain, self.xtrain, param, self.noiseParam)

        L = np.linalg.cholesky(K + 1e-5*np.eye(self.xtrain.shape[0]))
        Lk = np.linalg.solve(L, K_s)

        z = np.linalg.solve(L, self.ytrain)

        mu = np.dot(Lk.T, z).reshape((-1,))
        var = np.linalg.cholesky(K_ss + 1e-5*np.eye(xtest.shape[0]) - np.dot(Lk.T, Lk))

        # Compute the standard deviation so we can plot it
        s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)

        ### P value ###
        logP = -0.5*np.dot(z.T,z).reshape(1) - np.trace(np.log(np.abs(L)+1e-10)).reshape(1)

        return mu, var, s2**0.5, logP[0]

    ### cross-validated ###
    def crossvalidated(self, p0=None, kfold=4, **args):
        
        p0 = self.param.copy() if p0 is None else np.asarray(p0)
        size = int(self.xtrain.shape[0]/kfold)

        self.xtrain0 = self.xtrain.copy()
        self.ytrain0 = self.ytrain.copy()
        p_opti = []

        print("Starting crossvalidation")
        for k in range(kfold):
            index = np.ones(self.xtrain0.shape[0], dtype=bool)
            index[k*size:(k+1)*size] = False

            self.xtrain = self.xtrain0[index,:]
            self.ytrain = self.ytrain0[index,:]
            xtest = self.xtrain0[~index,:]
            ytest = self.ytrain0[~index,:]

            ### Tune system ###
            p_opti.append(self.tuneParameters(xtest, p0, **args))

            ### Calc Error ###
            yhat,_,_,_ = self.predict(xtest)
            rms = np.mean((yhat-ytest)**2)
            print("-> kfold: {}/{}, train/test: {}/{}, RMSE: {:.5}".format(k+1, kfold,self.xtrain.shape[0] ,xtest.shape[0] , rms))

        p_opti = np.asarray(p_opti)
        self.param = np.mean(p_opti, axis=0)
        self.xtrain = self.xtrain0
        self.ytrain = self.ytrain0
        del self.xtrain0, self.ytrain0


    ### Hyperparameter fitting with GD ###
    def tuneParameters(self, xtest, p0, max_iters=2000, learn_rate=0.01, delta=0.001, eps=1e-4, plot=False, verbose=False, iterLearnRateReduce=100):
        p = np.asarray(p0).reshape(-1).copy()
        logPs, yold = [], -1e+10

        for i in range(max_iters):

            ### Calc gradient ###
            grady = np.zeros_like(p)
            for k in range(p.shape[0]):
                dx = np.zeros_like(p)
                dx[k] = delta
                _, _, _, yf = self.predict(xtest, param=p+dx)
                _, _, _, yb = self.predict(xtest, param=p-dx)
                grady[k] = (yf-yb)/(2*delta)

            if i > iterLearnRateReduce:
                learn_rate*= 0.99

            step = (learn_rate*grady).reshape(-1)
            p += step
            p = np.abs(p)

            ### Get current value ###
            _, _, _, y = self.predict(xtest, param=p)
            logPs.append(y)

            ### Verbose ###
            if verbose:
                print("Iter: {}, LR: {:.2E}, logP: {:.2E}, stepmax: {:.2E}, params: {}".format(i, learn_rate, y, np.max(np.abs(step)), p))

            ### Stopping conditions ###
            if np.all(np.abs(step)<eps):
                if verbose:
                    print("No further improvement observed!")
                break
            elif i == max_iters-1:
                if verbose:
                    print("Maximum iteration number reached!")

            if y < yold:
                learn_rate *= 0.5

            yold = y
        ### Assign optimial parameters to param ###
        self.param = p

        ### Plot ###
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(range(len(logPs)), logPs, lw=3)
            plt.grid(True)
            plt.xlabel("Iterations")
            plt.ylabel("log P(y|x,p)")
            plt.show()

        ### Return optimized parameters ###
        return p




### TEST ###
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if True:
        Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
        ytrain = np.sin(Xtrain)
        Xtest = np.linspace(-5, 5, 50).reshape(-1,1)

        gp = GaussianProcess(Xtrain, ytrain, param=[0.8], noise=0.0)
        #gp.tuneParameters(Xtest, p0=[0.8,0.1], plot=True, eps=0.001)

        gp.crossvalidated(kfold=5, plot=False, eps=0.0001, verbose=False)

        YtestHat, Yvar, sigma, logP = gp.predict(Xtest)
        

        [plt.plot(Xtest, YtestHat.reshape(-1,1) + np.dot(Yvar,np.random.normal(size=(50,1)))) for _ in range(10)]

        
        plt.gca().fill_between(Xtest.flat, YtestHat-3*sigma, YtestHat+3*sigma, color="#dddddd")
        plt.plot(Xtest, YtestHat, 'r--', lw=5)
        plt.plot(Xtrain, ytrain, 'bs', ms=8)
        plt.axis([-5, 5, -3, 3])
        plt.grid(True)
        plt.show()

    if True:
        np.random.seed(87)
        ### 2D Test problem ###
        def rosenbrock(X):
            return (1-X[:,0])**2 + 100*(X[:,1]-X[:,0]**2)**2

        Xtrain = np.asarray([-2,-1]) + np.random.rand(18, 2)*(np.asarray([2,3])-np.asarray([-2,-1]))
        Ytrain = rosenbrock(Xtrain).reshape(-1,1)

        XG,YG = np.meshgrid(np.linspace(-2,2,20),np.linspace(-1,3,20))
        Xtest = np.vstack((XG.flatten(), YG.flatten())).T
        ZG = rosenbrock(Xtest).reshape(-1,20)

        zmin,zmax = ZG.min(),ZG.max()
        ZG = (ZG-zmin)/(zmax-zmin)
        Ytrain = (Ytrain-zmin)/(zmax-zmin)

        ### Initialiaze GP ###
        gp = GaussianProcess(Xtrain, Ytrain, noise=0.00)

        XtrainNew = np.asarray([-2,-1]) + np.random.rand(2, 2)*(np.asarray([2,3])-np.asarray([-2,-1]))
        YtrainNew = rosenbrock(XtrainNew).reshape(-1,1)

        gp.updateTrainingData(XtrainNew, (YtrainNew-zmin)/(zmax-zmin))
        gp.crossvalidated(kfold=2, plot=False, eps=0.0001, verbose=False)
        #gp.tuneParameters(Xtest, p0=[0.3,0.3], plot=True, eps=0.0001, verbose=True)

        YtestHat, Yvar, sigma, logP = gp.predict(Xtest)

        print(gp.param)
        ### The plotting ###
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(XG, YG, ZG, color='k')
        
        ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), YtestHat.reshape(20,20), color='r')
        ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), YtestHat.reshape(20,20)+3*sigma.reshape(20,20), color='gray')
        ax.plot_wireframe(Xtest[:,0].reshape(20,20), Xtest[:,1].reshape(20,20), YtestHat.reshape(20,20)-3*sigma.reshape(20,20), color='gray')

        ax.plot(Xtrain[:,0], Xtrain[:,1], Ytrain[:,0],'ro', markersize=12)
        plt.show()