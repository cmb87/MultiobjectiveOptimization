import numpy as np
import matplotlib.pyplot as plt

from src.particle import Particle
from src.pareto import Pareto
from src.database import Database

### Generic Optimizer class ###
class Optimizer(object):

    """docstring for Particle"""
    TARGETDIRECTION = {"maximize": 1, "minimize": -1,
                       "max": 1, "min": -1, 1: 1, -1: -1, "1": 1, "-1": 1}

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], itermax=10, optimizationDirection=None):

        self.fct = fct
        self.itermax = itermax
        self.nparas = len(xbounds)
        self.ntrgts = len(ybounds)
        self.ncstrs = len(cbounds)

        self.xlb, self.xub = np.asarray([x[0] for x in xbounds]), np.asarray([x[1] for x in xbounds])
        self.ylb, self.yub = np.asarray([x[0] for x in ybounds]), np.asarray([x[1] for x in ybounds])
        self.clb, self.cub = np.asarray([x[0] for x in cbounds]), np.asarray([x[1] for x in cbounds])

        optimizationDirection = self.ylb.shape[0]*["minimize"] if optimizationDirection is None else optimizationDirection 
        self.targetdirection = np.asarray([Optimizer.TARGETDIRECTION[x] for x in optimizationDirection])

        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

        ### Niching parameters ###
        self.delta_y = 0.1
        self.delta_x = 0.5
        self.epsDominanceBins = 6

        ### Sanity checks ###
        assert np.any(self.xlb<self.xub), "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb<self.yub), "Y: Lower bound must be smaller than upper bound"
        assert np.any(self.clb<self.cub), "C: Lower bound must be smaller than upper bound"
        assert len(self.targetdirection) == self.ylb.shape[0], "Target directions and number of targets dimension must match!"

    ### Augmented boundaries ###
    @property
    def yauglb(self):
        return np.append(self.ylb, 0.0)
    
    @property
    def yaugub(self):
        return np.append(self.yub, 1.0)

    @property
    def targetdirectionAug(self):
        return np.append(self.targetdirection, -1)

    ### Evaluate function ###
    def evaluate(self, X):
        ### Evaluate toolchain ###
        output = self.fct(X)

        ### Check toolchain output ###
        if isinstance(output,tuple):
            Y, C = output[0], output[1]
        elif isinstance(output,np.ndarray):
            Y, C = output[0], np.zeros((X.shape[0],0))
        else:
            print("ERROR: Datatype {} unkown".format(type(output)))
            return

        Y = Optimizer._sanityCheck(Y, X.shape[0], self.ntrgts)
        C = Optimizer._sanityCheck(C, X.shape[0], self.ncstrs)

        ### Build penalty function ###
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)
        P  = (Py+Px+Pc)

        ### Augmented ###
        Yaug = np.hstack((Y,P))
        return Yaug, Y, C, P

    ### Check boundary violation and penalizis it ###
    @staticmethod
    def boundaryCheck(Y, ylb, yub):
        Y = Optimizer._nondimensionalize(Y, ylb, yub)
        index = (Y<0) | (Y>1.0)
        Y[~index] = 0.0
        return np.sum(Y**2,axis=1).reshape(-1,1) 

    ### Sanity check response ###
    @staticmethod
    def _sanityCheck(Y, itarget, jtarget):
        if isinstance(Y,list):
            Y = np.asarray(Y)
        elif Y is None:
            Y = np.zeros((itarget,0))
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        assert Y.shape[0] == itarget and Y.shape[1] == jtarget, "Response shape {} does not match agreed shape ({},{})".format(Y.shape,itarget,jtarget)
        return Y

    ### Update leader particle without external archive ###
    def findParetoMembers(self, X, Yaug):
        ### Only designs without penality violation ###
        validIndex = Yaug[:,-1] == 0.0
        
        ### Some designs are valid ###
        if Yaug[validIndex].shape[0] > 0:
            Yvalid, X = Yaug[validIndex,:-1].reshape(-1,self.ntrgts), X[validIndex,:].reshape(-1,self.nparas)
            # Calculate pareto ranks
            X = np.vstack((self.Xbest, X))
            Y = np.vstack((self.Ybest, Yvalid))
            Xpareto, Ypareto, paretoIndex, _, _, _ = Pareto.computeParetoOptimalMember(X, Y, targetdirection=self.targetdirection)

            ### Update best particle archive ###                                                
            self.Xbest, self.Ybest = Xpareto, Ypareto.copy()

            ### Add zero columns again to keep it compatible ###
            Ypareto = np.hstack((Ypareto, np.zeros((Ypareto.shape[0],1))))

            ### Probality of being a leader ###
            py = Optimizer.neighbors(Ypareto, self.delta_y)
            px = Optimizer.neighbors(Xpareto, self.delta_x)
            pvals = py*px

        ### All designs have a penalty ==> Move to point with lowest violation ###
        else:
            index = np.argmin(Yaug[:,-1])
            Ypareto, Xpareto = Yaug[index, :].reshape(1,self.ntrgts+1), X[index, :].reshape(1,self.nparas)
            pvals = np.ones(1)

        return  Xpareto, Ypareto, pvals/pvals.sum()


    @staticmethod
    def neighbors(X, delta):
        D,N = np.zeros((X.shape[0],X.shape[0])), np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i,j] = np.linalg.norm(X[i,:]-X[j,:])

        N[D<delta] = 1.0
        del D
        return 1.0/np.exp(N.sum(axis=0))


    ### Epsilon Dominance ###
    def epsDominance(self):
        bins = np.linspace(0,1,self.epsDominanceBins)
        binDistance, index2delete = {}, []

        for n in range(self.Ybest.shape[0]):
            Ydim = Optimizer._nondimensionalize(self.Ybest[n,:], self.ylb, self.yub)
            Ydim = 0.5*(1-self.targetdirection) + self.targetdirection*Ydim

            inds = np.digitize(Ydim, bins)
            
            inds_key = '-'.join(map(str,inds))
            dist = sum([(Ydim[i]-bins[inds[i]-1])**2 for i in range(self.ntrgts)])

            if not inds_key in list(binDistance.keys()):
                binDistance[inds_key] = [dist, n]
            else:
                if binDistance[inds_key][0] < dist:
                    index2delete.append(n)
                else:
                    index2delete.append(binDistance[inds_key][1])
                    binDistance[inds_key][0] = dist
                    binDistance[inds_key][1] = n

        self.Ybest = np.delete(self.Ybest,index2delete,axis=0)
        self.Xbest = np.delete(self.Xbest,index2delete,axis=0)
        

    @staticmethod
    def _nondimensionalize(x, lb, ub):
        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(x, lb, ub):
        return lb + x * (ub - lb)

    ### Compare with external archive ###
    def compareWithArchive(self):
        pass

    ### Store to archive ###
    def store(self):
        pass

    ### add to database ###
    def archiveStore(self):
        pass





### Particle Swarm Optimizer ###
class Swarm(Optimizer):
    def __init__(self, fct, xbounds, ybounds, cbounds=[], nparticles=10, itermax=10, optimizationDirection=None):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, itermax=itermax, optimizationDirection=optimizationDirection)

        self.particles = [Particle(self.nparas, pid, vinitScale=0.5) for pid in range(nparticles)]
        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

    ### swarm size ###
    @property
    def swarmSize(self):
        return len(self.particles)
    

    ### Initialize ###
    def initialize(self):
        ### Evaluate the particle's initial position ###
        Xinit = np.asarray([Optimizer._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])
        Yaug, _, _, _ = self.evaluate(Xinit)

        ### Update the particle targets ###
        for i, particle in enumerate(self.particles):
            particle.y = self.targetdirectionAug*Optimizer._nondimensionalize(Yaug[i, :], self.yauglb, self.yaugub)
            particle.ypbest = particle.y.copy()

        ### Set global leaders ###
        self._updateLeadersWithoutExternalRepo(Xinit, Yaug)


    ### Iterate ###
    def iterate(self, visualize=False):
        ### For cinematic mode ###
        if visualize:
            Xcine = np.zeros((self.itermax, self.swarmSize,self.nparas))
            Ycine = np.zeros((self.itermax, self.swarmSize,self.ntrgts))

        ### Start iterating ###
        for n in range(self.itermax):
            print(" Episode : {}".format(n))
            ### Update particle positions ###
            [particle.update() for particle in self.particles]

            ### Evaluate the new particle's position ###
            X = np.asarray([Optimizer._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])
            Yaug, _, _, _ = self.evaluate(X)

            ### Update particle targets ###
            for i, particle in enumerate(self.particles):
                particle.y = self.targetdirectionAug*Optimizer._nondimensionalize(Yaug[i, :], self.yauglb, self.yaugub)

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###
            self._updateLeadersWithoutExternalRepo(X, Yaug)

            ### Use eps dominace to reduce pareto members ###
            #Ydim = Optimizer._nondimensionalize(self.Ybest[:,:], self.ylb, self.yub)
            #Ydim = 0.5*(1-self.targetdirection) + self.targetdirection*Ydim
            #plt.plot(Ydim[:,0], Ydim[:,1], 'bo')

            self.epsDominance()

            #bins = np.linspace(0,1,self.epsDominanceBins)
            #for b in bins:
            #    plt.axhline(y=b,color='k')
            #    plt.axvline(x=b,color='k')

            #Ydim = Optimizer._nondimensionalize(self.Ybest[:,:], self.ylb, self.yub)
            #Ydim = 0.5*(1-self.targetdirection) + self.targetdirection*Ydim
            #plt.plot(Ydim[:,0], Ydim[:,1], 'ro')



            plt.show()
            ### Only for cinematic mode ###
            if visualize:
                Xcine[n,:,:] = X
                Ycine[n,:,:] = Yaug[:,:-1]
        ### Only for cinematic mode ###
        if visualize:
            return Xcine, Ycine

    ### Update leader particle without external archive ###
    def _updateLeadersWithoutExternalRepo(self, X, Yaug):
        
        Xpareto, Ypareto, pvals = self.findParetoMembers(X, Yaug)

        ### Assign each particle a new leader ###
        for i, particle in enumerate(self.particles):
            index = np.random.choice(np.arange(Xpareto.shape[0]), p=pvals)
            particle.ygbest = self.targetdirectionAug * Optimizer._nondimensionalize(Ypareto[index, :], self.yauglb, self.yaugub)
            particle.xgbest = Optimizer._nondimensionalize(Xpareto[index, :], self.xlb, self.xub)

        


