import numpy as np
import json
import sys
import os
import gym
import matplotlib.pyplot as plt
import multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.neat.species import Specie
from src.optimizer.pareto import Pareto

class NEAT:
    ### Constructor ###
    def __init__(self, xbounds, ybounds, npop=100, maxtimelevel=1):
        self.currentIteration = 0
        self.xdim = len(xbounds)
        self.ydim = len(ybounds)
        self.species_best = []
        self.ybest = np.zeros((0, 1))
        self.ybest_adjusted = np.zeros((0, 1))
        self.npop = npop

    
    ### Initialize ### 
    def initialize(self):
        self.currentIteration = 0

    ### here goes your code ###
    def run(self, specie):
        pass

    ### Run different species for different generations ###
    def iterate(self, generationMax, sigmat=0.1):

        species = [Specie.initializeRandomly(ninputs=self.xdim, noutputs=self.ydim, maxtimelevel=1) for _ in range(self.npop)]

        for generation in range(generationMax):

            ### Run species ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=3)
            fitness = p.map(self.run, [specie for specie in species])
            fitness = np.asarray(fitness).reshape(len(species),1)

            ### Combine generation with best species ###
            species = species + self.species_best
            fitness = np.vstack((fitness, self.ybest))

            ### Calculate compability distance ###
            share = np.zeros((len(species), len(species)))
            for i, specie1 in enumerate(species):
                for j, specie2 in enumerate(species):
                    if np.all(fitness[i,:]<fitness[j,:]):
                        sigma = Specie.compabilityMeasure(specie1, specie2)
                    else:
                        sigma = Specie.compabilityMeasure(specie1, specie2)

                    share[i,j] = max([(1-sigma/sigmat),0])

            fitness_adjusted = fitness/share.sum(axis=1).reshape(-1,1)

            ### Compute fitness and pareto ranks ###
            idxs = np.argsort(fitness_adjusted, axis=0)[:,0]

            ### Sort by fitness ###
            species = [species[idx] for idx in idxs]
            fitness = fitness[idxs]
            fitness_adjusted = fitness_adjusted[idxs] 
            
            ### Summary ####
            for n, specie in enumerate(species):
                specie.iter_survived += 1
                print("\tSpecie ID: {}, Nodes: {} Fitness: {} Fitness_adj: {}".format(specie._id, len(specie.nids), np.around(fitness[n,:],3), np.around(fitness_adjusted[n,:],3)))
            print("Fitness Mean: {:.4f}, Fitness Min: {:.4f}, Fitness Std: {:.4f}".format(fitness.mean(), fitness.min(), fitness.std()))


            ### Update bestspecies ###
            self.ybest = fitness[:self.npop,:]
            self.species_best = [species[i] for i in range(self.npop)]

            ### Selection probabilities for crossover ###
            probs = 1.0/(fitness_adjusted[:,0])
            probs = probs/np.sum(probs)

            ### build next generation ###
            next_species = []

            ### Crossover ###
            while len(next_species) < self.npop:
                mate_candidates = np.random.choice(len(species), 2)
                if species[mate_candidates[0]].crossover and species[mate_candidates[1]].crossover:
                    if fitness[mate_candidates[0]] >= fitness[mate_candidates[1]]:
                        next_species.append(Specie.crossover(species[mate_candidates[0]], species[mate_candidates[1]], generation=generation))
                    else:
                        next_species.append(Specie.crossover(species[mate_candidates[1]], species[mate_candidates[0]], generation=generation))

            ### Mutations ###
            for n in range(self.npop):
                if np.random.rand() < 0.02:
                    next_species[n] = Specie.mutate_activation(next_species[n], generation=generation)
                if np.random.rand() < 0.05:
                    next_species[n] = Specie.mutate_add_node(next_species[n], generation=generation)
                if np.random.rand() < 0.03:
                    next_species[n] = Specie.mutate_remove_node(next_species[n], generation=generation)
                if np.random.rand() < 0.05:
                    next_species[n] = Specie.mutate_add_connection(next_species[n], generation=generation)
                if np.random.rand() < 0.03:
                    next_species[n] = Specie.mutate_remove_connection(next_species[n], generation=generation)
                if np.random.rand() < 0.8:
                    next_species[n] = Specie.mutate_weight(next_species[n], generation=generation)
                if np.random.rand() < 0.2:
                    next_species[n] = Specie.mutate_bias(next_species[n], generation=generation)

            ### Update species for next generation ###
            species = next_species.copy()

### Simulation environment for neat ###
def bestfit(specie):
    
    x = np.linspace(0,4,20).reshape(-1,1)
    y = np.sin(x)

    yhat = specie.run((x-0)/4)
    rmse = np.mean((y-yhat)**2)
    cost = specie.cost()

    return np.asarray(rmse).reshape(1,1)


# ### Simulation environment for neat ###
env = gym.make("Pendulum-v0")
def simulation(specie, timesteps=500, render=False):
    ep_reward = 0
    #print("Running specie {}".format(specie))
    s = env.reset()
    for t in range(timesteps):

        ### Run single species ###
        s_norm = np.asarray(s).reshape(1,-1) / 10
        a_norm = specie.run(s_norm)
        a = 4*a_norm
        cost = specie.cost()

        ### Run simulation environment ###
        s2, r, done, info = env.step(a)
        ep_reward += r
        s = s2
        ### plotting and stopping ###
        if done:
            s = env.reset()
        if render:
            env.render()

    return -ep_reward/timesteps



#### TEST #######
if __name__ == "__main__":

    if False:
        ### Use gym as test environment ###
        
        
        xdim = env.observation_space.shape[0]
        ydim = env.action_space.shape[0]
        ybounds =  [(env.action_space.low, env.action_space.high)]
        xbounds =  xdim*[(-10,10)]

        ### NEAT ###
        neat = NEAT(xbounds, ybounds, npop=40, maxtimelevel=1)
        neat.run = simulation
        neat.iterate(70)


    else:
        ### NEAT ###
        neat = NEAT(xbounds=[(0,4)], ybounds=[(-1,1)], npop=60, maxtimelevel=1)
        neat.run = bestfit
        neat.iterate(30)

        yhat = neat.species_best[0].run(np.linspace(0,4,20).reshape(20,1))

        plt.plot(np.linspace(0,4,20), np.sin(np.linspace(0,4,20)),'bo-')
        plt.plot(np.linspace(0,4,20), yhat,'ro-')
        plt.show()