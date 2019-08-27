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
    def __init__(self, xbounds, ybounds, nfitness, npop=100, maxtimelevel=1):
        self.currentIteration = 0
        self.xdim = len(xbounds)
        self.ydim = len(ybounds)
        self.nfitness = nfitness
        self.species = [Specie.initializeRandomly(ninputs=len(xbounds), noutputs=len(ybounds), maxtimelevel=maxtimelevel, paddnode=0.4) for _ in range(npop)]
        self.species_best = []
        self.best_fitness = np.zeros((0, self.nfitness))
        self.npop = npop

    
    ### Initialize ### 
    def initialize(self):
        self.currentIteration = 0

    ### here goes your code ###
    def run(self, specie):
        pass

    ### Run different species for different generations ###
    def iterate(self, generationMax, sigmat=0.1):

        for generation in range(generationMax):
            ### Run species ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            fitness = p.map(self.run, [specie for specie in self.species])
            fitness = np.asarray(fitness).reshape(len(self.species), self.nfitness)

            ### Calculate compability distance ###
            share = np.zeros((len(self.species), len(self.species)))
            for i, specie1 in enumerate(self.species):
                for j, specie2 in enumerate(self.species):
                    if np.all(fitness[i,:]<fitness[j,:]):
                        share[i,j] = 1 if Specie.compabilityMeasure(specie1, specie2) <= sigmat else 0
                    else:
                        share[i,j] = 1 if Specie.compabilityMeasure(specie2, specie1) <= sigmat else 0

            fitness_adjusted = fitness/share.sum(axis=1).reshape(-1,1)
            ### Summary ####
            for n, specie in enumerate(self.species):
                print("\tSpecie ID: {}, Nodes: {} Fitness: {} Fitness_adj: {}".format(specie._id, len(specie.nids), np.around(fitness[n,:],3), np.around(fitness_adjusted[n,:],3)))
            print("Fitness Mean: {:.4f}, Fitness Min: {:.4f}, Fitness Std: {:.4f}".format(fitness.mean(), fitness.min(), fitness.std()))

            ### Combine generation with best species ###
            self.species = self.species + self.species_best
            fitness_adjusted = np.vstack((fitness_adjusted, self.best_fitness))

            ### Compute fitness and pareto ranks ###
            ranks = np.argsort(fitness_adjusted, axis=0)[:,0]

            ### Sort by fitness ###
            self.species = [self.species[index] for index in ranks]
            fitness_adjusted = fitness_adjusted[ranks] 

            ### Update bestspecies ###
            self.best_fitness = fitness_adjusted[:10,:]
            self.species_best = [self.species[i] for i in range(10)]

            ### Selection probabilities for crossover ###
            probs = 1.0/(np.arange(0,len(ranks))+1)
            probs = probs/np.sum(probs)

            ### build next generation ###
            next_species = []

            ### Elitist ###
            next_species.extend(self.species[0:3])
            self.species[0].showGraph(store=True, picname="elitist_generation_{}.png".format(generation))

            ### Crossover ###
            while len(next_species) < self.npop:
                mate_candidates = np.random.choice(len(self.species), 2, p=probs)
                if ranks[mate_candidates[0]] >= ranks[mate_candidates[1]]:
                    next_species.append(Specie.crossover(self.species[mate_candidates[0]], self.species[mate_candidates[1]], generation=generation))
                else:
                    next_species.append(Specie.crossover(self.species[mate_candidates[1]], self.species[mate_candidates[0]], generation=generation))


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
                if np.random.rand() < 0.3:
                    next_species[n] = Specie.mutate_weight(next_species[n], generation=generation)
                if np.random.rand() < 0.05:
                    next_species[n] = Specie.mutate_bias(next_species[n], generation=generation)

            ### Update species for next generation ###
            self.species = next_species.copy()

# ### Simulation environment for neat ###
# def simulation(specie):
    
#     x = np.linspace(0,1,10).reshape(10,1)
#     y = np.sin(x)

#     yhat = specie.run(x)
#     rmse = np.mean((y-yhat)**2)
#     cost = specie.cost()

#     return np.asarray([rmse, cost]).reshape(1,2)


### Simulation environment for neat ###
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

    if True:
        ### Use gym as test environment ###
        
        
        xdim = env.observation_space.shape[0]
        ydim = env.action_space.shape[0]
        ybounds =  [(env.action_space.low, env.action_space.high)]
        xbounds =  xdim*[(-10,10)]

        ### NEAT ###
        neat = NEAT(xbounds, ybounds, nfitness=1, npop=40, maxtimelevel=1)
        neat.run = simulation
        neat.iterate(80)


    else:
        ### NEAT ###
        neat = NEAT(xbounds=[(0,1)], ybounds=[(0,1)], nfitness=2, npop=60, maxtimelevel=1)
        neat.run = simulation
        neat.iterate(1)

        yhat = neat.species_best[0].run(np.linspace(0,1,10).reshape(10,1))

        plt.plot(neat.best_fitness[:,0], neat.best_fitness[:,1], 'o')
        plt.show()

        plt.plot(np.linspace(0,1,10), np.sin(np.linspace(0,1,10)),'bo-')
        plt.plot(np.linspace(0,1,10), yhat,'ro-')
        plt.show()