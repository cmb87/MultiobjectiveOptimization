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
    def iterate(self, generationMax, sigmat=3.1):

        species = [Specie.initializeRandomly(ninputs=self.xdim, noutputs=self.ydim, maxtimelevel=1) for _ in range(self.npop)]
        elitists_species, elitists_fitness = [], []

        for generation in range(generationMax):

            ### Run species ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=3)
            fitness = p.map(self.run, [specie for specie in species])
            
            ### Add elitsts ###
            #species = species + elitists_species
            #fitness.extend([[f] for f in elitists_fitness])

            fitness = np.asarray(fitness).reshape(-1,1)

            ### Calculate compability distance ###
            if generation == 0:
                specieGroups = {0: {"species_prototype": species[0], "species": [species[0]],
                                    "fitness":[fitness[0]], "best_fitness":[fitness[0]]}}
                specieCtr = 0

            for f,specie in zip(fitness[1:], species[1:]):
                for nspecies, group in specieGroups.items():
                    if Specie.compabilityMeasure(specie, group["species_prototype"]) <= sigmat:
                        group["species"].append(specie)
                        group["fitness"].append(f)
                        break
                    else:
                        specieCtr+=1
                        specieGroups[specieCtr] = {"species_prototype": specie, "species": [specie], "fitness":[f]}
                        break


            ### build next generation ###
            self.species_best = []
            elitists_species = []
            survived_species = []
            fitness = []
            prob = []
            ### Adjust by group fitness and kill 50% ###
            for specieCtr, group in specieGroups.items():
                f = np.asarray(group["fitness"]).reshape(-1)
                idxs = np.argsort(f)[::-1]

                ikill = int(0.5*len(group["fitness"])) if int(0.5*len(group["fitness"])) > 0 else 1
                group["fitness"] = [f[idxs[i]]/len(group["fitness"]) for i in range(ikill)]
                group["species"] = [group["species"][i] for i in idxs[:ikill]]
                group["best_fitness"] = f[idxs[0]]

                elitists_species.append(group["species"][0])
                elitists_fitness.append(f[idxs[0]])

                self.species_best.append(group["species"][0])
                survived_species.extend(group["species"])
                fitness.extend(group["fitness"])
                
                prob.extend(len(group["species"])*[1.0/len(group["fitness"])])


                if generation%20 == 0:
                    print("Print network")
                    group["species"][0].showGraph(store=True, picname="specie_{}_best_network.png".format(specieCtr))

                    self.run(group["species"][0])

                print("Specie {}: Population: {}, Best Fitness: {:.4f}, Adjusted fitness: {}".format(specieCtr, len(group["fitness"]), 
                                                                                              f[idxs[0]], f[idxs[0]]/len(group["fitness"])))


            ### Crossover ###
            prob = np.asarray(prob)
            prob = prob/prob.sum()

            next_species = survived_species.copy()
            ctr = 0

            while len(next_species) < self.npop:
                
                mate_candidates = np.random.choice(len(survived_species), 2, p=prob)

                if survived_species[mate_candidates[0]].crossover and survived_species[mate_candidates[1]].crossover:
                    if fitness[mate_candidates[0]] >= fitness[mate_candidates[1]]:
                        next_species.append(Specie.crossover(survived_species[mate_candidates[0]], survived_species[mate_candidates[1]], generation=generation))
                    else:
                        next_species.append(Specie.crossover(survived_species[mate_candidates[1]], survived_species[mate_candidates[0]], generation=generation))

            for n in range(self.npop):
                if np.random.rand() < 0.02:
                    next_species[n] = Specie.mutate_activation(species[ctr], generation=generation)
                if np.random.rand() < 0.03:
                    next_species[n] = Specie.mutate_add_node(species[ctr], generation=generation)
                if np.random.rand() < 0.03:
                    next_species[n] = Specie.mutate_remove_node(species[ctr], generation=generation)
                if np.random.rand() < 0.05:
                    next_species[n] = Specie.mutate_add_connection(species[ctr], generation=generation)
                if np.random.rand() < 0.03:
                    next_species[n] = Specie.mutate_remove_connection(species[ctr], generation=generation)
                if np.random.rand() < 0.8:
                    next_species[n] = (Specie.mutate_weight(species[ctr], generation=generation))
                if np.random.rand() < 0.8:
                    next_species[n] = Specie.mutate_bias(species[ctr], generation=generation)

 

            ### Update species for next generation ###
            species = next_species.copy()

### Simulation environment for neat ###
def bestfit(specie):
    
    x = np.linspace(0,4,20).reshape(-1,1)
    y = np.sin(x)

    yhat = specie.run((x-0)/4)
    rmse = np.mean((y-yhat)**2)
    cost = specie.cost()

    return -np.asarray(rmse)


# ### Simulation environment for neat ###
env = gym.make("Pendulum-v0")
def simulation(specie, timesteps=1000, render=False):
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

    return ep_reward/timesteps



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
        neat.iterate(50)


    else:
        ### NEAT ###
        neat = NEAT(xbounds=[(0,4)], ybounds=[(-1,1)], npop=60, maxtimelevel=1)
        neat.run = bestfit
        neat.iterate(30)

        for specie in neat.species_best:
            yhat = specie.run(np.linspace(0,4,20).reshape(20,1))

            plt.plot(np.linspace(0,4,20), np.sin(np.linspace(0,4,20)),'bo-')
            plt.plot(np.linspace(0,4,20), yhat,'ro-')
        plt.show()