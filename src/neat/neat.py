import numpy as np
import json
import sys
import os
import gym
import matplotlib.pyplot as plt
import multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.neat.genom import Genom
from src.optimizer.pareto import Pareto

class NEAT:
    ### Constructor ###
    def __init__(self, xbounds, ybounds, npop=100, maxtimelevel=1):
        self.currentIteration = 0
        self.xdim = len(xbounds)
        self.ydim = len(ybounds)
        self.genomes_best = []
        self.ybest = np.zeros((0, 1))
        self.ybest_adjusted = np.zeros((0, 1))
        self.npop = npop

    
    ### Initialize ### 
    def initialize(self):
        self.currentIteration = 0

    ### here goes your code ###
    def run(self, genom):
        pass

    ### Run different genomes for different generations ###
    def iterate(self, generationMax, sigmat=2.5):

        genomes = [Genom.initializeRandomly(ninputs=self.xdim, noutputs=self.ydim, maxtimelevel=1) for _ in range(self.npop)]
        species = {0: {"genome_prototype": genomes[0], "genomes": [], "fitness_adjusted": None,
                       "fitness":[] , "best_fitness": -1e+4, "best_fitness_gen": 0, "noffspring": 0}}
        specieCtr = 0

        ### Start the evolution ###
        for generation in range(generationMax):

            ### Run genomes ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=3)
            fitness = p.map(self.run, [genom for genom in genomes])
            fitness = np.asarray(fitness).reshape(-1)


            ### Calculate compability distance ###
            for f,genom in zip(fitness, genomes):
                matched = False
                for ngenomes, specie in species.items():
                    if Genom.compabilityMeasure(genom, specie["genome_prototype"]) <= sigmat:
                        specie["genomes"].append(genom)
                        specie["fitness"].append(f)
                        matched = True
                        break
                if not matched:
                    specieCtr+=1
                    species[specieCtr] = {"genome_prototype": genom, "genomes": [genom], "fitness_adjusted": None,
                                          "fitness":[f], "best_fitness": f, "best_fitness_gen": generation, "noffspring": 0}
        

            ### Adjust by group fitness and kill 50% ###
            for specieID, specie in species.items():
                ### Check if specie is populated ###
                genomes = specie["genomes"]
                ngenomes = len(genomes)
                if ngenomes == 0:
                    specie["noffspring"] = 0
                    continue

                ### Sort by fitness ###
                f = np.asarray(specie["fitness"]).reshape(-1)
                idxs = f.argsort()[::-1]
                ikill = int(0.5*ngenomes) if ngenomes > 1 else 1

                ### Check if specie has gotten better ###
                if f[idxs[0]] > specie["best_fitness"]:
                    ### Specie has gotten better so we reset the best_fitness_gen to current generation ###
                    specie["best_fitness"], specie["best_fitness_gen"] = f[idxs[0]], generation

                elif (generation-specie["best_fitness_gen"]) > 10 and len(species)>1:
                    ### No improvement since Nextinguish generation ###
                    for genom in specie["genomes"]:
                        genom.crossover = False

                    specie["noffspring"] = 0
                    specie["fitness"], specie["genomes"] = [],[]
                    specie["best_fitness_gen"] = generation
                    print("-> Killing species {}".format(specieID))
                    continue

                ### Sort by fitness ###
                f = f[idxs]
                fadj = f/ngenomes
                genomes = [genomes[idx] for idx in idxs]
                ### kill half of the population ###
                specie["fitness"] = f[:ikill].tolist()
                specie["genomes"] = genomes[:ikill]
                specie["fitness_adjusted"] = fadj[:ikill].tolist()
                specie["noffspring"] = np.sum(fadj[:ikill])
                ### Print summary ###
                print("-> Specie: {}, Pop: {}, FitMin: {:.3f}/{:.3f}, FitMean: {:.3f}/{:.3f}, FitMax: {:.3f}/{:.3f}".format(specieID, ngenomes, f.min(), fadj.min(), f.mean(), fadj.mean(), f.max(), fadj.max()))


            ### Offspring ###
            noffsprings = np.asarray([specie["noffspring"] for specieID, specie in species.items()]).reshape(-1)
            noffsprings = noffsprings/noffsprings.sum()
            next_genomes = []

            for noffspring, (specieID, specie) in zip(noffsprings, species.items()):

                print(specieID, int(noffspring*self.npop))
                if len(specie["genomes"]) == 0:
                    continue

                for i in range(int(noffspring*self.npop)):
                    ### Crossover ###
                    if np.random.rand() < 0.3 and len(specie["genomes"])>1:
                        idxs = np.random.choice(len(specie["genomes"]), 2, replace=False)

                        if specie["fitness"][idxs[0]] > specie["fitness"][idxs[1]]:
                            genom1, genom2 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]
                        else:
                            genom2, genom1 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]

                        next_genomes.append(Genom.crossover(genom1, genom2, generation=generation))
                    ### Mutation ###
                    else:
                        idx = np.random.randint(0,len(specie["genomes"]))
                        genom = specie["genomes"][idx]

                        if np.random.rand() < 0.02:
                            genom = Genom.mutate_activation(genom, generation=generation)
                        if np.random.rand() < 0.03:
                            genom = Genom.mutate_add_node(genom, generation=generation)
                        if np.random.rand() < 0.03:
                            genom = Genom.mutate_remove_node(genom, generation=generation)
                        if np.random.rand() < 0.05:
                            genom = Genom.mutate_add_connection(genom, generation=generation)
                        if np.random.rand() < 0.03:
                            genom = Genom.mutate_remove_connection(genom, generation=generation)
                        if np.random.rand() < 0.8:
                            genom = Genom.mutate_weight(genom, generation=generation)
                        if np.random.rand() < 0.8:
                            genom = Genom.mutate_bias(genom, generation=generation)

                        next_genomes.append(genom)

            ### Fill up the remaining ones with ###
            while len(next_genomes) < self.npop:
                species1 = species[np.random.randint(0,len(species))]
                species2 = species[np.random.randint(0,len(species))]

                if len(species1["genomes"]) == 0 or len(species2["genomes"]) == 0:
                    continue

                idx1 = np.random.randint(0,len(species1["genomes"])) if len(species1["genomes"])>1 else 0
                idx2 = np.random.randint(0,len(species2["genomes"])) if len(species2["genomes"])>1 else 0

                if species1["fitness"][idx1] > species2["fitness"][idx2]:
                    next_genomes.append(Genom.crossover(species1["genomes"][idx1], species2["genomes"][idx2], generation=generation))
                else:
                    next_genomes.append(Genom.crossover(species2["genomes"][idx2], species1["genomes"][idx1], generation=generation))

            ### Reset species population ###
            for specieID, specie in species.items():
                specie["genomes"], specie["fitness"] = [], []

            ### Update genomes for next generation ###
            genomes = next_genomes.copy()

### Simulation environment for neat ###
def bestfit(genom):
    
    x = np.linspace(0,4,20).reshape(-1,1)
    y = np.sin(x)

    yhat = -1 + 2*genom.run(-1+2*(x-0)/4)
    rmse = np.mean((y-yhat)**2)
    cost = genom.cost()

    return 10 -np.asarray(rmse)


# ### Simulation environment for neat ###
env = gym.make("Pendulum-v0")
def simulation(genom, timesteps=1000, render=False):
    ep_reward = 0
    #print("Running genom {}".format(genom))
    s = env.reset()
    for t in range(timesteps):

        ### Run single genomes ###
        s_norm = np.asarray(s).reshape(1,-1) / 10
        a_norm = genom.run(s_norm)
        a = 4*a_norm
        cost = genom.cost()

        ### Run simulation environment ###
        s2, r, done, info = env.step(a)
        ep_reward += r
        s = s2
        ### plotting and stopping ###
        if done:
            s = env.reset()
        if render:
            env.render()

    return 10+ep_reward/timesteps



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
        neat.iterate(100)

        for genom in neat.genomes_best:
            yhat = genom.run(np.linspace(0,4,20).reshape(20,1))

            plt.plot(np.linspace(0,4,20), np.sin(np.linspace(0,4,20)),'bo-')
            plt.plot(np.linspace(0,4,20), yhat,'ro-')
        plt.show()