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
    def __init__(self, xdim, ydim, npop=100, maxtimelevel=1, output_activation=None):
        self.currentIteration = 0
        self.xdim = xdim
        self.ydim = ydim
        self.species = {}
        self.ybest = np.zeros((0, 1))
        self.ybest_adjusted = np.zeros((0, 1))
        self.npop = npop
        self.output_activation = ydim*[0] if output_activation is None else output_activation

    ### Initialize ###
    def initialize(self):
        self.currentIteration = 0

    ### here goes your code ###
    def run(self, genom):
        pass

    ### Run different genomes for different generations ###
    def iterate(self, generationMax, sigmat=1.0):

        genomes = [Genom.initializeRandomly(ninputs=self.xdim, noutputs=self.ydim, maxtimelevel=1, output_activation=self.output_activation) for _ in range(self.npop)]
        self.species = {0: {"genome_prototype": genomes[0], "genomes": [], "fitness_adjusted": None,
                       "fitness":[] , "best_fitness": -1e+4, "best_fitness_gen": 0, "noffspring": 0}}
        specieCtr = 0

        ### Start the evolution ###
        for generation in range(generationMax):

            ### Clean species dict ###
            for specieID, specie in self.species.items():
                specie["genomes"], specie["fitness"] = [], []

            ### Run genomes ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=3)
            fitness = p.map(self.run, [genom for genom in genomes])
            fitness = np.asarray(fitness).reshape(-1)

            ### Calculate compability distance ###
            for f,genom in zip(fitness, genomes):
                matched = False
                for ngenomes, specie in self.species.items():
                    if Genom.compabilityMeasure(genom, specie["genome_prototype"]) <= sigmat:
                        specie["genomes"].append(genom)
                        specie["fitness"].append(f)
                        matched = True
                        break
                if not matched:
                    specieCtr+=1
                    self.species[specieCtr] = {"genome_prototype": genom, "genomes": [genom], "fitness_adjusted": None,
                                          "fitness":[f], "best_fitness": f, "best_fitness_gen": generation, "noffspring": 0}

            ### Adjust by group fitness and kill 50% ###
            for specieID, specie in self.species.items():
                ### Check if specie is populated ###
                genomes = specie["genomes"]
                ngenomes = len(genomes)
                if ngenomes == 0:
                    specie["noffspring"] = 0
                    continue

                ### Sort by fitness ###
                f = np.asarray(specie["fitness"]).reshape(-1)
                idxs = f.argsort()[::-1]
                ikill = int(0.2*ngenomes) if ngenomes > 1 else 1

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

                ### Check if specie has gotten better ###
                if f[idxs[0]] > specie["best_fitness"]:
                    ### Specie has gotten better so we reset the best_fitness_gen to current generation ###
                    specie["best_fitness"], specie["best_fitness_gen"] = f[idxs[0]], generation

                elif (generation-specie["best_fitness_gen"]) > 50 and len(self.species)>1:
                    ### No improvement since Nextinguish generation ###
                    for genom in specie["genomes"]:
                        genom.crossover = False
                    print("Kill order for specie {}".format(specieID))

            ### Offspring ###
            noffsprings = np.asarray([specie["noffspring"] for specieID, specie in self.species.items()]).reshape(-1)
            noffsprings = noffsprings/noffsprings.sum()

            ### Elitists ###
            genomes = [specie["genomes"][0] for specieID, specie in self.species.items() if len(specie["genomes"]) > 0]

            ### Next generation ###
            while len(genomes)<self.npop:
                #print(list(species.keys()), noffsprings, )
                specieID = int(np.random.choice(list(self.species.keys()), 1, noffsprings.tolist()))
                specie = self.species[specieID]

                ### Sanity check ###
                if len(specie["genomes"]) == 0:
                    continue
                ### Crossover ###
                elif len(specie["genomes"])>1:
                    idxs = np.random.choice(len(specie["genomes"]), 2, replace=False)

                    if specie["genomes"][idxs[0]].crossover and specie["genomes"][idxs[1]].crossover:
                        if specie["fitness"][idxs[0]] > specie["fitness"][idxs[1]]:
                            genom1, genom2 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]
                        else:
                            genom2, genom1 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]

                        genom = Genom.crossover(genom1, genom2, generation=generation)
                    else:
                        #print("not allowed to mate", specieID)
                        continue

                ### Species just got one genom ###
                else:
                    genom = specie["genomes"][0]

                ### Mutations ###
                if np.random.rand() < 0.00:
                    pass #genom = Genom.mutate_activation(genom, generation=generation)
                if np.random.rand() < 0.03:
                    genom = Genom.mutate_add_node(genom, generation=generation)
                if np.random.rand() < 0.03:
                    genom = Genom.mutate_remove_node(genom, generation=generation)
                if np.random.rand() < 0.07:
                    genom = Genom.mutate_add_connection(genom, generation=generation)
                if np.random.rand() < 0.05:
                    genom = Genom.mutate_remove_connection(genom, generation=generation)
                if np.random.rand() < 0.9:
                    genom = Genom.mutate_weight(genom, generation=generation)
                if np.random.rand() < 0.9:
                    genom = Genom.mutate_bias(genom, generation=generation)

                ### Finally append it ###
                genomes.append(genom)



### Simulation environment for neat ###
def bestfit(genom):

    x = np.linspace(0,8,20).reshape(-1,1)
    y = np.sin(x)

    yhat = -1 + 2*genom.run(0.1*(x-4))
    rmse = np.mean(np.abs(y-yhat)**0.5)

    return 10.0-rmse


# ### Simulation environment for neat ###
def pendulum(genom, timesteps=1000, render=False, repeat=15):
    ep_reward = 0
    ylb, yub = np.asarray([-2.0]), np.asarray([2.0])
    xlb, xub = np.asarray([-1,-1,-8]), np.asarray([ 1, 1, 8])
    #print("Running genom {}".format(genom))
    for _ in range(repeat):
        s = env.reset()
        for t in range(timesteps):

            ### Run single genomes ###
            s_norm = (np.asarray(s).reshape(1,-1)-xlb)/(xub-xlb)
            a_norm = genom.run(s_norm)
            a = yub*a_norm.reshape(-1)

            ### Run simulation environment ###
            s2, r, done, info = env.step(a)
            ep_reward += r
            s = s2
            ### plotting and stopping ###
            if done:
                break
            if render:
                env.render()

    return 10+ep_reward/timesteps/repeat



def cartPole(genom, timesteps=400, render=False, repeat=15):
    ep_reward = 0
    #print("Running genom {}".format(genom))
    for _ in range(repeat):
        s = env.reset()
        for t in range(timesteps):

            ### Run single genomes ###
            probs = genom.run(s.reshape(1,-1))
            probs = probs.reshape(-1)/np.sum(probs)

            a = np.random.choice([0, 1], 1, p=probs)[0]

            ### Run simulation environment ###
            s2, r, done, info = env.step(a)
            ep_reward += r
            s = s2
            ### plotting and stopping ###
            if done:
                break
            if render:
                env.render()

    return ep_reward/repeat



#### TEST #######
if __name__ == "__main__":

    if True:
        ### Use gym as test environment ###
        # ### Simulation environment for neat ###
        env = gym.make("Pendulum-v0")

        ### NEAT ###
        neat = NEAT(xdim=3, ydim=1, npop=100, maxtimelevel=1, output_activation=[0])
        neat.run = pendulum
        neat.iterate(30, sigmat=2.5)

        for specieID, specie in neat.species.items():
            if len(specie["genomes"])>0:
                neat.run(specie["genomes"][0], render=True)
                specie["genomes"][0].showGraph()

    elif False:
        ### Use gym as test environment ###
        # ### Simulation environment for neat ###
        env = gym.make("CartPole-v0")
        ### NEAT ###
        neat = NEAT(xdim=4, ydim=2, npop=100, maxtimelevel=1, output_activation=[1,1])
        neat.run = cartPole
        neat.iterate(30, sigmat=2.0)

        for specieID, specie in neat.species.items():
            if len(specie["genomes"])>0:
                neat.run(specie["genomes"][0], render=True)
                specie["genomes"][0].showGraph()

    else:
        ### NEAT ###
        neat = NEAT(xdim=1, ydim=1, npop=60, maxtimelevel=1)
        neat.run = bestfit
        neat.iterate(20)

        for specieID, specie in neat.species.items():
            x = np.linspace(0,8,20).reshape(20,1)
            y = np.sin(x)
            if len(specie["genomes"])>0:
                pass
            else:
                continue
            specie["genomes"][0].showGraph()
            yhat = -1 + 2*specie["genomes"][0].run( 0.1*(x-4))
            plt.plot(x.reshape(-1), y.reshape(-1),'bo-')
            plt.plot(x.reshape(-1), yhat.reshape(-1),'ro-')
            print(np.mean((y-yhat)**2))
        plt.show()