import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("CartPole-v0")


def simulation(timesteps=1000, render=False):
    ep_reward = 0
    #print("Running genom {}".format(genom))
    s = env.reset()
    for t in range(timesteps):

        ### Run single genomes ###
        a = np.zeros((2, 1))
        idx = np.random.choice([0, 1], 1)[0]

        ### Run simulation environment ###
        s2, r, done, info = env.step(idx)
        ep_reward += r
        s = s2
        ### plotting and stopping ###
        if done:
            print(ep_reward)
            break
        if render:
            env.render()

    return ep_reward


import sys

reward = []
for _ in range(15):
    reward.append(simulation())
sys.exit()
plt.plot(reward)
plt.axhline(y=np.mean(np.asarray(reward)), color="r")
plt.show()
