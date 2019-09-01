import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("Pendulum-v0")


def simulation(timesteps=1000, render=False):
    ep_reward = 0
    #print("Running genom {}".format(genom))
    s = env.reset()
    for t in range(timesteps):

        ### Run single genomes ###
        s_norm = np.asarray(s).reshape(1, -1) / 10
        a_norm = np.random.rand(1)
        a = 4 * a_norm

        ### Run simulation environment ###
        s2, r, done, info = env.step(a)
        ep_reward += r
        s = s2
        ### plotting and stopping ###
        if done:
            s = env.reset()
        if render:
            env.render()

    return 10 + ep_reward / timesteps


import sys

reward = []
for _ in range(15):
    reward.append(simulation())

print(np.mean(np.asarray(reward)))
sys.exit()
plt.plot(reward)
plt.axhline(y=np.mean(np.asarray(reward)), color="r")
plt.show()
