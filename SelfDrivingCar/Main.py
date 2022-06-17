from CartPoleEnv import CartPoleEnv
import time
import random
from NeuralNetwork import NeuralNetwork, sigmoid, sigmoid_derivative
import numpy as np
import copy

SIZE = 200

def main():
    env = CartPoleEnv()
    networks = []
    for _ in range(SIZE):
        networks.append(NeuralNetwork([4, 10, 1], [[sigmoid] * 2, [sigmoid_derivative] * 2]))

    for _ in range(SIZE):
        # run
        for network in networks:
            run_network(env, network)
            env.reset()

        # get top networks
        top_networks = sorted(networks, key=lambda x: x.score, reverse=True)[:SIZE//10]
        run_network(env, top_networks[0], True)

        # reproduce
        networks = copy.deepcopy(top_networks)
        for network in top_networks:
            for _ in range(8):
                new_network = copy.deepcopy(network)
                new_network.mutate()
                networks.append(new_network)
        for _ in range(SIZE//10):
            networks.append(NeuralNetwork([4, 10, 1], [[sigmoid] * 2, [sigmoid_derivative] * 2]))

def run_network(env, network, render=False):
    t = 0
    movement = 0
    for i in range(15):
        ct = 0
        while not env.done and ct < 500:
            action = network.get_output(env.current_state())[0]
            env.step(action)
            if render and i == 1:
                env.render()
                time.sleep(3 / t)
            ct += 1
            movement += abs(env.x_pos) + abs(env.angle)
        t += ct
        env.reset()
    network.score = (t - movement / t * 100) / 15
    if render:
        print(t / 15, network.score)

if __name__ == "__main__":
    main()