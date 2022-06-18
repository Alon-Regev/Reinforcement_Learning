from CarDrivingEnv import CarDrivingEnv
import time
from NeuralNetwork import NeuralNetwork, sigmoid, sigmoid_derivative
import copy
import matplotlib.pyplot as plt

TOP_NETWORKS = 4
SIZE = TOP_NETWORKS * 3    # 300
gen = 0

def main():
    global gen
    env = CarDrivingEnv()
    networks = []
    for _ in range(SIZE):
        networks.append(new_net())

    for _ in range(SIZE):
        # run
        for i, network in enumerate(networks):
            run_network(env, network)
            env.reset()
            env.render_progress(i + 1, SIZE)

        # get top networks
        top_networks = sorted(networks, key=lambda x: x.score, reverse=True)[:TOP_NETWORKS]
        run_network(env, top_networks[0], True)

        # reproduce
        networks = copy.deepcopy(top_networks)
        for network in top_networks:
            for _ in range(SIZE // TOP_NETWORKS - 2):
                new_network = copy.deepcopy(network)
                new_network.mutate()
                networks.append(new_network)
        for _ in range(TOP_NETWORKS):
            networks.append(new_net())
        gen += 1

def new_net():
    return NeuralNetwork([4, 8, 6, 2], [[sigmoid] * 3, [sigmoid_derivative] * 3])

def run_network(env, network, render=False):
    global gen
    network.score = 0
    t = 0
    while not env.done and t < 750:
        action = network.get_output(env.current_state())
        env.step(action)
        if render:
            env.render(gen=gen)
            time.sleep(1 / 60)
        t += 1
    network.score += env.score()
    env.reset()
    if render:
        print(network.score)

if __name__ == "__main__":
    main()