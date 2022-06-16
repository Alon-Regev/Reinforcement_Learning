from CartPoleEnv import CartPoleEnv
import time
import random

def main():
    env = CartPoleEnv()

    for _ in range(120):
        env.update_state(random.random())
        print(env.current_state())
        env.render()
        time.sleep(1 / 30)

if __name__ == "__main__":
    main()