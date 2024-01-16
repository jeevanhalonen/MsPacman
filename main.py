import gym
import numpy as np
from train_test_agent import Learn

train_episodes = 50  # number of episodes to train agent
val_episodes = 10  # number of episodes to evaluate agent

# hyperparameters
gamma = 0.99  # discount factor
lr = 0.0001  # learning rate
bsize = 32  # batch size
epsilon_start = 1.0  # maximum exploration value
epsilon_end = 0.05  # minimum exploration value
decay_rate = 0.001  # the rate of reduction in exploration
tau = 0.001  # hyperparameter to control weight in networks (ranges between 0 and 1)


def main():
    """
    The main function for gym atari reinforcement learning.
    """
    env_name = 'MsPacman-v0'
    env = gym.make(env_name)
    game = (env_name, env, True)
    action_space = np.array([i for i in range(env.action_space.n)], dtype=np.uint8)
    out_channels = len(action_space)  # the number of actions in the environment
    img_size = (1, 84, 84)  # grayscale image size

    agent = Learn(img_size, gamma, lr, epsilon_start, decay_rate, epsilon_end,
                  bsize, action_space, game, train_episodes, val_episodes, out_channels, tau)
    agent.train_model()
    agent.evaluate_model()


if __name__ == '__main__':
    main()
