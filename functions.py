import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


class Reinforcement:
    def __init__(self, bsize, gamma, epsilon_start, epsilon_end, decay_rate, tau):
        """
        This is a Reinforcement Learning agent. In this class, some common RL parameters and
        the whole RL frame have been defined. For special algorithm agent, initialization and
        some functions should be redefined. The functions are as follows: epsilon_decay,
        q_learning_update, soft_update_target. Other functions also could be redefined for
        special requirements.
        """
        self.batch_size = bsize
        self.discount_factor = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.tau = tau

    def epsilon_decay(self, episode):
        epsilon = self.epsilon_end + ((self.epsilon_start - self.epsilon_end) * np.exp(-self.decay_rate * episode))
        return epsilon

    def q_learning_update(self, optimizer, criterion, target_net, dql_net):
        replay_memory = deque(maxlen=2000)

        # Sample a random minibatch from replay memory and perform Q-learning update
        if len(replay_memory) >= self.batch_size:
            minibatch = random.sample(replay_memory, self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

            state_batch = torch.cat(state_batch).unsqueeze(0).to(device)
            next_state_batch = torch.cat(next_state_batch).unsqueeze(0).to(device)
            state_batch = state_batch.transpose(0, 1)
            next_state_batch = next_state_batch.transpose(0, 1)

            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)
            action_batch = torch.tensor(action_batch, dtype=torch.int64)

            q_values_current = dql_net(state_batch)
            with torch.no_grad():
                q_values_next_target = target_net(next_state_batch)
            target_values = reward_batch + self.discount_factor * torch.max(q_values_next_target, dim=1).values * (
                    1 - done_batch)

            optimizer.zero_grad()
            action_batch = action_batch.unsqueeze(1)
            loss = criterion(q_values_current.gather(dim=1, index=action_batch), target_values.unsqueeze(1))
            loss.backward()
            optimizer.step()

    def soft_update_target(self, target_model, behavior_model):
        """
        Update target network softly.
        """
        for target_param, local_param in zip(target_model.parameters(), behavior_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def preprocess_observation(observation: np.array) -> torch.Tensor:
    """
    Transform rgb observation image to a smaller gray image.

    Args:
        observation: The image data.

    Returns:
        tensor_observation: A sequence represents a gray image.
    """
    image = cv2.cvtColor(observation[0], cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    tensor_observation = torch.FloatTensor(image).unsqueeze(0) / 255.0
    return tensor_observation


def save_best_result(best_score, best_episode):
    """
    Save information about the best results.
    """
    save_ = os.path.dirname(__file__)
    results_ = r'results'
    best_score_ = 'best_score_{}'.format(time)
    if not os.path.exists(os.path.join(save_, results_, best_score_)):
        os.makedirs(os.path.join(save_, results_, best_score_))
    file_name = os.path.join(save_, results_, best_score_, "The_best_episode_info.txt")
    with open(file_name, "a+") as file:
        file.write("Best score" + ' >>> ' + str(best_score) + '\n\n')
        file.write("Best episode" + ' >>> ' + str(best_episode) + '\n\n')

    file.close()


def plot_performance(var1, var2):
    """
    Plot different variables to check for performance.
    """
    save_ = os.path.dirname(__file__)
    results_ = r'results'
    plots_ = 'plots_{}'.format(time)
    if not os.path.exists(os.path.join(save_, results_, plots_)):
        os.makedirs(os.path.join(save_, results_, plots_))
    # Plot the variables
    plt.plot(var1, label='episode')
    plt.plot(var2, label='score_values')
    # Add labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    save_path = os.path.join(save_, results_, plots_, 'agent_performance.png')
    plt.savefig(save_path)


def load_model_weights(model, weights_path):
    """
    Load saved model according to different algorithm.

    Args:
        model: The model to be loaded.
        weights_path: The path that contains saved model weights.
    """
    model.load_state_dict(torch.load(weights_path))
    model.eval()
