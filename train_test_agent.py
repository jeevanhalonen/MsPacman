import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from collections import deque
from deep_net import DqlNet
from functions import Reinforcement
from functions import load_model_weights, preprocess_observation
from functions import save_best_result, plot_performance
sys.path.append(os.path.dirname(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Learn(Reinforcement):
    def __init__(self, img_size, gamma, lr, eps_start, decay_rate, eps_end,
                 bsize, action_space: np.array, game: tuple, train_episodes: int, val_episodes: int, out_channels, tau):
        super().__init__(bsize, gamma, eps_start, eps_end, decay_rate, tau)
        self.img_size = img_size
        self.action_space = action_space
        self.out_channels = out_channels
        self.action_space_len = len(self.action_space)
        # Hyperparameters
        self.gamma = gamma
        self.learning_rate = lr
        self.eps_start = eps_start
        self.decay_rate = decay_rate
        self.eps_end = eps_end
        self.batch_size = bsize
        self.num_train_episodes = train_episodes
        self.num_val_episodes = val_episodes
        self.game_name, self.environment, _ = game
        self.model = DqlNet(self.img_size, self.out_channels)
        self.behavior_model = self.model.to(device)
        self.target_model = self.model.to(device)

    # Training function
    def train_model(self):
        optimizer = optim.Adam(self.behavior_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        replay_memory = deque(maxlen=2000)
        epsilon = self.eps_start
        for episode in range(self.num_train_episodes):
            state = self.environment.reset()
            state = preprocess_observation(state)
            score = 0
            lives = 3
            step_count = 0
            done = False
            episode_len = 100
            while not done and step_count < episode_len:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = self.environment.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.behavior_model(state)
                        action = torch.argmax(q_values).item()
                # Take action in the environment
                next_state = self.environment.step(action)
                reward = next_state[1]
                info = next_state[4]
                next_state = preprocess_observation(next_state)
                # Store experience in replay memory
                experience = (state, action, reward, next_state, done)
                replay_memory.append(experience)
                state = next_state
                dead = info['lives'] < lives
                lives = info['lives']
                reward = reward if not dead else -10
                score += reward
                if lives == 0:
                    done = True
            self.q_learning_update(optimizer, criterion, self.target_model, self.behavior_model)
            # Update target network periodically
            if episode % 10 == 9:
                self.soft_update_target(self.target_model, self.behavior_model)
            # Print information about the episode
            print(f"Training episode {episode + 1}/{self.num_train_episodes},"
                  f"Score: {score}")
            # Decay epsilon
            epsilon = self.epsilon_decay(episode)
            step_count += 1
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_ = os.path.dirname(__file__)
        results_ = r'results'
        weights_ = 'model_weights_{}'.format(time)
        if not os.path.exists(os.path.join(save_, results_, weights_)):
            os.makedirs(os.path.join(save_, results_, weights_))
        # Save the trained model
        result_path = os.path.join(save_, results_, weights_, 'pacman_dqn_model.pth')
        torch.save(self.target_model.state_dict(), result_path)

    # Testing function
    def evaluate_model(self):
        best_score = 0
        best_episode = 0
        # Initialize empty lists to store data
        episode_number = []
        score_values = []
        target_net = self.target_model
        save_ = os.path.dirname(__file__)
        results_ = r'results'
        saved_weights_folder = input("Saved_weights_folder_name:  ")
        weights_file = input("Weights_file: ")
        weights_path = os.path.join(save_, results_, saved_weights_folder, weights_file)
        load_model_weights(target_net, weights_path)
        print("Model loaded")
        for episode in range(self.num_val_episodes):
            state = self.environment.reset()
            state = preprocess_observation(state)
            score = 0
            done = False
            lives = 3
            while not done:
                # Choose actions using the trained model (no exploration)
                with torch.no_grad():
                    q_values = target_net(state)
                    action = torch.argmax(q_values).item()
                # Take action in the environment
                next_state = self.environment.step(action)
                reward = next_state[1]
                info = next_state[4]
                next_state = preprocess_observation(next_state)
                score += reward
                state = next_state
                dead = info['lives'] < lives
                lives = info['lives']
                reward = reward if not dead else -10
                score += reward
                if lives == 0:
                    done = True
                if score > best_score:
                    best_score = score
                    best_episode = episode + 1
            episode_number.append(episode + 1)
            score_values.append(score)
            # Print information about the episode
            print(f"Testing episode {episode + 1}/{self.num_val_episodes},"
                  f"Score: {score}")
        average_reward = sum(score_values) / self.num_val_episodes
        print(f"Average Reward over {self.num_val_episodes} episodes: {average_reward}")
        save_best_result(best_score, best_episode)
        plot_performance(episode_number, score_values)
