import numpy as np
import random
import pygame
from utils import handle_pygame_events
import pickle


class Agent:
    def __init__(self, environment, load_rewards=False):
        self.total_episodes = 10000
        self.max_actions = environment.grid_size * environment.grid_size
        self.alpha = 0.1
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.001
        self.grid_size = environment.grid_size
        self.environment = environment
        self.target_positions = environment.present_positions

        self.q_table = self.load_rewards() if load_rewards else self.initialize_rewards_table()

    def initialize_rewards_table(self):
        dimensions = (self.grid_size, self.grid_size, 2 ** len(self.target_positions), 4)
        return np.zeros(dimensions)

    def choose_action(self, position, collected_targets):
        encoded_state = self.encode_collected_targets(collected_targets)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        return np.argmax(self.q_table[position[0]][position[1]][encoded_state])

    def train(self, display, grid_cell_size):
        for episode in range(self.total_episodes):
            current_position, collected_targets = self.environment.reset_environment()
            is_complete, step_count = False, 0

            while not is_complete and step_count < self.max_actions:
                handle_pygame_events()
                action = self.choose_action(current_position, collected_targets)
                next_position, next_targets, reward, is_complete, _ = self.environment.perform_action(action)

                self.update_rewards_table(
                    current_position, collected_targets, action, reward, next_position, next_targets
                )

                current_position, collected_targets = next_position, next_targets
                step_count += 1

            self.exploration_rate = max(
                self.min_exploration, self.exploration_rate * (1 - self.exploration_decay)
            )

            if episode % 1000 == 0:
                self.environment.render_environment(display, grid_cell_size)
                pygame.time.wait(350)

        self.save_rewards()

    def test(self, display, grid_cell_size):
        current_position, collected_targets = self.environment.reset_environment()
        is_complete, step_counter = False, 0

        while not is_complete:
            handle_pygame_events()
            encoded_state = self.encode_collected_targets(collected_targets)
            action = np.argmax(self.q_table[current_position[0]][current_position[1]][encoded_state])

            current_position, collected_targets, _, is_complete, status = self.environment.perform_action(action)
            self.environment.render_environment(display, grid_cell_size)
            pygame.time.wait(500)
            step_counter += 1

        return status, collected_targets, step_counter

    def update_rewards_table(
        self, position, collected_targets, action, reward, next_position, next_targets
    ):
        current_state_index = self.encode_collected_targets(collected_targets)
        next_state_index = self.encode_collected_targets(next_targets)

        current_q_value = self.q_table[position[0]][position[1]][current_state_index][action]
        max_next_q_value = np.max(self.q_table[next_position[0]][next_position[1]][next_state_index])

        self.q_table[position[0]][position[1]][current_state_index][action] = current_q_value + self.alpha * (
            reward + self.gamma * max_next_q_value - current_q_value
        )

    def encode_collected_targets(self, collected_targets):
        return int("".join(["1" if target in collected_targets else "0" for target in self.target_positions]), 2)

    def save_rewards(self):
        print("//")
        print("Saving Rewards Table...")
        print("//")
        with open("rewards_table.pkl", "wb") as file:
            pickle.dump(self.q_table, file)

    def load_rewards(self):
        print("//")
        print("Loading Rewards Table...")
        print("//")
        try:
            with open("rewards_table.pkl", "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            print("Rewards table file not found.")
            return self.initialize_rewards_table()
