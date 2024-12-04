import numpy as np
import random
import pygame
import pickle

class Environment:
    def __init__(self, grid_size=10, num_zombies=8, num_presents=5, num_obstacles=3, load_existing=False, cell_size=50):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_map = np.zeros((grid_size, grid_size), dtype=int)
        self.start_position = (0, 0)
        self.goal_position = (grid_size - 1, grid_size - 1)
        self.num_presents = num_presents
        self.collected_presents = set()
        self.agent_image = pygame.image.load('images/agent.png')
        self.zombie_image = pygame.image.load('images/zombie.png')
        self.reward_image = pygame.image.load('images/reward.png')
        self.goal_image = pygame.image.load('images/goal.png')
        self.stone_image = pygame.image.load('images/stone.png')
        self.agent_image = pygame.transform.scale(self.agent_image, (cell_size, cell_size))
        self.zombie_image = pygame.transform.scale(self.zombie_image, (cell_size, cell_size))
        self.reward_image = pygame.transform.scale(self.reward_image, (cell_size, cell_size))
        self.goal_image = pygame.transform.scale(self.goal_image, (cell_size, cell_size))
        self.stone_image = pygame.transform.scale(self.stone_image, (cell_size, cell_size))

        if load_existing:
            grid_data = self.load_environment_data()
        else:
            grid_data = None

        if grid_data:
            self.zombie_positions, self.present_positions, self.obstacle_positions = grid_data
        else:
            self.zombie_positions = self.place_random_items(num_zombies)
            self.present_positions = self.place_random_items(num_presents, exclude=self.zombie_positions)
            self.obstacle_positions = self.place_random_items(
                num_obstacles, exclude=self.zombie_positions + self.present_positions
            )
            self.save_environment_data()

        self.update_grid()
        self.current_position = self.start_position

    def place_random_items(self, num_items, exclude=None):
        exclude = set(exclude or [])
        items = set()
        while len(items) < num_items:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (
                (i, j) not in exclude
                and (i, j) not in items
                and (i, j) not in {self.start_position, self.goal_position}
            ):
                items.add((i, j))
        return list(items)

    def update_grid(self):
        self.grid_map.fill(0)
        for (i, j) in self.zombie_positions:
            self.grid_map[i][j] = 1
        for (i, j) in self.present_positions:
            self.grid_map[i][j] = 2
        for (i, j) in self.obstacle_positions:
            self.grid_map[i][j] = 3

    def reset_environment(self):
        self.current_position = self.start_position
        self.collected_presents.clear()
        return self.current_position, tuple(self.collected_presents)

    def perform_action(self, action):
        movements = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        i, j = self.current_position
        di, dj = movements.get(action, (0, 0))
        new_position = (max(0, min(i + di, self.grid_size - 1)), max(0, min(j + dj, self.grid_size - 1)))

        if new_position in self.obstacle_positions:
            new_position = self.current_position

        self.current_position = new_position
        return self.evaluate_current_position()

    def evaluate_current_position(self):
        status_message = ""
        reward = -0.1
        task_complete = False

        if self.current_position == self.goal_position:
            if len(self.collected_presents) == len(self.present_positions):
                reward, status_message = 10, "Completed successfully"
            else:
                reward, status_message = -1, "Left without collecting all rewards"
            task_complete = True
        elif self.current_position in self.zombie_positions:
            reward, status_message = -5, "Encountered a zombie"
            task_complete = True
        elif self.current_position in self.present_positions and self.current_position not in self.collected_presents:
            self.collected_presents.add(self.current_position)
            reward = 2

        return self.current_position, tuple(self.collected_presents), reward, task_complete, status_message
    
    def render_environment(self, display, cell_size=60):
        display.fill((230, 230, 230))
        for i in range(self.grid_size + 1):
            pygame.draw.line(display, (0, 0, 0), (0, i * cell_size), (self.grid_size * cell_size, i * cell_size), 2)
            pygame.draw.line(display, (0, 0, 0), (i * cell_size, 0), (i * cell_size, self.grid_size * cell_size), 2)


        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.current_position:
                    display.blit(self.agent_image, (j * cell_size, i * cell_size))
                elif (i, j) == self.goal_position:
                    display.blit(self.goal_image, (j * cell_size, i * cell_size))
                elif self.grid_map[i][j] == 1:
                    display.blit(self.zombie_image, (j * cell_size, i * cell_size))
                elif self.grid_map[i][j] == 2 and (i, j) not in self.collected_presents:
                    display.blit(self.reward_image, (j * cell_size, i * cell_size))
                elif self.grid_map[i][j] == 3:
                    display.blit(self.stone_image, (j * cell_size, i * cell_size))

        pygame.display.flip()


    def save_environment_data(self):
        data = {
            'zombie_positions': self.zombie_positions,
            'present_positions': self.present_positions,
            'obstacle_positions': self.obstacle_positions
        }
        with open('environment.pkl', 'wb') as file:
            pickle.dump(data, file)

    def load_environment_data(self):
        try:
            with open('environment.pkl', 'rb') as file:
                data = pickle.load(file)
                return data['zombie_positions'], data['present_positions'], data['obstacle_positions']
        except FileNotFoundError:
            return None
