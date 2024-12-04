from utils import init_pygame, quit_pygame
from environment import Environment
from agent import Agent

if __name__ == '__main__':
    size = 10
    zombies = 8
    presents = 5
    obstacles = 8

    loadGrid = False
    loadQTable = False

    env = Environment(size, zombies, presents, obstacles, loadGrid)
    agent = Agent(env, loadQTable)

    screen, cellSize = init_pygame(env)

    print(f'Training:')
    agent.train(screen, cellSize)

    print(f'Testing')
    status, collectedPresents, steps = agent.test(screen, cellSize)

    print(f'//')
    print(f'Status: {status}')
    print(f'Rewards collected: {len(collectedPresents)} de {presents}')
    print(f'Total of Steps: {steps}')
    print(f'//')

    quit_pygame()