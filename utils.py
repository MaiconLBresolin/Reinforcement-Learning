import pygame
import sys

def handle_pygame_events():
   for event in pygame.event.get():
      if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def init_pygame(env):
   pygame.init()
   
   cellSize = 100 - (env.grid_size * 4)

   if cellSize <= 0:
      cellSize = 10

   screen = pygame.display.set_mode((env.grid_size * cellSize, env.grid_size * cellSize))
   pygame.display.set_caption('RL')

   return screen, cellSize


def quit_pygame():
   pygame.quit()
   sys.exit()