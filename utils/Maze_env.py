import pygame
import numpy as np
import random
from collections import deque

# Constants
cell_size= 40
width, height= 15, 10
screen_size= (width*cell_size, height*cell_size)
screen = pygame.display.set_mode(screen_size)
ACTIONS= ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Colors
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GOLD = (255, 215, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

def load_maze(file_path):
        walls = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == 'W':
                    walls.append((int(parts[1]), int(parts[2])))
        return walls

class MazeEnv:
    def __init__(self):
        self.width = width
        self.height = height
        self.walls = load_maze(file_path)
        self.reset()

    def reset(self):
        self.harry = self.random_empty_cell()
        self.death_eater = self.random_empty_cell()
        self.cup = self.random_empty_cell()
        while self.cup == self.harry:
            self.cup = self.random_empty_cell()
        while self.death_eater == self.harry:
            self.death_eater = self.random_empty_cell()
        while self.death_eater == self.cup:
            self.death_eater = self.random_empty_cell()
        self.done = False
        self.reward = 0
        return self._get_state()

    def _get_state(self):
        return np.array([self.harry[0],self.harry[1],
                         self.cup[0]-self.harry[0],
                         self.cup[1]-self.harry[1],
                         self.death_eater[0]-self.harry[0],
                         self.death_eater[1]-self.harry[1],
                         int(self.is_near_wall('UP')),            # Wall proximity indicators
                         int(self.is_near_wall('DOWN')),
                         int(self.is_near_wall('LEFT')),
                         int(self.is_near_wall('RIGHT'))], dtype=np.float32)
        
    def is_near_wall(self, dir):
        x, y = self.harry
        if dir== 'UP':
            return (x, y-1) in self.walls
        elif dir== 'DOWN':
            return (x, y+1) in self.walls
        elif dir== 'LEFT':
            return (x-1, y) in self.walls
        elif dir== 'RIGHT':
            return (x+1, y) in self.walls
        else:
            raise ValueError("Invalid direction")


    def random_empty_cell(self):
        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos not in self.walls:
                return pos

    def step(self, action):
        old_cup_dist=abs(self.harry[0]-self.cup[0])+ abs(self.harry[1]-self.cup[1])  #distance between harry and cup initially
        old_death_dist=abs(self.harry[0]-self.death_eater[0])+abs(self.harry[1]-self.death_eater[1])   #distance between harry and death eater initially
        x, y = self.harry
        if random.random()< 0.1:    #10% of the time harry can move two steps
          dx,dy=0,0     
          if action== 'UP':
              dy-= 2
          elif action== 'DOWN':
              dy+= 2
          elif action== 'LEFT':
              dx-= 2
          elif action== 'RIGHT':
              dx+= 2

          #check intermediate cell for walls
          inter_pos= (x+(dx/2), y+(dy/2))
          new_pos= (x+dx, y+dy)
            
          if (0<= new_pos[0]<self.width and 0<=new_pos[1]<self.height and 
              inter_pos not in self.walls and new_pos not in self.walls):
              self.harry= new_pos

        else:
          if action== 'UP':
              y-= 1
          elif action== 'DOWN':
              y+= 1
          elif action== 'LEFT':
              x-= 1
          elif action== 'RIGHT':
              x+= 1

          new_pos= (x, y)
          if 0<=x<self.width and 0<=y<self.height and new_pos not in self.walls:
              self.harry= new_pos

        new_cup_dist=abs(self.harry[0]-self.cup[0]) + abs(self.harry[1]-self.cup[1])  #distance between harry and cup finally
        
        #move the death eater
        self.death_eater_move()
        new_death_dist= abs(self.harry[0]-self.death_eater[0])+abs(self.harry[1]-self.death_eater[1])  #distance between harry and death eater finally
        done= False
        reward=0

        reward+=(old_cup_dist-new_cup_dist)*2  #rewarding the agent progress towards the cup
        reward-=(old_death_dist-new_death_dist)*1  #penalty as the agent progress towards the death eater

        if self.harry== self.cup:
            reward+= 100
            done= True
        elif self.harry== self.death_eater:
            reward+= -50
            done= True

        reward-=0.5  #step penalty
        return self._get_state(), reward, done

    def death_eater_move(self):
        path= self.bfs(self.death_eater, self.harry)
        if len(path)> 1:
          self.death_eater= path[1]
        else:
          self.death_eater = random.choice(self.get_neighbors(self.death_eater))

    def bfs(self, start,goal):
        queue =deque()
        queue.append((start,[start]))
        visited =set()
        visited.add(start)

        while queue:
            (current, path)= queue.popleft()
            if current ==goal:
                return path

            neighbors= self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path+[neighbor]))

        return [start]

    def get_neighbors(self, pos):
        x,y =pos
        candidates = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        valid = [(nx, ny) for (nx, ny) in candidates
                 if 0<= nx<self.width and 0<= ny <self.height and (nx,ny) not in self.walls]
        return valid

    def render(self):
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if (x, y) in self.walls:
                    pygame.draw.rect(screen, GRAY, rect)
                elif (x, y) == self.cup:
                    pygame.draw.rect(screen, GOLD, rect)
                elif (x, y) == self.harry:
                    pygame.draw.rect(screen, BLUE, rect)
                elif (x, y) == self.death_eater:
                    pygame.draw.rect(screen, RED, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
        pygame.display.flip()