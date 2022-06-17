from Environment import Environment
import math
import numpy as np
import pygame

g = 10
L = 0.4
a = 10

class CarDrivingEnv(Environment):
    def __init__(self):
        super().__init__()

        # pygame winodw
        self.canvas = pygame.display.set_mode((600, 600))

    def reset(self):
        super().reset()
        self.angle = np.random.random() / 100 - 0.005
        self.x_pos = 0.0
        self.x_vel = 0.0
        self.theta_vel = 0.0
        self.last_action = -1

    def update_state(self, action):
        dt = 1 / 30
        action = action * 2 - 1
        action = sign(action) * abs(action) ** 0.25
        self.last_action = action
        x_acc = a * action
        self.x_vel += x_acc * dt
        self.x_pos += self.x_vel * dt

        theta_acc = g / L * math.sin(self.angle) + math.cos(self.angle) * x_acc / L
        self.theta_vel += theta_acc * dt
        self.angle += self.theta_vel * dt

        if abs(self.angle) > math.pi / 6:
            self.done = True

    def current_state(self):
        return np.array([self.angle, self.theta_vel, self.x_pos, self.x_vel])

    def render(self):
        self.canvas.fill((0, 0, 0))
        position = (self.x_pos * 10 + 300, 300)
        pygame.draw.circle(self.canvas, (255, 255, 255), position, 10)
        pygame.draw.line(self.canvas, (255, 255, 255), position, (position[0] - math.sin(self.angle) * L * 200, position[1] - math.cos(self.angle) * L * 200))
        # draw arrow for last action
        pygame.draw.line(
            self.canvas, 
            (255, 0, 0) if self.last_action > 0 else (0, 255, 0),
            (position[0] + 10 * sign(self.last_action), position[1]), 
            (position[0] + 10 * sign(self.last_action) + 50 * self.last_action, position[1]), width=2
        )
        pygame.display.update()

def sign(x):
    return 1 if x > 0 else -1