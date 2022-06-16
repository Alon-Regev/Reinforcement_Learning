from Environment import Environment
import math
# pygame
import pygame

g = 9.8
L = 0.5
a = 5

class CartPoleEnv(Environment):
    def __init__(self):
        super().__init__()

        # pygame winodw
        self.canvas = pygame.display.set_mode((600, 600))

        self.angle = 0.0
        self.x_pos = 0.0
        self.x_vel = 0.0
        self.theta_vel = 0.0

    def reset(self):
        super().reset()

    def update_state(self, action):
        dt = 1 / 30
        if action < 1/3:
            x_acc = -a
        elif action < 2/3:
            x_acc = 0
        else:
            x_acc = a
        self.x_vel += x_acc * dt
        self.x_pos += self.x_vel * dt

        theta_acc = g / L * math.sin(self.angle) + math.cos(self.angle) * x_acc / L
        self.theta_vel += theta_acc * dt
        self.angle += self.theta_vel * dt

    def current_state(self):
        return [self.angle, self.x_vel]

    def rewards(self):
        raise NotImplementedError

    def render(self):
        self.canvas.fill((0, 0, 0))
        position = (self.x_pos * 100 + 300, 300)
        pygame.draw.circle(self.canvas, (255, 255, 255), position, 10)
        pygame.draw.line(self.canvas, (255, 255, 255), position, (position[0] - math.sin(self.angle) * L * 200, position[1] - math.cos(self.angle) * L * 200))
        pygame.display.update()