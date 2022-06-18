from Environment import Environment
import math
import numpy as np
import pygame

DRAG_COEFFICIENT = 0.008
ROLLING_RESISTANCE = 0.1
ENGINE_FORCE = 300
L = 32  # distance between wheels
W = 25
H = 50
DT = 1 / 15

SCREEN_WIDTH = 1240
SCREEN_HEIGHT = 640

PATH_COLOR = (100, 60, 20)

MAX_SENSOR_DISTANCE = 250
MIN_SENSOR_STEP = 2

class CarDrivingEnv(Environment):
    def __init__(self):
        # generate random points for path
        self.path = np.array([[100, 100], [200, 350], [200, 500], [600, 600], [900, 400], [400, 100]])
        # add first again
        self.path = np.append(self.path, self.path[0])
        self.path = self.path.reshape((7, 2))

        super().__init__()

        # pygame winodw
        pygame.init()
        self.canvas = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.path_radius = 50

        self.ray_angles = [math.radians(-45), 0, math.radians(45)]
        self.last_raycasts = [0, 0, 0]
        self.plot = None
        self.plot_width = None
        self.quit = False

    def reset(self):
        super().reset()

        self.angle = 0
        self.pos = (self.path[0][0], self.path[0][1])
        self.vel = 0
        self.angle = 0
        self.last_action = [0, 0]
        self.flags = 0
        self.closest_index = 0
        self.t = 0
        self.laps = 0

    def action_effects(self, action):
        """ returns acceleration factor and wheel angle """
        f = lambda x: abs(x) ** 0.4 * sign(x)
        return [action[0] * 2 - 1, f(action[1] * 2 - 1) * math.pi / 4]

    def update_state(self, action):
        dt = DT
        # get acceleration
        acc_factor, wheel_angle = self.action_effects(action)
        acc = acc_factor * ENGINE_FORCE
        acc -= DRAG_COEFFICIENT * self.vel * abs(self.vel) + ROLLING_RESISTANCE * self.vel

        # calc angle (-45 to 45 degrees)
        wheel_angle = wheel_angle
        if(wheel_angle != 0):
            R = L / math.sin(wheel_angle)
            angular_vel = self.vel / R
            turn_resistance = 0.75 * abs(angular_vel) * self.vel
            acc -= turn_resistance
            self.angle += angular_vel * dt

        # update velocity
        self.vel += acc * dt
        # update position
        self.pos += self.vel * dt * np.array([math.sin(self.angle), math.cos(self.angle)])

        self.path_dist = self.distance_from_path()
        if(self.path_dist > self.path_radius):
            self.done = True

        self.last_action = action
        self.update_flags()

    def update_flags(self):
        # get new closest index from path points
        next_goal = self.path[self.closest_index + 1]
        if magnitude(next_goal - self.pos) < self.path_radius:
            self.closest_index += 1
            # check if we are at the end of the path
            if self.closest_index == len(self.path) - 1:
                self.closest_index = 0
                self.laps += 1
            next_goal = self.path[self.closest_index + 1]
        # check distance from previous goal
        prev_goal = self.path[self.closest_index]
        
        a = magnitude(prev_goal - self.pos)
        b = magnitude(next_goal - self.pos)
        self.t = a / (a + b)

    def distance_from_path(self, P=None, check_next=False):
        if(P is None):
            P = self.pos
        # get closest point on path
        closest_distance = self.distance_from_segment(P, self.path[self.closest_index], self.path[self.closest_index + 1])
        if check_next:
            next_distance = self.distance_from_segment(P, self.path[self.closest_index + 1], self.path[self.closest_index + 2])
            if next_distance < closest_distance:
                closest_distance = next_distance
        # get distance from closest point
        return closest_distance

    def distance_from_segment(self, P, A, B):
        # segment: P = A + t * (B - A) {t in [0, 1]}
        t = np.dot(P - A, B - A) / np.dot(B - A, B - A)
        # clamp t to [0, 1]
        t = max(0, min(1, t))
        # get closest point on segment
        Q = A + t * (B - A)
        # get distance
        return np.linalg.norm(P - Q)

    def raycast(self, angle):
        angle += self.angle
        # get ray start and end
        l = MAX_SENSOR_DISTANCE / 2
        step_size = l / 2
        while step_size > MIN_SENSOR_STEP:
            d = self.distance_from_path(self.pos + l * np.array([math.sin(angle), math.cos(angle)]), True)
            if d < self.path_radius:
                l += step_size
            else:
                l -= step_size
            step_size /= 2
        # return ray length
        return l

    def current_state(self):
        # get raycasts
        data = []
        for angle in self.ray_angles:
            data.append(self.raycast(angle) / 50)
        self.last_raycasts = np.array(data) * 50
        return np.array([self.vel / 5] + data)

    def render(self, gen):
        self.canvas.fill((0, 0, 0))
        position = np.array([self.pos[0], self.pos[1]])
        self.angle *= -1

        # draw path
        for i in range(len(self.path) - 1):
            self.draw_thick_line(PATH_COLOR, self.path[i], self.path[i + 1], self.path_radius * 2)
            pygame.draw.circle(self.canvas, PATH_COLOR, self.path[i], self.path_radius)
        # draw path points
        for i in range(len(self.path) - 1):
            pygame.draw.circle(self.canvas, (100, 255, 100), self.path[i], 4)
        # draw connections to last path point
        pygame.draw.line(self.canvas, (0, 150, 0), self.pos, self.path[self.closest_index], 1)
        pygame.draw.line(self.canvas, (0, 150, 0), self.pos, self.path[self.closest_index + 1], 1)

        # draw rays
        for i in range(len(self.ray_angles)):
            pygame.draw.line(self.canvas, (255, 0, 0), self.pos, self.pos + self.last_raycasts[i] * np.array([math.sin(self.ray_angles[i] - self.angle), math.cos(self.ray_angles[i] - self.angle)]), 2)

        # draw back wheels
        self.draw_rect((100, 100, 100), 
            position + self.get_rotated_position(W / 2, -L / 2, self.angle), 
            6, 15, self.angle
        )
        self.draw_rect((100, 100, 100), 
            position + self.get_rotated_position(-W / 2, -L / 2, self.angle), 
            6, 15, self.angle
        )
        # draw front wheels
        wheel_angle = -self.action_effects(self.last_action)[1]
        self.draw_rect((100, 100, 180),
            position + self.get_rotated_position(W / 2, L / 2, self.angle),
            6, 15, self.angle + wheel_angle
        )
        self.draw_rect((100, 100, 180),
            position + self.get_rotated_position(-W / 2, L / 2, self.angle),
            6, 15, self.angle + wheel_angle
        )
        # draw car (rect at angle)
        self.draw_rect((255, 255, 255), position, W, H, self.angle)
        # draw front window
        self.draw_rect((200, 200, 200), position + self.get_rotated_position(0, H / 2 - 18, self.angle), W, 14, self.angle)
        self.draw_rect((100, 100, 100), position + self.get_rotated_position(0, H / 2 - 18, self.angle), W - 5, 10, self.angle)

        self.draw_graph()

        # draw text
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(f"Score: {self.score():.2f}", True, (255, 255, 255))
        text2 = font.render(f"Laps: {str(self.laps)} + {self.closest_index}/{len(self.path) - 1}", True, (255, 255, 255))
        text3 = font.render(f"Gen: {gen}", True, (255, 255, 255))
        self.canvas.blit(text, (10, 5))
        self.canvas.blit(text2, (10, 25))
        self.canvas.blit(text3, (SCREEN_WIDTH - 100, 5))

        self.angle *= -1
        pygame.display.update()

        self.pygame_event_update()

    def render_progress(self, done, total):
        pygame.draw.rect(self.canvas, (0, 0, 0), (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, 250, 50))
        # draw text
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(f"Simulation: {int(done / total * 100)}% done", True, (255, 255, 255))
        self.canvas.blit(text, (SCREEN_WIDTH - 240, SCREEN_HEIGHT - 50))
        # draw progress bar (bottom right)
        pygame.draw.rect(self.canvas, (255, 255, 255), (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 20, 250 * done / total, 20))
        pygame.display.update()

        self.pygame_event_update()

    def pygame_event_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True

    def draw_graph(self, data=None):
        """
        draws graph on screen
        input: data - tuple (image bytes, (width, height))
        """
        if data is not None:
            self.plot = pygame.image.fromstring(data[0], data[1], "RGB")
            self.plot_width = data[1][0]
        if self.plot is None:
            return
        self.canvas.blit(self.plot, (SCREEN_WIDTH - self.plot_width, 0))

    def draw_rect(self, color, pos, w, h, angle):
        corner_positions = [(w/2, h/2), (-w/2, h/2), (-w/2, -h/2), (w/2, -h/2)]
        corners = []
        for corner_pos in corner_positions:
            corners.append(pos + self.get_rotated_position(corner_pos[0], corner_pos[1], angle))
        # draw
        pygame.draw.polygon(self.canvas, color, corners)

    def draw_thick_line(self, color, start, end, width):
        # get center
        center = (start + end) / 2
        # get angle
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        # get length
        length = magnitude(end - start)
        # draw
        self.draw_rect(color, center, length, width, angle)

    def get_rotated_position(self, x, y, angle):
        return np.array([
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle)
        ])

    def score(self):
        return len(self.path) * self.laps + self.closest_index + self.t


def sign(x):
    return 1 if x > 0 else -1

def magnitude(x):
    return math.sqrt(sum([a ** 2 for a in x]))