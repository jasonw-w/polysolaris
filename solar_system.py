import matplotlib.pyplot as plt
import math
from body import solar_sys_body
import itertools
class SolarSystemSimulation:

    def __init__(self, size, G, log_path, dt):
        self.size = size
        self.bodies = []
        self.log_path = log_path,
        self.dt = dt
        self.fig, self.ax = plt.subplots(
            1,
            1,
            subplot_kw={"projection" : "3d"},
            figsize=(10, 10)
        )
        self.fig.tight_layout()
        self.ax.view_init(90, -90)

    def add_body(self, body):
        self.bodies.append(body)

    def update_all(self, draw=False):
        for body in self.bodies:
            body.move()
            if draw:
                body.draw()

    def draw_all(self):
        self.bodies.sort(key=lambda item: item.position[0])
        self.ax.set_xlim((-self.size/2, self.size/2))
        self.ax.set_ylim((-self.size/2, self.size/2))
        self.ax.set_zlim((-self.size/2, self.size/2))
        plt.pause(0.001)
        for body in self.bodies:
            if len(body.position_history) > 2500:  # Limit history size
                body.position_history = body.position_history[-2500:]
    
    def calculate_body_interactions(self):
        copy = self.bodies.copy()
        for idx, first in enumerate(copy):
            for second in copy[idx+1:]:
                first.acceleration(second)


