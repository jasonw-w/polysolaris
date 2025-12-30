import matplotlib.pyplot as plt
import math
from body import solar_sys_body
import itertools
from matplotlib.widgets import Slider, Button
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
        self.fig.subplots_adjust(bottom=0.2, top=1, right=1, left=0)
        self.ax.dist=0
        self.ax.view_init(45, -45)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        speed_ax = plt.axes([0.25, 0.02, 0.5, 0.03])
        self.speed_slider = Slider(
            ax=speed_ax,
            label='time(dt)',
            valmin=1e-9,
            valmax=1e-5,
            valinit=dt
        )
        self.speed_slider.on_changed(self.update_speed)

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            zoom_factor = 1.1
            if event.button == 'up':
                self.size /= zoom_factor
            if event.button == 'down':
                self.size *= zoom_factor
            if self.size < 1:
                self.size = 1
            if self.size > 1000:
                self.size = 1000

    def update_speed(self, val):
        self.dt=val

    def add_body(self, body):
        self.bodies.append(body)

    def update_all(self, draw=False):
        bodies_to_remove = []
        for body in self.bodies:
            if abs(body.position[0]) > 200 or abs(body.position[1]) > 200 or abs(body.position[2]) > 200:
                bodies_to_remove.append(body)
            body.move()
            if draw:
                body.draw()
        
        for body in bodies_to_remove:
            if body in self.bodies:
                self.bodies.remove(body)
            if body.point_artist:
                body.point_artist.remove()

    def draw_all(self):
        # Remove bodies that are out of bounds
        bodies_to_remove = []
        # Re-check or pass from update_all? 
        # Ideally update_all handles removal, but bodies_to_remove is local there.
        # I will modify update_all to include the removal loop.

        self.bodies.sort(key=lambda item: item.position[0])
        self.ax.set_xlim((-self.size/2, self.size/2))
        self.ax.set_ylim((-self.size/2, self.size/2))
        self.ax.set_zlim((-self.size/2, self.size/2))
        # plt.pause(0.001)
        for body in self.bodies:
            if len(body.position_history) > 2500:  # Limit history size
                body.position_history = body.position_history[-2500:]
    
    def calculate_body_interactions(self, quick_sim):
        if not quick_sim:
            copy = self.bodies.copy()
            for idx, first in enumerate(copy):
                for second in copy[idx+1:]:
                    first.acceleration(second)
        else:
            pass


