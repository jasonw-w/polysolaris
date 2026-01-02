import matplotlib.pyplot as plt
import matplotlib
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import math
from .body import solar_sys_body
import itertools
from .solar_system import SolarSystemSimulation
from .loadplanets import json_loader
import time
import matplotlib.animation as animation
from .vector import Vector
count = 0
G = 1.0
# log_path = r"simulation.txt"
log_path = None
dt = 1e-8
record = False
quick_sim = False #Barnes-Hut Algorithm
# json_paths = [r"solar_system.json", r"solar_system2.json"]
# json_paths = [r"solar_system2.json"]
import os
# dynamic path finding
from importlib import resources
data_path1 = str(resources.files("system_simulation").joinpath("data/solar_system.json"))
json_paths = [data_path1]
shift = [Vector(0, 0, 0)]
def main(G: float, log_path: str, dt: float, record: bool, quick_sim: bool, json_paths: list):
    solarsys = SolarSystemSimulation(100, G, log_path, dt)
    loader = json_loader(solarsys, G, log_path, dt)
    planets = loader.load_planets(shift, json_paths)
    quick_sim = False
    # for planet in planets:
    #     solarsys.add_body(planet)
    # counter = 0
    # while True:
    #     draw = (counter % 100 == 0)
    #     solarsys.calculate_body_interactions()
    #     solarsys.update_all(draw)
    #     if draw:
    #         solarsys.draw_all()
    #     counter += 1
    def update(frame):
        global count
        count += 1
        if count % 100 == 0:
            print(count*100)
        for _ in range(100):
            solarsys.calculate_body_interactions(quick_sim)
            solarsys.update_all(draw=False)
        solarsys.update_all(draw=True)
        solarsys.draw_all()
    anim = animation.FuncAnimation(
        fig=solarsys.fig,
        func=update,
        frames=60*50,
        interval=1
    )
    if record:
        print("Recording")
        anim.save(
        'simulation.mp4',
        writer='ffmpeg',
        fps=30
        )
        print("video saved")
    else:
        plt.show()

def run():
    main(
        G=G,
        log_path=log_path,
        dt=dt,
        record=record,
        quick_sim=quick_sim,
        json_paths=json_paths
    )

if __name__ == "__main__":
    run()