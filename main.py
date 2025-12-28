import matplotlib.pyplot as plt
import math
from body import solar_sys_body
import itertools
from solar_system import SolarSystemSimulation
from loadplanets import json_loader
import time
import matplotlib.animation as animation
G = 1
# log_path = r"simulation.txt"
log_path = None
dt = 0.01
solarsys = SolarSystemSimulation(8000, G, log_path, dt)
loader = json_loader(r"solar_system.json", solarsys, G, log_path, dt)
planets = loader.load_data()
# for planet in planets:
#     solarsys.add_body(planet)
counter = 0
while True:
    draw = (counter % 100 == 0)
    solarsys.calculate_body_interactions()
    solarsys.update_all(draw)
    if draw:
        solarsys.draw_all()
        time.sleep(0.001)
    counter += 1