import matplotlib.pyplot as plt
import math
from body import solar_sys_body
import itertools
from solar_system import SolarSystemSimulation
from loadplanets import json_loader
G = 1
log_path = r"simulation.txt"
solarsys = SolarSystemSimulation(4000, G, log_path)
loader = json_loader(r"solar_system.json", solarsys, G)
planets = loader.load_data()
# for planet in planets:
#     solarsys.add_body(planet)

while True:
    solarsys.calculate_body_interactions()
    solarsys.update_all()
    solarsys.draw_all()