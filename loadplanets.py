from body import solar_sys_body
import json
from vector import Vector
# from solar_system import SolarSystemSimulation
class json_loader:
    def __init__(self, path, solar_sys, G):
        self.json_path = path
        self.planets = []
        self.solar_sys = solar_sys
        self.G = G
    def planet_class_creator(self, simulate, mass, initial_position, initial_velocity, colour, stable_orbit, e, central_body_id):
            if bool(simulate):
                central_body_obj = next(
                    (body for body in self.data['planets'] if body['id']==central_body_id),
                    None
                )
                central_body_mass = central_body_obj['mass'] if central_body_obj else 0
                central_body_position = Vector(*central_body_obj['initial_position']) if central_body_obj else Vector(0, 0, 0)
                if colour == None:
                    colour = 'black'
                if not stable_orbit:
                    central_body_mass=0
                    e=0
                x = solar_sys_body(solar_system=self.solar_sys,
                                mass=mass,
                                position=initial_position,
                                velocity=initial_velocity,
                                colour=colour,
                                G=self.G,
                                mass_of_central_body=central_body_mass,
                                stable_orbit=stable_orbit,
                                position_of_central_body=central_body_position,
                                e=e)
                return x

    def load_data(self):
        with open (self.json_path) as f:
            data = json.load(f)
            self.data = data
        planets = []
        for planet in data['planets']:
            planet = self.planet_class_creator(
                simulate=planet['simulation'],
                mass=planet['mass'], 
                initial_position=Vector(*planet['initial_position']), 
                initial_velocity=Vector(*planet['initial_velocity']),
                colour = planet['colour'],
                stable_orbit = bool(planet['create_stable_orbit']),
                e=planet['eccentricity'],
                central_body_id=planet['id_of_central_body']
            )
            planets.append(planet)
        return planets
    
    def load_planets(self):
        solarsys = self.solar_sys(400)
        planets = self.load_data()
        return planets

# if __name__ == "__main__":
#     sys = SolarSystemSimulation
#     loader = json_loader(path=r"planets.json", solar_sys=sys, G=1)
#     print(str(loader.load_planets()))




