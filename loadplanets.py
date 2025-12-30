from body import solar_sys_body
import json
from vector import Vector
# from solar_system import SolarSystemSimulation
class json_loader:
    def __init__(self, solar_sys, G, log_path, dt):
        self.planets = []
        self.solar_sys = solar_sys
        self.G = G
        self.log_path = log_path
        self.dt = dt
    def planet_class_creator(self, simulate, mass, initial_position, initial_velocity, colour, stable_orbit, e, central_body_id, body_id, created_bodies, shift=None):
            if bool(simulate):
                # Try to find central body in already created bodies first (to get updated velocity/pos)
                central_body_instance = created_bodies.get(central_body_id)
                
                central_body_obj = next(
                    (body for body in self.data['planets'] if body['id']==central_body_id),
                    None
                )

                if central_body_instance:
                     central_body_mass = central_body_instance.mass
                     central_body_position = central_body_instance.position
                     central_body_velocity = central_body_instance.velocity
                else:
                    central_body_mass = central_body_obj['mass'] if central_body_obj else 0
                    raw_pos = Vector(*central_body_obj['initial_position']) if central_body_id else Vector(0, 0, 0)
                    if shift:
                        central_body_position = raw_pos + shift  
                    else:
                        central_body_position = raw_pos
                    central_body_velocity = Vector(*central_body_obj['initial_velocity']) if central_body_obj else Vector(0, 0, 0)

                
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
                                velocity_of_central_body=central_body_velocity,
                                e=e,
                                log_path=self.log_path,
                                body_id=body_id,
                                dt=self.dt)
                return x

    def load_data(self, shift, json_path):
        with open (json_path) as f:
            data = json.load(f)
            self.data = data
        planets = []
        planets = []
        created_bodies = {}
        for planet_data in data['planets']:
            planet = self.planet_class_creator(
                simulate=planet_data['simulation'],
                mass=planet_data['mass'], 
                initial_position=Vector(*planet_data['initial_position']) + shift if shift else Vector(*planet_data['initial_position']),
                initial_velocity=Vector(*planet_data['initial_velocity']),
                colour = planet_data['colour'],
                stable_orbit = bool(planet_data['create_stable_orbit']),
                e=planet_data['eccentricity'],
                central_body_id=planet_data['id_of_central_body'],
                body_id=planet_data['id'],
                created_bodies=created_bodies,
                shift=shift
            )
            if planet:
                planets.append(planet)
                created_bodies[planet.id] = planet
        return planets
    
    def load_planets(self, shifts, json_paths):
        planets = []
        for i in range(len(json_paths)):
            planets.append(self.load_data(shift=shifts[i], json_path=json_paths[i]))
        return planets

# if __name__ == "__main__":
#     sys = SolarSystemSimulation
#     loader = json_loader(path=r"planets.json", solar_sys=sys, G=1)
#     print(str(loader.load_planets()))




