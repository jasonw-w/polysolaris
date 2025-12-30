from vector import Vector
import math

import matplotlib.pyplot as plt
class solar_sys_body:
    #consts
    minimum_display_size = 10
    display_log_base = 2

    def __init__(
            self,
            solar_system,
            G,
            mass,
            position=Vector(0,0,0),
            velocity=Vector(0,0,0),
            colour="black",
            stable_orbit=False,
            e=0,#eccentricity
            mass_of_central_body=0,
            position_of_central_body = Vector(0, 0, 0),
            velocity_of_central_body = Vector(0,0,0),
            log_path=None,
            body_id=None,
            dt=None
        ):
        self.dt = dt
        self.solarsys = solar_system
        self.mass = mass
        self.mass_of_central_position = position_of_central_body
        self.velocity_of_central_body = velocity_of_central_body
        self.position = position
        self.stable_orbit = stable_orbit
        self.velocity = Vector(*velocity)
        self.log_path = log_path
        self.id = body_id if body_id is not None else id
        self.counter = 0
        if self.stable_orbit:
            radial = self.position - self.mass_of_central_position
            r_mag = radial.get_magnitude() or 1e-9
            self.r = r_mag
            self.v_circular = math.sqrt(G * mass_of_central_body * ((1 + e) / self.r))
            radial_norm = radial.normalise()
            arbitrary = Vector(0, 0, 1)
            if abs(radial_norm[2]) > 0.9:
                arbitrary = Vector(1, 0, 0)
            tangent = radial_norm.cross(arbitrary)
            tangent_norm = tangent.normalise()
            self.velocity = tangent_norm * self.v_circular + self.velocity_of_central_body
        else:
            self.r = math.sqrt(sum(x**2 for x in position))
        self.solarsys.add_body(self)
        self.display_size = max(math.log(self.mass, self.display_log_base), self.minimum_display_size)
        self.colour = "black" if colour is None else colour
        self.position_history = []
        self.G = G
        self.point_artist = None
        self.trail_artist = None
            
    
    def draw(self):
        # self.solarsys.ax.plot(
        # *self.position,
        # marker = "o",
        # markersize = self.display_size,
        # color=self.colour
        # )
        ax = self.solarsys.ax
        if self.point_artist is None:
             (self.point_artist, ) = ax.plot(
                  [self.position[0]], [self.position[1]], [self.position[2]],
                  marker="o", markersize=self.display_size, color=self.colour, linestyle="None"
             )
        else:
             self.point_artist.set_data_3d(
                  [self.position[0]], [self.position[1]], [self.position[2]]
             )
        if len(self.position_history) > 1:
                positions = self.position_history[-4000:]
                x_vals = [p[0] for p in positions]
                y_vals = [p[1] for p in positions]
                z_vals = [p[2] for p in positions]
                if self.trail_artist is None:
                    (self.trail_artist,) = ax.plot(
                        x_vals, y_vals, z_vals, color=self.colour, alpha=0.3, linewidth=2
                    )
                else:
                    self.trail_artist.set_data_3d(x_vals, y_vals, z_vals)
                    if self.log_path:
                        with open (self.log_path, "a") as f:
                                f.write(f"{self.id}|{self.position}|{self.velocity}\n")

    def move(self):
            dt = getattr(self.solarsys, "dt", self.dt)   
            self.position = Vector (
                self.position[0] + self.velocity[0]*dt,
                self.position[1] + self.velocity[1]*dt,
                self.position[2] + self.velocity[2]*dt
            )
            if self.counter % 100 == 0:
                self.position_history.append(self.position)
            self.counter += 1
    
    def acceleration(self, other):
        dt = getattr(self.solarsys, "dt", self.dt)
        distance = Vector(*other.position) - Vector(*self.position)
        dist_mag = distance.get_magnitude()
        if dist_mag < 1e-9:
            dist_mag = 1e-9
        force_mag = self.G*self.mass*other.mass/(dist_mag**2)
        force = distance.normalise()*force_mag
        reverse = 1
        for body in self, other:
            acceleration = force / body.mass
            body.velocity += acceleration*reverse*dt
            reverse = -1
    
    