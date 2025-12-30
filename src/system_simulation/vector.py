import math
import numbers
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"
    
    def __str__(self):
        return f"{self.x}i, {self.y}j, {self.z}k"
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        else:
            raise IndexError("There are only 3 components in a Vector")
    def __add__(self, other):
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other):
        return Vector (
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector (
                self.x*other.x,
                self.y*other.y,
                self.z*other.z
            )
        elif isinstance(other, numbers.Real):
            return Vector (
                self.x*other,
                self.y*other,
                self.z*other
            )
        else:
            raise TypeError("operand must be Vector, int or float")
        
    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vector(
                self.x / other,
                self.y / other,
                self.z / other
            )
    
    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("There are only 3 component in a vector")
    def get_magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalise(self):
        mag = self.get_magnitude()
        return Vector(self.x/mag, self.y/mag, self.z/mag)
    
    def cross(self, other):
        return Vector (
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
                                 
if __name__ == "__main__":
    test = Vector(1,2,3)
    test2 = Vector(2,3,4)
    print(test - test2)
    print(test.normalize())
