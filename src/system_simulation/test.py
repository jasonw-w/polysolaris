from vector import Vector
v1 = Vector (3, 4, 5)
v2 = Vector (1.5, 0, 4)
print(sum(v1*v2))
result = (0-(v1[0]*v2[0] + v1[2]*v2[2]))/v1[1]
v2[1] = result
print(sum(v1*v2))
print(5e8)