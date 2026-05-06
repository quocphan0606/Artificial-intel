import matplotlib.pyplot as plt
from  Point import Spiral, Line, Circle,Zone,Zone_3D
data = Spiral(100, 4, 2)
P, L = data.generate()

plt.scatter(P[:,0], P[:,1], c=L, cmap='viridis')
plt.show()