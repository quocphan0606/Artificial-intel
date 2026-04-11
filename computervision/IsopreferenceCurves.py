import numpy as np
import matplotlib.pyplot as plt

# trục N (số pixel)
N = np.linspace(50, 1000, 200)

# mô phỏng các đường isopreference
k1 = 10 - 0.004*N      # ảnh ít chi tiết (face)
k2 = 9 - 0.0025*N      # chi tiết trung bình (cameraman)
k3 = 8 - 0.0008*N      # ảnh nhiều chi tiết (crowd)

plt.figure(figsize=(8,6))

plt.plot(N, k1, label="Low detail image (Face)")
plt.plot(N, k2, label="Medium detail image (Cameraman)")
plt.plot(N, k3, label="High detail image (Crowd)")

plt.xlabel("N (Spatial Resolution - Number of Pixels)")
plt.ylabel("k (Intensity Resolution - Bits)")
plt.title("Isopreference Curves in N-k Plane")

plt.legend()
plt.grid(True)

plt.show()