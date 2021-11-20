import numpy as np
import time
from matplotlib import pyplot as plt

# Question 1a
xs, ys, zs = np.loadtxt('rand_points.txt').T
theta = 0, 60
            
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs=zs, s=0.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(*theta)
plt.tight_layout()
plt.savefig('Results/q1a.png')
plt.close()

# Question 2b
n = int(3e7)
random_ints = np.random.randint(0, 2**31, size=(n, 3))

max_thresh = 1e8
max_vals = np.max(random_ints, axis=1)
idx = np.where(max_vals < max_thresh)
xs, ys, zs = random_ints[idx].T

fig = plt.figure(figsize=(12, 18))
for i, theta in enumerate(np.linspace(0, 180, 12)):
    ax = fig.add_subplot(4, 3, i+1, projection='3d')
    ax.scatter(xs, ys, zs=zs, s=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f'Rotation: {theta:.2f} degrees')
    ax.view_init(0, theta)

plt.tight_layout()
plt.savefig('Results/q1b.png')


# Question 2