import sympy as sym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# PARABOLAS: kind of hill-shaped
# GENERAL FORMULA:
# 4*p(y-k) = (x - h)**2
# k and h shift y and x respectively
# if k, h = 0 -> 4*p*y = x**2
# solve for y:
# y = (1/4*p) * (x - h)**2 + k


# example 1:
# params
# When you rewrite into the familiar “y=a(x−h)2+k” form, the coefficient in front of (x−h)2 is a
a = 1  # 1/4p
h = 1  # x-axis offset
k = -2

n = 100

# x-axis points to evalute the function
# h is the x-coordinate of the vertex
# “Take a bunch of x-values (n points) that go from h−5 to h+5.”
# Since the parabola’s vertex is at x=h,
# this gives you a symmetric interval around the vertex (so you see the parabola nicely centered).
x = np.linspace(h-5, h+5, n)

# add params in parabola formula
y = a*(x-h)**2 + k

plt.plot(x, y)

plt.grid()
plt.show()  # a vertical parabola

# example 2: get a horizontal parabola
# just plot x and y reverse
plt.plot(y, x)
plt.show()

# exercise: plot the parabola, its directix, vertex and focus
# vertex: point of max curve
# focus and directrix: parabola curve points' distance from directrix is equal as distance from focus

h = 1  # max point/curve is 2
k = -2
a = 1
p = 1/(4*a)
n = 100
x = np.linspace(h-2, h+2, n)
parabola = a * (x - h)**2 + k
vertex = (h, k)
focus = (h, k + p)
directrix = k - p

plt.plot(x, parabola, label='parabola')
plt.plot(vertex[0], vertex[1], 'ro', label='Vertex')
plt.plot(focus[0], focus[1], 'go', label='focus')

# directrix is confusing
plt.plot([x[0], x[-1]], [directrix, directrix], 'y-', label='directrix')

# plt.ylim(-2, 2)
plt.legend()
plt.grid()
plt.show()

# CREATING CONTOURS from meshes in Python
X, Y = np.meshgrid(range(0, 10), range(0, 15))

plt.subplot(121)
plt.pcolormesh(X, edgecolor='k', linewidth=.1)

plt.subplot(122)
plt.pcolormesh(X, edgecolor='k', linewidth=.1)


plt.show()


# set mesh
x = np.linspace(-np.pi, 2*np.pi, 40)
y = np.linspace(0, 4*np.pi, 72)

X, Y = np.meshgrid(x, y)
# create a function of x and y
Fxy = np.cos(X) + np.sin(Y)
# cosine across columns, sine across rows

plt.imshow(Fxy)
plt.show()

# exercise: plot a 2D Gaussian, image smoothing kernel, blur
x = np.linspace(-3, 3, 1001) # if we use even number of steps, we do not get step 0 exactly.
y = np.linspace(3, -3, 1001)
h = 0.9  # 1 is max
a = 0.05
X, Y = np.meshgrid(x, y)

gaussian_x = np.exp(-a*np.log(2)*X**2 / h**2)
gaussian_y = np.exp(-a*np.log(2)*Y**2 / h**2)

# here it's g(x) * g(y) for g(x, y), not g(x) + g(y)
Fxy = gaussian_x * gaussian_y

plt.imshow(Fxy)


plt.show()