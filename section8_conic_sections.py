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

### graphing circles
# using meshgrids for circles
# (h, k) -> h coordinate on x axis, k on y
# formula:
# r**2 = (y - k)**2 + (x - h)**2
# if circle originates at origin:
# r**2 = y**2 + x**2

h = 2
k = -3
radius = 3

# most extreme point on the circle h and k + r
axlim = radius + np.max((abs(h),abs(k)))

x = np.linspace(-axlim, axlim, 100)
X,Y = np.meshgrid(x, x)

Fxy = (X-h)**2 + (Y-k)**2 - radius**2

# contour mapping -> constant elevations along the contour lines
plt.contour(X, Y, Fxy, 0)
plt.plot([-axlim, axlim],[0, 0], 'k--')
plt.plot([0, 0], [-axlim, axlim], 'k--')
plt.plot(h, k, 'go')

plt.gca().set_aspect('equal')  # get current axis
plt.show()

# exercise: two circles that overlap -> diff radii and the circles at 2 diff locs
# interference patterns
# thick black line from one center to the other
h1 = -1.5
h2 = 1.5
k = 2
radii = np.linspace(0.5, 5, 15)


x = np.linspace(-5, 5, 100)

center1 = (h1, k)
center2 = (h2, k)
for i, radius in enumerate(radii):
    X, Y = np.meshgrid(x, x)
    Fxy1 = (X-h1)**2 + (Y-k)**2 - radius**2
    Fxy2 = (X - h2)**2 + (Y - k)**2 - radius**2
    plt.contour(X, Y, Fxy1, levels=[0], colors=[(i/3, i/3, i/3)]) # instead of cmap, use RGBA ()
    plt.contour(X, Y, Fxy2, levels=[0], colors=[(i/3, i/3, i/3)])

plt.gca().set_aspect('equal')  # get current axis
plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 'k-', linewidth=5)
plt.axis('off')
plt.show()


# graphing ellipses
# kinda of an egg shape
# an ellipse has 2 radii: a and b
# a -> x component of radius
# b -> y component of radius
# center (h, k)
# formula:
# (y - k) **2 / b**2 + (x - h)**2/a**2 -1 = 0
# if a = b -> circle, not ellipse

h = 1
k = 2
a = 2
b = 3

axlim = np.max((a,b)) + np.max((abs(h), abs(k)))
x = np.linspace(-axlim, axlim, 100)

X, Y = np.meshgrid(x, x)

# create the function
ellipse = (X-h)**2/(a**2) + (Y-k)**2 / (b**2) -1

plt.contour(X, Y, ellipse, 0)
plt.plot(h, k, 'go')

plt.gca().set_aspect('equal')
plt.show()

# exercise:
# vary k from -4 to 4
# vary a from abs -4 to 4
k = np.linspace(-4, 4, 20)
k_max = int(np.max(k))
a = abs(np.linspace(-4, 4, 20))
a_max = int(np.max(a))
b = 1  # radius b -> looks cool -> how circular each ellipsis will look like
h = 0  # x axis off center

axlim = np.max((a_max, b)) + np.max((abs(k_max), abs(h)))
x = np.linspace(-axlim, axlim, 100)

X, Y = np.meshgrid(x, x)

for i in range(0, 20):
        a_val = a[i]
        k_val = k[i]
        ellipse = (X - h) ** 2 / (a_val ** 2) + (Y - k_val) ** 2 / (b ** 2) - 1
        plt.contour(X, Y, ellipse, 0, colors=[(i/20, 0, i/20)])

plt.axis('off')
plt.gca().set_aspect('equal')
plt.show() # yay, I wasn't far off. just needed to specify level=0, and b = 4, and h = 0

# graphing HYPERBOLAS -> waist-looking plot
a = 1
b = 0.5
h = 1
k = 2

axlim = np.max((a, b)) + np.max((abs(k), abs(h))) * 2 # almost the same as for ellipsis

x = np.linspace(-axlim, axlim, 100)
X,Y = np.meshgrid(x, x)

# let's create the function:
hyperbola = (X-h)**2/a**2 - (Y-k)**2/b**2 - 1
# h + k are shifting, a + b are stretching!!!!

plt.contour(X, Y, hyperbola, 0)
plt.plot(h, k, 'go')
plt.plot([-axlim, axlim], [0, 0], color=(0.8, 0.8, 0.8))
plt.plot([0, 0], [-axlim, axlim], color=(0.8, 0.8, 0.8)) # color overwrites k--
plt.gca().set_aspect('equal')


plt.show()

# hyperbola exercise:
# vary a from ? to ?
# b from ? to ?
# if h = k = 0

n = 20
h, k = 0,0
a = np.linspace(1, 4, n)
a_max = int(np.max(a))
b = np.linspace(1, 4, n)
b_max = int(np.max(b))


axlim = np.max((a_max,b_max)) + np.max((h, k)) * 2

x = np.linspace(-axlim, axlim, n)

X, Y = np.meshgrid(x, x)

for i in range(0, n):
    a_val = a[i]
    b_val = b[i]
    hyperbola1 = (X-h)**2/a_val**2 - (Y-k)**2/b_val**2 - 1
    plt.contour(hyperbola1, 0, colors=[(i/n, 0, i/n)])
    hyperbola2 = - (X-h)**2/a_val**2 + (Y-k)**2/b_val**2 - 1
    plt.contour(hyperbola2, 0, colors=[(0, i/n, i/n)])

plt.axis('off')
plt.gca().set_aspect('equal')
plt.show()


# bug hunt:
# 1: 2D gaussian
x = np.linspace(-2, 2, 100)
# gaussian curve:
X, Y = np.meshgrid(x, y)
a, h = 4, 1 # first define a and h
# for g(x, y) = g(x) * g(y)
gauss2d_x = np.exp(-a*np.log(2)*X**2 / h**2)
gauss2d_y = np.exp(-a*np.log(2)*Y**2 / h**2)

Gxy = gauss2d_x * gauss2d_y

plt.imshow(Gxy)
plt.axis('off')
plt.show()

#2.  draw a circle using meshgrid
r = 3
# grid space
x = np.linspace(-r, r, 100)
X, Y = np.meshgrid(x, x) # second input was missing

# create function
Fxy = X**2 + Y**2 - r**2

# draw it:
plt.contour(Fxy, 0)
plt.axis('off')
plt.show()

#3. params (ellipse)
a = 1
b = 2
h = 2
k = -3

axlim = np.max((a, b)) + np.max((abs(h), abs(k)))
x = np.linspace(-axlim, axlim, 100)
y = np.linspace(-axlim, axlim, 100)

X,Y = np.meshgrid(x, y)

# create function
ellipsis = (X-h)**2/(a**2) + (Y-k)**2 / (b**2) -1

# draw as contour:
plt.contour(X, Y, ellipsis, 0)
plt.plot(h, k, 'go')
plt.grid()
plt.title('Ellipsis centered at (x, y) = (%s, %s)' %(h,k))
plt.gca().set_aspect('equal')
plt.show()

# 4. hyperbola, not an X:
# params
a, b, h, k = 1, .5, 1, 2
# grid space:
axlim = np.max((a, b)) + np.max((abs(h), abs(k))) * 2
x = np.linspace(-axlim, axlim, 100)
y = np.linspace(-axlim, axlim, 100)

X, Y = np.meshgrid(x, y)

# function:
hyperbola = (X-h)**2 / a**2 - (Y-k)**2 / b**2 - 1

# contour
plt.contour(X, Y, hyperbola)
# dot in center:
plt.plot(h, k, 'go')
# guide lines
plt.plot([-axlim, axlim], [0, 0], 'k--')
plt.plot([0, 0], [-axlim, axlim], 'k--')

plt.gca().set_aspect('equal')
plt.show()