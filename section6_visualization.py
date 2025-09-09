# PLOTTING COORDINATES ON A PLANE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
x = 3
y = 5

plt.plot(x, y, 'kp')  # r for red and o for circle
# other options: 'gs' (green square), kp (black pentagon)

# how to make it better:
plt.axis('square') # the way the commands are ordered affect the final visualization
# order is important
plt.axis([-6, 6, -6, 6])
plt.grid()
# and now changing 0,0 at its origin

plt.show()

# example 2:
x = [-4, 5, 2, 4, 5, 6]
y = [2, 5, 6, 8, 8, 9]

for i in range(len(x)):
    plt.plot(x[i], y[i], 'o', label='point %s' %(1+i))

plt.axis('square')
plt.grid()
plt.legend() # needed to activate the label

plt.show()

plt.plot(4, 2, 'r+')
# now accessing this plot object
axis = plt.gca() # get current access to the plot
ylim = axis.get_ylim() # how we get the actual ylim
print(ylim)
# now let's change ylim:
axis.set_ylim([ylim[0], 5.3567])

plt.show()

# Exercise:
# plot
import sympy as sym
x = sym.symbols('x')
eq = x**2 - 3*x

int_list = list(range(-10, 12))
solutions_y = []
for integer in int_list:
    solution = eq.subs({x:integer})
    solutions_y.append(solution)
    plt.plot(integer, solution, 'o')

plt.xlabel('x')
plt.ylabel(f'f(x)={eq}')
# plt.axis('square')
plt.axis([-10, 10, -4, 120])
plt.grid()
plt.show()

# GRAPHING LINES
# how to describe a line using start/stop form
# how to plot with colors
p1 = [-3, -1]
p2 = [4, 4]
# plt.plot(p1, p2)  # FALSE
plt.plot([p1[0], p2[0]],[p1[1], p2[1]], color=[1, 0, 0], linewidth=4)
# what does 1, 0, 0 mean:
# 100% RED, 0% GREEN, 0% BLUE RBG

plt.axis('square')
plt.axis([-6, 6, -6, 6])
plt.show()

# next:
x = 3
y = 5
plt.plot(x, y, 'ro')
# how to plot a line through the origin of the plot
plt.plot([0, x], [0, y])

plt.axis('square')
plt.axis([-6, 6, -6, 6])
plt.grid()

plt.plot([-6, 6], [0, 0], 'k') # the horizontal line is bolder
plt.plot([0, 0], [-6, 6], 'k') # the vertical line is bolder


# exercise 2:
import numpy as np
x, y = sym.symbols('x y')
y = sym.sqrt(sym.Abs(x))
x_list = list(range(-20, 21))

for x_value in x_list:
    solution = y.subs({x:x_value})
    plt.plot([0, x_value], [0, solution], linewidth=3)


plt.xticks(np.arange(-20, 21, 5))
plt.yticks(np.arange(0, 5, 1))
plt.xlabel('x')
plt.ylabel('y')

plt.show()

# now plot a square in the middle
black_bottom = [0, 2], [0, 0]
red_top = [0, 2], [2, 2]
green_left = [0, 0], [0, 2]
magenta_right = [2, 2], [0, 2]

colors_list = ['k', 'r', 'g', 'm']

lines_list = [black_bottom, red_top, green_left, magenta_right]

for lines, col in zip(lines_list, colors_list):
    plt.plot(lines[0], lines[1], col)


plt.axis('square')

plt.xticks(np.arange(-2, 5, 2))
plt.yticks(np.arange(-3, 5, 1))

plt.show()

# slope intercept
x, y, m, b = sym.symbols('x, y, m, b')
y = m*x + b # m = slope, b = intercept (when x = 0, then y = b)
x = [-5, 5]
m = 2
b = 1

y = [340, 204]

for i in range(0, len(x)):
    y[i] = m * x[i] + b

plt.plot(x, y)
plt.axis('square')
plt.grid()

plt.xlim(x)
plt.ylim(x)

axis = plt.gca()
plt.plot((axis.get_xlim()), [0, 0], 'k--', label='y=%sx + %s' %(m, b))
# so that it ranges from start to finish horizontally?
plt.plot([0, 0], (axis.get_ylim()), 'k--')  # same for vertical line but inverted
plt.legend()

# exercise
# plot a fig with grid, bold lines in the center vetrically and horizontally
# two lines, one blue, one yellow intercepting
x = [-5, 5]
m = [0.7, -1.25]
b = [-2, 0.75]
colors = ['b', 'y']
for i in range(0, len(m)):
    # for each line we need two coordinates (two points: x1, y1 and x2, y2)
    y = [m[i] * x_val + b[i] for x_val in x]
    plt.plot(x,y,  f'{colors[i]}-', label=f'y={m[i]}*x + {b[i]}')

plt.legend()
plt.axis('square')
plt.grid()

axis = plt.gca()
plt.xlim(-4, 4)
plt.ylim(-6, 6)
plt.plot(axis.get_xlim(), [0, 0], 'k--')
plt.plot([0, 0], axis.get_ylim(), 'k--')

plt.show()
# ok so apparently the intercept of the two lines is the solution to each separate line

# RATIONAL FUNCTIONS
# it's about ratios: a rational function is a ratio of polynomials
# i.e.
y = 2 - x**2
x = list(np.arange(-3, 4))
y = np.zeros(len(x))

for i in range(len(x)):
    y[i] = 2 - x[i]**2

plt.plot(x, y, 's-')
# now increasing the resolution of the points

x = np.linspace(-3, 3, 100)  # last input is the number of steps between -3 and +3
y = 2 + np.sqrt(abs(x))

plt.plot(x, y, 'ms-')

# exercise: plot a bunch of lines using the following:
power_list = np.linspace(-1, 3, 5)
x = np.linspace(-3, 3, 100)  # last input is the number of steps between -3 and +3
for power_num in power_list:
    y = x**power_num
    plt.plot(x, y, linewidth=4)

plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))

plt.grid()


# PLOTTING USING SYMPY
from sympy.abc import x
import sympy.plotting.plot as symplot
y = x**2
p = symplot(x, y, show=False)  # false so that it does not show the plot unless I say so
p.xlim = [0, 50]
p[0].line_color = 'm'  # be mindful whether you are changing a feature of the entire plot,
# or of a PIECE of the plot p vs p[i]
p.title = 'TITLE'
p.show()

x, a = sym.symbols('x, a')
expr = a/x
n = 1
# p = symplot(expr)
# ValueError: Too many free symbols.
# Expected 1 free symbols.
# Received 2: {x, a}
p = symplot(expr.subs(a, n),(x, -5, 5), show=False)
p[0].label = '$y=%s$' %sym.latex(expr.subs(a, n))
# $ for latex text

p2 = symplot(expr.subs(a, 2), show=False)
p.extend(p2)

p.ylim = [-5, 5]
p.legend = True
p.show()


# exercise: try and plot this:
x, a = sym.symbols('x a')
y = a / (x**2 - a)
a_vals = [2, 3, 4]
# initialize the p variable:
p = None
p = symplot(y.subs(a, 1), (x, -5, 5), show=False)
# ok so the teacher first created the plot variable using the 1st params, and then
# iterated through the rest to extend the p variable and add more lines in the plot
p[0].label = '$%s$' %sym.latex(y.subs(a, 1))
for index, alpha in enumerate(a_vals):
    p.extend(symplot(y.subs(a, alpha),(x, -5, 5), show=False))
    p[index].label = '$%s$' %sym.latex(y.subs(a, alpha))
    p[index].line_color = np.random.rand(3)

p.ylim = [-10, 10]
p.xlim = [-4, 4]
p.legend = True
p.show()

# alternative way:
x, a = sym.symbols('x a')
y = a / (x**2 - a)
a_vals = [1, 2, 3, 4]
# use [x for x in y] formatting to iterate through the a_vals and substitue x, get y expression solution all in one list
exprs = [y.subs(a, alpha) for alpha in a_vals]

# plot all in one go
# apparently you can use an asterisk to specify that you want this line to use all the variables(?) in the list expr
p = symplot(*exprs, (x, -4, 4), ylim=(-10, 10), show=False)

# set labels
for expr, series in zip(exprs, p):
    series.label = f"$y={sym.latex(expr)}$"

p.legend = True
p.show()

# how to create images based on matrices!
# create a matrix
A = [[1, 2], [1, 4]]
plt.imshow(A) # image show
plt.xticks([0, 1])
plt.yticks([0.85, 1.04])
plt.show()
#################
A = np.zeros((10, 14))
print(np.shape(A))

for i in range(0, A.shape[0]):
    for j in range(0, A.shape[1]):
        A[i, j] = 3*i - 4 * j

plt.imshow(A)

plt.plot([0, 3], [8, 2], 'r', linewidth=4)

# how to plot the numbers in the matrix
for i in range(np.shape(A)[0]):
    for j in range(np.shape(A)[1]):
        plt.text(j, i, int(A[i, j]), horizontalalignment='center', verticalalignment='center')

# interesting input: horizontal and verticalalignment

plt.set_cmap('twilight_shifted')
plt.show()


# imshow matrix exercise
# create a checkers board matrix lol
# don't forget tick marks, but without labels??
C = np.zeros((20, 20))
for i in range(np.shape(C)[0]):
    for j in range(np.shape(C)[1]):
        C[i, j] = (-1)**(i+j)

plt.imshow(C)
plt.set_cmap('gray') # much simpler than I thought lol
plt.xticks(color='w')
plt.yticks(color='w')

# POLYGONS TO CREATE PATCHES IN PYTHON
# i.e. a triangle in a matrix
M = np.zeros((5, 5))
coordinates = np.array([[1, 1], [1, 4], [4, 1]])
# now I want another triangle on top of the other one:
coordinates2 = np.array([[2, 2], [2.5, 4], [3.5, 1]])

from matplotlib.patches import Polygon
p = Polygon(coordinates, facecolor='purple', alpha=0.5) # alpha to adjust transparency of triangle
p2 = Polygon(coordinates2, facecolor='black', alpha=0.5, edgecolor='y')
# edgecolor adds a 'frame to the triangle

fig, ax = plt.subplots()  # the figure and the axes within the figure
ax.add_patch(p)
ax.add_patch(p2)

ax.set_ylim([0, 5]) # adjust the lims so that the triangle is visible!
ax.set_xlim([0, 5])
plt.show()

# POLYGON exercise:
y = -x**2
x_range = np.linspace(-2, 2, 101)

y_vals = [y.subs(x, i) for i in x_range]

fig, ax = plt.subplots()  # the figure and the axes within the figure
p_coordinates = np.vstack((x_range, y_vals)).T
p_polygon = Polygon(p_coordinates, facecolor='g', alpha=0.5) # alpha to adjust transparency of triangle

rectangle = np.array([[-0.5, -4], [-0.5, -2.5], [0.5, -2.5], [0.5, -4]])
p_rectangle = Polygon(rectangle, facecolor='k')

ax.add_patch(p_polygon)
ax.add_patch(p_rectangle)

plt.plot(x_range, y_vals, 'k')
plt.plot([-2, 2], [-4, -4], 'k')
plt.axis('off')
plt.show()  # well this was a pain in the ass

from pathlib import Path
default_path = Path.cwd()
images_path = default_path / 'images'
import os
images_path.mkdir(exist_ok=True)
plt.savefig(images_path/'annoying_polygon_patches.png', dpi=300)

# Section 6 BUG HUNT:
# 1
plt.plot(3, 2, 'ro')
plt.axis('square')
plt.axis([-6, 6, -6, 6]) # issue was the brackets were missing
plt.show()

# 2
plt.plot([0, 3], [0,5]) # the commas were missing
plt.show()

# 3
import numpy as np
x = range(-3, 4)
y = np.zeros(len(x))

for i in range(0, len(x)):
    y[i] = 2 - x[i]**2 # it should be x[i], not the x range

plt.plot(x, y, 's-')
plt.show()

# 4 plot two lines
plt.plot([-2, 3], [4, 0], 'b')
plt.plot([0, 3], [-3, 3], 'r')

plt.legend() # this is unnecessary
plt.show()

# 5:
randmat = np.random.randn(5, 9) # random matrix
axis = plt.gca()
x_lim = axis.get_xlim()
y_lim = axis.get_ylim()
# draw a line from lower-left corner to upper left
plt.plot([8,0], [0, 4], color=(0.4, 0.1, 0.9), linewidth=5) # linewidth, not line_width
plt.imshow(randmat)
plt.set_cmap('Purples')
plt.show()

# 6: plot two lines with labels
plt.plot([-2, 3], [4, 0], 'b', label='line1')
plt.plot([-0, 3], [-3, 3], 'r', label='line2')

plt.legend() # plt legend needed to be included
plt.show()

# 7:
x_range = np.linspace(1, 4, 20)
x = sym.symbols('x')
y = x**2 / (x-2)
y_vals = [y.subs(x, i) for i in x_range]

plt.plot(x_range, y_vals)
plt.xlim([x_range[0], x_range[-1]]) # adjust x-axis limits according to the first and last points of x
plt.show()

# 8:
x = sym.symbols('x')
y = x**2 - 3*x

xrange = range(-10, 10)
for i in range(len(xrange)):
    plt.plot(xrange[i], y.subs(x, xrange[i]), 'o')
    # not y(xrange[i]), 'o' -> stupid shittt

plt.xlabel('x')
plt.ylabel('$f(x) = %s' %sym.latex(y))
plt.show()


# 9:
x = [-5, 5]
m = 2
b = 1

y = [m*i + b for i in x]
# it was just y = m*x + b and that does not work with a list of x values

plt.plot(y, x)
plt.show()

# 10:
x = range(-20, 21)
for i in range(0, len(x)):
    print(i)
    plt.plot([0, x[i]], [0, abs(x[i])**(1/2)], color=(i/len(x), i/len(x), i/len(x)))
    # color, not line_colors?
plt.axis('off') # not 'of' I guess?
plt.show()

# 11:
m = 8
n = 4

# initialize matrix
C = np.zeros((m, 4))

# populate the matrix
for i in range(0, m):
    for j in range(0, n):
        C[i, j] = (-1)**(i+j)
        print(i, j)

# display some numbers
for i in range(0, m):
    for j in range(0, n):
        plt.text(j, i, i+j,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontdict=dict(color='m'))
        # switched x and y (from i, j to j, i)
'''
In a matrix like C[i, j],

i = row (goes down = vertical = y)

j = column (goes across = horizontal = x)

In Matplotlib plotting (plt.text(x, y, ...)),

the first number is the x-position

the second number is the y-position

So the meaning of the two coordinates is swapped:
matrix → (i, j) = (y, x)
plot → (x, y)
'''
plt.imshow(C)
plt.set_cmap('gray')
plt.show()
