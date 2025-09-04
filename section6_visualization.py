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


