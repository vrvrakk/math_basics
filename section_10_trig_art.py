import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Asteroid radial curve
# combine sines and cosines to powers
# powers as params in a family of equations
# curvy diamonds in python
# converting linear indices to subscripts

# Asteroid Radial Curve:
# x = a * cos(t)**3
# y = a * sin(t)**3
# in this case:
# a = t

t = np.linspace(0, 60, 1000)
a = t

x = a * np.cos(t)**3
y = a * np.sin(t)**3

plt.plot(x, y, '-', color='fuchsia', linewidth=3)
plt.axis('off')
plt.show()

# exercise: plot the asteroid radial curves but with a = t**n
# for n from 0 to 8
n = np.arange(0, 9, 2)
fig, axes = plt.subplots(2, 3)
axes = axes.flatten()

t = np.linspace(0, 60, 1000)
for i, n_val in enumerate(n):
    a = t**n_val
    x = a * np.cos(t) ** 3
    y = a * np.sin(t) ** 3
    axes[i].plot(x, y, '-', color='fuchsia', linewidth=3, label='$a^%g$'%(n_val))
    axes[i].legend(loc='upper right')  # legends placed in upper right loc for each subplot
    axes[i].axis('off')
    axes[i].axis('square')


fig.suptitle('Asteroid Radial Curves')
fig.delaxes(axes[-1])  # delete final plot
plt.show()


# Rose Curves:
# k is a single param
# k does not have to be an integer: if k is even -> 2k pedals
# if k is odd -> k pedals
k = 8.6
t = np.linspace(0, 6*np.pi, 1000)
x = np.cos(k * t) * np.cos(t)
y = np.cos(k * t) * np.sin(t)

plt.plot(x, y, linewidth=3, color='fuchsia')
plt.axis('off')
plt.show()


# rose curves exercise:
# 9 roses
# with diff k vals 0 to 4
k = np.arange(0, 4.5, 0.5)
t = np.linspace(0, 6*np.pi, 200)

fig, axis = plt.subplots(3, 3)
axis = axis.flatten()
for i, k_val in enumerate(k):
    x = np.cos(k_val * t) * np.cos(t)
    y = np.cos(k_val * t) * np.sin(t)
    axis[i].plot(x, y, linewidth=3, color=(0.7, 0, 0.3))
    axis[i].set_title('k = %g'%(k_val))
    axis[i].axis('off')
    axis[i].axis('square')

fig.suptitle('Rose Curves')
plt.show()

# Squircle
# nth sqrt(num) = num^1/n
# x**4  = a**4 - y**4
a = 1
x = np.linspace(-10, 10, 1001)
y = (a**4 - x**4)**(1/4)
# If you only plot y, you’d only see the top half.
# If you only plot -y, you’d only see the bottom half.
# Together, they complete the full figure.
plt.plot(x, y)
plt.plot(x, -y)
plt.axis('off')
plt.show()


# Logarithmic Spiral
# basically a circle, with a radius changing in log steps
n = 1000
x = np.linspace(0, 6*np.pi, n)
acc = np.logspace(np.log10(1), np.log10(40), n)
sine_vals = np.sin(x) * acc
cos_vals = np.cos(x) * acc

plt.plot(cos_vals, sine_vals)
plt.show()  # my way s:

# teacher's way:
t = np.linspace(0, 10*np.pi, 1234)
k = -3
x = np.cos(t) * np.exp(t/k)
y = np.sin(t) * np.exp(t/k)
# Logarithmic spiral formula:
# x = cos(t) * exp(t/k)
# y = sin(t) * exp(t/k)

# t = angle around the circle (in radians)
#     - as t increases, you "walk" around the circle
#     - one full rotation = 0 → 2π

# k = spiral tightness factor
#     - controls how quickly the spiral opens up
#     - large |k| → radius grows slowly → tight spiral
#     - small |k| → radius grows fast → wide spiral
#     - negative k → spiral winds inward instead of outward
plt.plot(x, y)
plt.show()

# exercise: plot a bunch of spirals with diff k values, with changing color from black to pink I guess
k_range = np.linspace(-1, -5, 1234)
t_range = t

for i, k_val in enumerate(k_range):
    ratio = i / (len(k_range) - 1)  # goes smoothly from 0 → 1
    x = np.cos(t) * np.exp(t / k_val)
    y = np.sin(t) * np.exp(t / k_val)
    plt.plot(x, y, color=(ratio, 0, ratio), linewidth=2)

plt.axis('off')
plt.axis('square')
plt.show()

# Logistic map:
# formula:
#x n+1 = r*xn(1-xn)


# n E N: n <= 500
# r E [1, 4] in 1000 steps
# for each value of r, plot the unique values in the last 10% of the vector x.
# r goes on the x-axis and x goes on the y-axis
n = np.linspace(1, 500, 500)
r = np.linspace(1, 4, 1000)

# Vary r between 1 and 4.
# For each r, run the logistic map 500 steps.
# Ignore the warm-up, only keep the final values.
# Plot r vs those final values to see the long-term behavior (that famous “bifurcation diagram” picture).
x_arrays = []
for r_index, r_val in enumerate(r):
    x_list = np.zeros(500)
    for x_index, x_value in enumerate(x_list):
        if x_index == 0:
            x_list[x_index] = 0.5
        else:
            x_list[x_index] = r_val * x_list[x_index-1] * (1 - x_list[x_index - 1])

        x_arrays.append(x_list)


x_arrays = []
for r_index, r_val in enumerate(r):
    x_list = np.zeros(500)
    for x_index, x_value in enumerate(x_list):
        if x_index == 0:
            x_prev = 0.5
            x_list[x_index] = x_prev
            x_next = r_val * x_prev * (1 - x_prev)
            x_list[x_index + 1] = x_next
        elif 0 < x_index < len(x_list) - 1:
            x_prev = x_list[x_index - 1]
            x_next = r_val * x_prev * (1 - x_prev)
            x_list[x_index] = x_next
    x_arrays.append(x_list)

# plot only 10% of vector x
# r goes on the x-axis and x goes on the y-axis

for r_val, x_list in zip(r, x_arrays):
    length = len(x_list)
    len_to_keep = length * 0.1
    y = x_list[int(length-len_to_keep):-1]
    plt.scatter([r_val] * len(y), y, s=1)

plt.xlabel('r'); plt.ylabel('x (last 10%)')
plt.style('dark')
plt.show()