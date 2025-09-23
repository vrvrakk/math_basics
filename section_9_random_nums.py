import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sympy as sym

# let's get some random ass nums
nums = np.random.rand(100)  # generate 100 random nums
# center distribution around 3:
# nums = nums + 3
# set min and max val of distribution:
# UNIFORM DISTRIBUTION
minval = 2
maxval = 17

nums = nums * (maxval-minval) + minval # stretching it out so that it varies from 0 to 15

plt.plot(nums, 's')
plt.show()

plt.hist(nums)
plt.show()

# NORMAL DISTRIBUTION
nums = np.random.randn(1000) # n states the numbers dist should be NORMAL
plt.plot(nums, 's', alpha=.5)
plt.show()

plt.hist(nums)
plt.show()

# EXERCISE
# generate a normal dist with mean = 15
# std = 4.3
# X=μ+σ⋅Z
normal_dist = np.random.randn(1000)
mean = 15
std = 4.3

adjusted_dist = mean + std * normal_dist
# Multiplying by the std stretches or squeezes the spread of the values,
# and adding the mean shifts the whole distribution so its center is at that mean.
plt.hist(adjusted_dist)
plt.show()

# exercise 2: plot unit vectors with random phase angles
# what random distributions are most sensible?
# simplified: “Generate arrows of length 1 that point in random directions, and plot them.”

# convert to angles:
n = 1000
sin_angles = np.random.uniform(0, 2*np.pi, n) # from 0 to 360 degrees
cos_angles = sin_angles.copy()

sin_conv = np.sin(sin_angles)
cos_conv = np.cos(cos_angles)

for i in range(n):
    plt.plot([0, cos_conv[i]], [0, sin_conv[i]])


plt.axis('square')
plt.axis('off')
plt.show()

# converting between radians and degrees
# radians the pi stuff
# degrees theta
degree = 1804543
# formula:
rad = degree * np.pi / 180
rad = rad%(2*np.pi)

# formula = degree * np.pi/180
print('%g is %g radians' %(degree, rad))

# the other way:
rad = 4*np.pi

degrees = (rad * 180) / np.pi
degrees = degrees%360 # angle of 0

# ORR use:
rad = np.deg2rad(degree)
deg = np.rad2deg(rad)

# create a function:
# 2 inputs, angle to convert and unit
# generate a plot -> 2 vectors -> original vector pi/4 and another along the x axis
# the cosine axis -> the reference, 45 degrees relative to the other line
# title: Angle of 45, or x rad.
# print value error: Unknown unit!

def angle_convert_plot(unit=''):
    '''Take an angle (like 45 or π/4).
    Take a unit string, e.g. "deg" or "rad".
    Convert the input to radians internally (since np.sin / np.cos use radians).
    Plot two vectors:
    One along the x-axis (the reference).
    One at the given angle.
    Title the plot with the angle in both degrees and radians.
    If someone gives an unsupported unit (like "grads" or "bananas"),
    raise ValueError("Unknown unit!").'''
    if unit == 'deg':
        radians_str = input("Please specify radians: ")
        radians = float(eval(radians_str))  # e.g. user enters "2*np.pi"
        degrees = np.rad2deg(radians)
        print(radians)
    elif unit == 'rad':
        degrees_str = input('Please specify degrees: ')
        degrees = float(eval(degrees_str))
        radians = np.deg2rad(degrees)
        print(radians)
    else:
        raise ValueError('Unknown unit!')

    # plot reference
    x_val = np.cos(radians)
    y_val = np.sin(radians)
    plt.plot([0, 1], [0, 0], 'r-', linewidth=4)
    # plot angle
    plt.plot([0, x_val], [0, y_val], 'r-', linewidth=4)

    plt.axis([-1, 1, -1, 1])
    plt.grid()
    plt.title(f'Angle of {degrees}, or {radians} rad.')
    plt.show()

angle_convert_plot(unit='deg')
angle_convert_plot(unit='rad')