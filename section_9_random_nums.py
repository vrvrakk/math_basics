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

# pythagorean theorem
# how to create triangles
# application of the pythagorean theorem to complex numbers
# a**2 + b**2 = c**2 (υποτείνουσα)
a = 3
b = 4
c = 5 # when a and b are 3 and 4 respectively, c is 5 -> one of the pythagorean triplets
c = np.sqrt(a**2 + b**2)

plt.plot([0, a], [0, 0], 'k', linewidth=2)
plt.plot([0, 0], [0, b], 'k', linewidth=2)
plt.plot([a, 0], [0, b], 'k', linewidth=2)
# or:
# plt.plot([0, a], [b, 0], 'k', linewidth=3)

# little square of 90°
plt.plot([0.3, 0], [0.3, 0.3], 'k', linewidth=1)
plt.plot([0.3, 0.3], [0.3, 0], 'k', linewidth=1)
plt.plot()

# add lengths at each side, because triangle is lonely :(
# plt text with x, y coordinates and string as inputs
a_txt = a/3  # divided by 2 isn't great
b_txt = b/3
c_txt = c/3
offset = .25
# a:
plt.text(a_txt, 0 - offset, s=f'a = {a}')
# b:
plt.text(0 - offset,b_txt, s=f'b = {b}', rotation='vertical')
# c:
plt.text(c_txt, c_txt, s=f' c = {c}', rotation=-55)

plt.axis('square')
axlim = np.max((a,b)) + .5
plt.axis([-0.5, axlim, -0.5, axlim])
# plt.axis('off')
plt.show()


# exercise:
# z = 3 * 3j
# plot a 2D plane, with a line starting from origin
# extract angle of line in relation to x real axis
# solve with the pythagorean theorem
# so:
# abs(z)**2 = 2**2 + 3**2 (without the j, because 3 only represents distance along the im axis)
# get tangent k, which is imaginary/real

z = complex(2, 3)

magnitude = np.sqrt(np.real(z)**2 + np.imag(z)**2)
mag2 = np.abs(z)
angle = np.arctan2(np.imag(z), np.real(z))
angle2 = np.angle(z)

# The first list [n1, n2] = all the x-coordinates of the points.
# The second list [n3, n4] = all the y-coordinates of the points.
plt.plot([0, np.real(z)], [0, np.imag(z)], 'y')
# plot the yellow point:
plt.plot([np.real(z), np.real(z)], [np.imag(z), np.imag(z)], 'yo')

plt.plot([np.real(z), np.real(z)], [0, np.imag(z)], 'g--')


plt.show()

# plotting and computing sine, cosine and tangent
x = np.linspace(0, 6*np.pi, 400)

plt.plot(x, np.sin(x), 'r', label='$sin(\\theta)$')
plt.plot(x, np.cos(x), 'b', label='$cos(\\theta)$')

plt.legend()
plt.xlabel('rad.')
plt.ylabel('function value')
plt.show()

# plot
th = np.linspace(0, 4*np.pi, 400)
plt.plot(th, np.tan(th), 'k')
plt.show()

ang = np.random.rand()*2*np.pi

tan = np.tan(ang)

sc = np.sin(ang) / np.cos(ang)

# c ** 2 + s ** 2 = 1
thetas = np.linspace(0, 2*np.pi, 13)
np.cos(thetas)**2 + np.sin(thetas)**2

# Nobody arbitrarily "decided" sine and cosine.
# They come straight from geometry:
# On a unit circle (radius = 1), if you stand at angle θ:
#   - x-coordinate = cos(θ)  (left/right position)
#   - y-coordinate = sin(θ)  (up/down position)
#
# That's literally how sine and cosine were *defined* in the first place:
# as the coordinates of a rotating point on the circle.
#
# Two ways to visualize it:
#
# 1) Circle picture:
#    If you plot (cos θ, sin θ) with θ going from 0 → 2π, you trace a circle.
#    x-axis = cosine (left/right), y-axis = sine (up/down).
#
# 2) Wave picture:
#    If you plot (θ, sin θ), the x-axis is just the angle growing steadily
#    (a straight line to the right), while the y-axis is the up/down position.
#    As θ increases, sine values go smoothly up and down → the sine wave.
#
# 👉 Same data, just plotted differently:
#    - Circle → "where am I on the unit circle?"
#    - Wave   → "how does my up/down position change as the angle grows?"

# At θ = 0 → point = (1,0). Cos = 1 = far right. Sin = 0 = no up/down.
# At θ = π/2 → point = (0,1). Cos = 0 = centered horizontally. Sin = 1 = far up.
# At θ = π → point = (-1,0). Cos = -1 = far left. Sin = 0.
# At θ = 3π/2 → point = (0,-1). Cos = 0. Sin = -1 = far down.

# Exercise 1:
theta = np.linspace(0, 10*np.pi, 400)
y1 = np.sin(theta + np.cos(theta))
y2 = np.cos(theta + np.sin(theta))

# First, compute cos(theta).
# That’s a wave (left-right movement of the point on the circle).
# Then you add it to theta → this doesn’t mean “increasing the distance left/right.”
# It actually shifts the angle itself by a small wiggly amount.

plt.plot(y1, 'r')
plt.plot(y2, 'k')
# Comparison with the others
# sin(θ+cos(θ)): wobble is driven by left/right coordinate.
# cos(θ+sin(θ)): wobble is driven by up/down coordinate, but final wave is cosine.
# sin(θ+sin(θ)): wobble and output are both sine → kind of a “self-modulated” sine wave.
plt.show()

# 1. sin(θ)
#    - Plain sine wave.
#    - Smooth, evenly spaced peaks and valleys.

# 2. sin(θ + cos(θ))
#    - Start with sine.
#    - Timing (phase) is nudged forward/back depending on cosine.
#    - Wave speeds up/slows down when point on circle is far right/left.

# 3. cos(θ + sin(θ))
#    - Start with cosine.
#    - Timing is nudged by sine.
#    - Wave speeds up/slows down when point on circle is high/low.

# 4. sin(θ + sin(θ))
#    - A sine wave that "wobbles itself."
#    - Timing is sped up when sine is positive, slowed when sine is negative.
#    - Looks sine-like, but peaks shift slightly.

# exercise 2:
# complicated ass shite
# make circle (unit)
# draw line for theta = 7*pi/6 - black dotted
# add little black dot
# connect with origin lines
# horizontal greem vertical red, dotted lines as well
# circle from 1, 1 x axis to theta point, linewidth larger

radians = np.linspace(0, 2*np.pi, 360)
theta = 7*np.pi / 6
theta_cos = np.cos(theta)
theta_sin = np.sin(theta)

plt.plot(np.cos(radians), np.sin(radians), 'k')
plt.plot([0, theta_cos], [0, theta_sin], 'k--')
# lil dot
plt.plot([theta_cos, theta_cos], [theta_sin, theta_sin], 'ko')
# dotted axis lines
plt.plot([0, 0], [-1, 1], '--', color=(0.5, 0.5, 0.5))
plt.plot([-1, 1], [0, 0], '--', color=(0.5, 0.5, 0.5))
# plot little connecting lines and red and green ones
# green:
plt.plot([0, theta_cos], [0, 0], 'g-', linewidth=3, label='cos part')
plt.plot([theta_cos, theta_cos],[0, theta_sin], 'g--', linewidth=1)
# red
plt.plot([0, 0], [0, theta_sin], 'r-', linewidth=3, label='cos part')
plt.plot([0, theta_cos],[theta_sin, theta_sin], 'r--', linewidth=1)

# plot the arc:
arc_range = np.linspace(0, theta)
plt.plot(np.cos(arc_range), np.sin(arc_range), 'k', linewidth=3)

plt.title('$\\theta = \\frac{7\\pi}{6}$')
plt.legend()

plt.axis('square')
plt.show()

# euler's formula
# e^ik = cos(k) +isin(k)
# for a point outside the unit circle:
# me^ik = m (cos(h) + isin(k))
# m is the magnitude, basically the distance away from the origin
# e^i*pi + 1 = 0 -> considered the most elegant equation in all of human civilization

k = np.pi/4
m = 2.3

euler = m* np.exp(1j*k)
# cosine imaginary sine
cis = m * (np.cos(k) + 1j * np.sin(k))

print(cis)
print(euler)

# extract magnitude and anlge from this formula
magnitude = np.abs(euler)
ang = np.angle(euler)

plt.polar([0, ang], [0, magnitude], 'r')
plt.polar(k, m, 'bo')

plt.show()

# euler exercise:
# write a function with euler
# two inputs, one cos num, one sine num
# generate a polar graph
# vector corresponding with cos part 3 and sine part 4
# title me^iphi, m = 5, phi = 0.93

# You get two numbers:
# one is the x-part (cosine / real part)
# one is the y-part (sine / imaginary part)
# Together they make a vector (like an arrow) from the origin to that point.
# Example: (3, 4).
# Then:
# Find how long the arrow is (magnitude).
# Find the angle the arrow makes with the x-axis.
# Plot that arrow on a polar graph (like a circle with angles).

def eulerFromCosSin():
    re = float(eval(input('cosine part: ')))
    im = float(eval(input('sine part: ')))
    # z = m*e^i⋅angle
    z = complex(re, im)
    m = np.abs(z)
    m = np.sqrt(re**2 + im**2)
    angle = np.arctan2(im, re)
    angle = np.angle(z)

    plt.polar([0, angle], [0, m], 'r')
    plt.polar(angle, m, 'bo')
    plt.title('me$^{i\\phi}$, m = %g, $\\phi$ = %g' %(m, angle))
    plt.thetagrids([0, 45, 130, 200, 222.2])

    plt.show()

eulerFromCosSin()

# # We have cos_part = 3 and sine_part = 4
# # → Think of them as coordinates (x=3, y=4)
# # → Draw an arrow from (0,0) to (3,4)
#
# # Magnitude (length of arrow):
# # By Pythagoras: sqrt(3^2 + 4^2) = 5
# # In Python: np.abs(z)
#
# # Angle (direction of arrow):
# # Angle = arctan(4/3) ≈ 0.93 rad (≈ 53 degrees)
# # In Python: np.angle(z)
#
# # Euler’s formula says:
# #   e^(i·phi) = cos(phi) + i·sin(phi)
# # → This gives the point on the unit circle at angle phi
#
# # To get our arrow, we just scale it by the magnitude:
# #   z = 5 · e^(i·0.93)
#
# # Metaphor:
# # - Cartesian (3,4): "Go 3 steps east, then 4 steps north"
# # - Polar/Euler (5·e^(i·0.93)): "Go 5 steps in direction 53° northeast"
# # Both ways describe the SAME destination (the vector z).

# exercise: random exploding euler
n = 100
cos = np.cos(np.random.randn(n))
sin = np.sin(np.random.randn(n))
z = [complex(cosine, sine) for cosine, sine in zip(cos, sin)]
m = [np.abs(z_val) for z_val in z]
angle = [np.angle(phi) for phi in z]

# random m's and k's
# 40% pink, 40% purple, 20% green
# make color list: 40% purple, 40% pink, 20% green
colors = ['purple'] * int(0.4 * n) + ['red'] * int(0.4 * n) + ['yellow'] * int(0.2 * n)
import random
random.shuffle(colors)  # shuffle so they’re spread randomly

# plot with assigned colors
for i in range(0, n):
    plt.polar([0, angle[i]], [0, m[i]], color=colors[i])

plt.axis('off')
plt.show()

# mike version:
nvects = 200
ms = np.random.rand(nvects)
ks = np.random.rand(nvects) * 2 * np.pi # convert to radians

for i in range(0, nvects):
    r = np.random.rand() # random num
    if r < .4:
        clr = [1, .2, .7]
    elif r > .4 and r < .8:
        clor = [.7, .2, 1]
    plt.polar([0, ks[i]], [0, ms[i]], color=clr)

plt.axis('off')
plt.show()

# or better:
n = 100
angles = np.random.uniform(0, 2*np.pi, n)   # uniform around full circle
m = np.random.rand(n) * 1.0                 # random lengths (0–1 for example)

# color list
colors = ['purple'] * int(0.4 * n) + ['red'] * int(0.4 * n) + ['yellow'] * int(0.2 * n)
import random
random.shuffle(colors)

# plot
for i in range(n):
    plt.polar([0, angles[i]], [0, m[i]], color=colors[i])

plt.axis('off')
plt.show()

# exercise: random rnakes from trigonometry...
#r1, r2 = random(0,1)
n = 200
theta = np.linspace(0, 4*np.pi, n)
r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
x = np.cos(r1*theta)
y = np.sin(r2*theta)


plt.plot(x, y, 'k')

plt.axis('square')
plt.axis([-1, 1, -1, 1])
plt.axis('off')
# plt.title('r1 = %s, r2 = %s' %(r1, r2))
# or
plt.title(f'r1 = {np.round(r1, 2)}, r2 = {np.round(r2, 2)}')
plt.show()

# trigonometry bug hunt!
# 1: plot a series of random numbers:
s = np.random.randn(100, 1)
plt.plot(s, 'o')
plt.show()

# 2: create an image of a matrix of random integers between and including 3 and 29
M = np.zeros((10, 10))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        replacement = np.random.randint(3, 30, 1)
        M[i, j] = replacement

# or simply do:
M = np.random.randint(3, 30, (10, 10)) # cool!

# 3: create 100 random phase angles [0, 2pi] and show unit vectors with those angles
# 👉 Make 100 random angles between 0 and all the way around circle (0 → 2π).
# 👉 For each angle, draw arrow of length 1 pointing that way.
n = 100
randphases = np.random.rand(n) * 2 * np.pi
for i in range(0, n):
    plt.polar([0, randphases[i]], [0, 1], '-', color=np.random.rand(3), markersize=20, alpha=0.8)

plt.show()

# 4: create an outwards spiral using phase angles and amplitudes
n = 100
amplitude = np.linspace(0, 1, n)
radians = np.linspace(0, 4*np.pi, n)
# it's not plot cos x sine, that's a simple cirlce; also it's not amplitude x theta (cos or sine) -> wave
# how to implement amplitude with the theta?
# actual answer:
# Multiply by amplitude
# Now imagine the circle, but instead of always having radius = 1, you let radius = amplitude.
# So the point doesn’t just spin around — it also slowly moves away from the origin.
cosine_angles = np.cos(radians) * amplitude
sine_angles = np.sin(radians) * amplitude
plt.plot(cosine_angles, sine_angles)
plt.show()

# 5: convert radians to degrees:
n = 10
rad = (np.logspace(np.log10(1), np.log10(360), n)) * 2*np.pi / 360
print(np.rad2deg(rad))

# 6: famous equality in trigonometry:
ang = np.logspace(np.log10(1), np.log10(2*np.pi), 10) # log cannot be 0
np.cos(ang)**2 + np.sin(ang)**2

# create euler's number
phi = np.pi / 4  # angle 45°
m = .5

euler = m * np.exp(1j*phi)

# extract magnitude and phase:
m = np.abs(euler)
angle = np.angle(euler)

# plot
plt.polar([0, angle], [0, m], 'b', linewidth=3)
plt.show()