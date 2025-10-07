import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sympy as sym

# math proofs and intuition
# proof: a rigorous demonstration that a hypothesis is true, built on axioms and previously proven claims
# intuition: implementing examples, trusting computers implementing math accurately
# good idea to test several examples, to avoid over-generalizing from special cases


# LIMITS of a function:
# lim f(x)
# x -> a

# example:
# lim x^2/2 = 4^2/2 = 8
# x -> 4

# f(x) = x^2 / x - 2
# becomes:
# lim f(x) = +inf
# x -> 2+ # from right side
# and
# lim f(x) = -inf
# x -> 2- # from left side

# how to do this stuff in Python:
x = sym.symbols('x')

fx = x**3 # function

lim_pnt = 1.5  # limit point (var a, aka x0)

lim = sym.limit(fx, x, lim_pnt)
print('$\\lim_{x\\to %g} %s = %g$' %(lim_pnt, sym.latex(fx), lim))

# function to evaluate fx based on whatever vals I want

fxx = sym.lambdify(x, fx)  # LAMBDIFY
xx = np.linspace(-5, 5, 200)

plt.plot(xx, fxx(xx), 'k--', linewidth='2')
plt.plot(lim_pnt, lim, 'ro')
plt.show()

# next example:
fx = (x**2)/(x-2)
fxx = sym.lambdify(x, fx)

xx = np.linspace(1, 3, 102)
# we cannot use just x = 2 -> denominator will be 0
lim_pnt = 2
lim = sym.limit(fx, x, lim_pnt, dir='+')  # we coming from the right side

plt.plot(xx, fxx(xx))
plt.title('$\\lim_{x\\to %g} %s = %g^-$' %(lim_pnt, sym.latex(fx), lim))
plt.show()

# Exercise:
# part 1
fx = sym.exp(-x) * sym.sqrt(x + 1)
gx = sym.cos(x + sym.sin(x))

xx = np.linspace(1, 100, 200)

fxx = sym.lambdify(x, fx)
gxx = sym.lambdify(x, gx)

x0 = 5

plt.plot(fxx(xx), 'k--')
plt.plot(gxx(xx), 'r--')
plt.show()

# now  prove that lim fx / lim gx with x -> 5
# is the same as lim fx/gx with x -> 5
flim = sym.limit(fx, x, x0, dir='+')  # we coming from the right side
glim = sym.limit(gx, x, x0, dir='+')

rhs = flim / glim

lhs = sym.limit(fx/gx, x, x0, dir='+')

assert rhs == lhs

# === General steps for solving limits ===

# 1. Direct substitution
#    - Try plugging the value directly into the function.
#    - If you get a normal number → that’s the limit.
#    - If you get 0/0 or ∞/∞ → go to next step.

# 2. Simplify the expression
#    - Factor things, cancel common terms.
#    - Example: (x^2 - 4) / (x - 2) = (x-2)(x+2)/(x-2) → x+2.

# 3. For infinity limits (x → ∞ or -∞)
#    - Look at the highest power of x in numerator and denominator.
#    - Divide top and bottom by that highest power.
#    - Small terms (like 5/x^2) vanish → only dominant terms remain.

# 4. Check one-sided limits
#    - If denominator → 0, test from left and right.
#    - Tiny positive denominator → +∞.
#    - Tiny negative denominator → -∞.

# 5. Trig limits
#    - Remember the "famous ones":
#      lim (x→0) sin(x)/x = 1
#      lim (x→0) (1 - cos(x))/x = 0

# 6. If stuck, rewrite cleverly
#    - Substitution (e.g. x = 2 + h).
#    - Divide top and bottom by a useful factor.
#    - Multiply by conjugates for square roots (√(x+1) - √x).

# 🔑 LIMIT RULES CHEAT SHEET

# 1. Direct substitution
# If f(x) is continuous at a:
#   lim x→a f(x) = f(a)

# 2. Sum / Difference Rule
#   lim (f(x) + g(x)) = lim f(x) + lim g(x)

# 3. Product Rule
#   lim (f(x) * g(x)) = (lim f(x)) * (lim g(x))

# 4. Quotient Rule
#   lim (f(x) / g(x)) = (lim f(x)) / (lim g(x))   (if denominator ≠ 0)

# 5. Constant Multiple
#   lim (c * f(x)) = c * lim f(x)

# 6. Power Rule
#   lim (f(x))^n = (lim f(x))^n   (for integer n)

# 7. Root Rule
#   lim √[n]{f(x)} = √[n]{lim f(x)}   (if inside stays valid)

# 8. Special trig limits
#   lim x→0 (sin x / x) = 1
#   lim x→0 ((1 - cos x) / x) = 0
#   lim x→0 ((1 - cos x) / x^2) = 1/2

# 9. Infinity behavior (rational functions)
#   - if degree top < degree bottom → limit = 0
#   - if degree top = degree bottom → limit = ratio of leading coefficients
#   - if degree top > degree bottom → limit = ±∞ (depends on sign)


# piecewise functions
# f(x) = 0, -2x
# if x <= 0
#     x > 0

piece1 = 0
piece2 = -2*x
piece3 = x**3/10

fx = sym.Piecewise((piece1, x < 0), (piece2, (x >= 0) & (x < 10)), (piece3, x >= 10))
fxx = sym.lambdify(x, fx)
xx = np.linspace(-3, 15, 1234)

plt.plot(xx, fxx(xx))
plt.show()

# piecewise exercise:
# implement function, print out in latex, and make graph in xkcd style!
# comic sans style!

piece1 = x**3  # x <= 0
piece2 = sym.log(x, 2)  # otherwise

fx = sym.Piecewise((piece1, x <= 0), (piece2, x > 0))
fxx = sym.lambdify(x, fx)
xx = np.linspace(-3, 15, 1234)

with plt.xkcd():
    plt.plot(xx, fxx(xx), 'r-')

plt.show()

# derivatives:
# how the function changes over time or x
# derivative of a polynomial
# d/dx(x**2) = 2x**1
# d/dx(x**3) = 2*x**2
# d/dx(3x**3) = 9 * x**2

'THE RULE'
# d/dx (a*x^n) = n * a * x ^n-1

fx = x**2
dfx = sym.diff(fx)  # the derivative

# indicating derivatives using latex and python:
# leibniz notation: lol
print('$f(x) = %s, \\quad \\frac{df}{dx}=%s$' %(sym.latex(fx), sym.latex(dfx)))

# lagrange notation:
print('$f(x) = %s, \\quad f\'=%s$' %(sym.latex(fx), sym.latex(dfx))) # f\' good trick if I need ' in the text

# newton notation:
print('$f(x) = %s, \\quad ´\\dot{f}=%s$' %(sym.latex(fx), sym.latex(dfx))) # f\' good trick if I need ' in the text

# plotting:
import sympy.plotting.plot as symplot

fx = 3 - x**3
dfx = sym.diff(fx)

p = symplot(fx, (x, -5, 5), show=False)
p.extend(symplot((dfx), (x, -5, 5), show=False))
p[1].line_color = 'orange'
p[0].label = '$f(x) = %s$' %sym.latex(fx)
p[0].label = '$f(x) = %s$' %sym.latex(dfx)

p.legend = True
p.ylim = [-10, 10]
p.show()

# Exercise:
fx = 3 + 2*x - 5*x**2 + 7*x**4
gx = 4*x**2 + x**5

# implement these functions
# demonstrate product rule and summarion rule of derivatives


product1 = sym.diff(fx + gx)
product2 = sym.diff(fx) + sym.diff(gx)
assert product1 == product2

product3 = sym.diff(fx * gx)
product4 = sym.diff(fx) * sym.diff(gx)
assert product3 != product4

product5 = sym.diff(fx * gx)
product6 = (sym.diff(fx) * gx) + (fx * sym.diff(gx))
assert product5 == product6


# circular derivative:
# cos(x) = -sin(x)
# -sin(x) = -cos(x)
# -cos(x) = sin(x)
# sin(x) = cos(x)

q = sym.symbols('q')
print(sym.diff(sym.cos(q)))
print(sym.diff(sym.sin(q)))

f = sym.cos(q)
dfx = sym.diff(f)
for i in range(0, 8):
    print('$\\frac{d}{dx}%s $= ' %(sym.latex(f), sym.latex(dfx)))

# next:
    for i in range(0, 4):
        if i == 0:
            p = symplot(f, show=False, label=sym.latex(f))
        else:
            p.extend(symplot(f, show=False, label=sym.latex(f)))

        f = sym.diff(f)

p.legend = True
p.xlim = [-3, 3]
p.show()

# exercise: plot function results and its derivatives separately
x, a = sym.symbols('a x')

fax = sym.cos(x + sym.sin(x)) + a
for i in range(0, 5):
    fax_res = fax.subs(a, i)
    dfax = sym.diff(fax_res)
    if i == 0:
        p = symplot(fax_res, show=False, label='a = %g'%(i))
        q = symplot(dfax, show=False, label='Derivative of a = %g'%(i))
    else:
        p.extend(symplot(fax_res, show=False, label='a = %g'%(i)))
        q.extend(symplot(dfax, show=False, label='Derivative of a = %g' % (i)))



p.legend = True
q.legend = True
p.xlim = [-10, 10]
q.xlim = [-10, 10]
q.show()
p.show()

# drawing tangent lines (efaptomeni)
# barely touches the function line
# tangent line is equal to derivative of the function
# f'a(x - xa) + fa
# y = m * x + b # slope intercept form of a line (m)
# the derivative is the slope of a function

x = sym.symbols('x')

f = x**2
df = sym.diff(f)

xa = 1  # value at which to compute the tangent line

# get the function and derivative at value xa
fa = f.subs(x, xa)
dfa = df.subs(x, xa)

xx = np.linspace(-2, 2, 200)
f_fun = sym.lambdify(x, f)(xx)
df_fun = sym.lambdify(x, df)(xx)

# compute the tangent line
tanline = dfa * (xx-xa) + fa

plt.plot(xx, f_fun, label='f(x)')
plt.plot(xx, tanline, label='tangent')
plt.plot(xa, fa, 'ro')

plt.axis('square')
plt.axis([-2,2, -.5, 2])

ax = plt.gca()
plt.plot(ax.get_xlim(), [0, 0], 'k--')
plt.plot([0, 0], ax.get_ylim(), 'k--')
plt.legend()

plt.show()

# exercise: create a function that returns the tangent line, given
# a function xa, and domain bounds
# generate plot using
# "Write a function that takes any f(x), picks a point a∈(−2,2), finds
# f′(a) builds the tangent line equation, and plots both the original curve and the tangent line."
a = np.random.randint(-2, 2)
x_list = np.linspace(-5, 5, 1000)
x = sym.symbols('x')
fx = x**2
dfx = sym.diff(fx)

fx_list = sym.lambdify(x, fx)(x_list)
dfx_list = sym.lambdify(x, dfx)(x_list)
# if a = x
tangents = []
for i, a in enumerate(x_list):
    tangent = fx_list[i] + dfx_list[i] * (x_list - a)
    tangents.append(tangent)

plt.plot(x_list, fx_list)
plt.plot(x_list, tangents)
plt.axis('off')
plt.show()

'''
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

x = sym.symbols('x')
f = x**2
df = sym.diff(f)

# Pick a random point where we want the tangent
a = np.random.randint(-2, 2)

# Convert symbolic f and df to real functions
f_num = sym.lambdify(x, f)
df_num = sym.lambdify(x, df)

# Evaluate f(a) and f'(a)
fa = f_num(a)
dfa = df_num(a)

# Plot range
x_vals = np.linspace(-5, 5, 200)

# Compute original function and tangent line
f_res_list = f_num(x_vals)
tangent_list = fa + dfa * (x_vals - a)

plt.plot(x_vals, f_res_list, label='f(x)')
plt.plot(x_vals, tangent_list, label='Tangent at x={}'.format(a), linestyle='--')
plt.scatter(a, fa, color='red')  # Mark the point of tangency
plt.legend()
plt.show()

'''

# teacher's way:
def compute_tangent(f, a, x_list):
    df = sym.diff(f)
    fa = f.subs(x, a)
    dfa = df.subs(x, a)

    # tangent:
    return dfa * (x_list - a) + fa


x = sym.symbols('x')
f = x ** 2
x_list = np.linspace(-2, 2, 200)
f_function = sym.lambdify(x, f)(x_list)  # solve function by x with x= values from x_list

a_val = np.random.randint(-2, 2)
tanline = compute_tangent(f, a_val, x_list)

for x_val in x_list:
    # x_val = a_val in this case..I guess?
    tan = compute_tangent(f, x_val, x_list)
    plt.plot(x_list, tan)

plt.plot(x_list, f_function)
plt.axis('square')
plt.axis([-2, 2, -1, 3])
plt.axis('off')
plt.show()


# finding critical points in python (in plots)
# how: got fx -> get dfx -> solve by 0 (like a polynomial
# get polynomial roots
from scipy.signal import find_peaks

# empirical method
x = np.linspace(-5, 5, 1001)
fx = x**2 * np.exp(-x**2)
dfx = np.diff(fx) / np.diff(x)
# or x[1] - x[0]

local_max = find_peaks(fx)
local_min = find_peaks(-fx)

print(f'The critical points are: {x[local_max[0]]}, {x[local_min[0]]}')

plt.plot(x, fx)
plt.plot(x[0:-1], dfx) # derivative is one point fewer
plt.plot(x[local_max[0]], fx[local_max[0]], 'ro')
plt.plot(x[local_min[0]], fx[local_min[0]], 'go')
plt.show()


# now the analytic method:
x = sym.symbols('x')
x_list = np.linspace(-5, 5, 1001)
fx = x**2 * sym.exp(-x**2)
dfx = sym.diff(fx)
fx_list = sym.lambdify(x, fx)(x_list)
dfx_list = sym.lambdify(x, dfx)(x_list)

local_max = find_peaks(fx_list)
local_min = find_peaks(-fx_list)


print(f'The critical points are: {x_list[local_max[0]]}, {x_list[local_min[0]]}')

plt.plot(x_list, fx_list)
plt.plot(x_list, dfx_list) # derivative is one point fewer
plt.plot(x_list[local_max[0]], fx_list[local_max[0]], 'ro')
plt.plot(x_list[local_min[0]], fx_list[local_min[0]], 'go')
plt.show()

# exercise: determine which values of a give the function a critical value at x=1
# or x=2
x, a = sym.symbols('x a')
fxa = x**2 * sym.exp(-a*x**2)
a_list = np.arange(0, 2.25, 0.25)
x_range = np.linspace(-3, 3, 101)

fig, ax = plt.subplots(1, 2)
for a_idx in a_list:
    fx = fxa.subs(a, a_idx)
    dfx = sym.diff(fx)
    crit_points = sym.solve(dfx)
    crit_points = [int(num) for num in crit_points]
    # plot
    ax[0].plot(x_range, sym.lambdify(x, fx)(x_range))
    ax[1].plot(x_range, sym.lambdify(x, dfx)(x_range))
    for a_idx in a_list:
        if 1 in crit_points:
            print(fr'$x^2 e^{{-{a_idx} x^2}}$ has a critical point at $x = 1$. Woohoo!')
        elif 2 in crit_points:
            print(fr'$x^2 e^{{-{a_idx} x^2}}$ has a critical point at $x = 2$. Woohoo!')
        else:
            print(fr'$x^2 e^{{-{a_idx} x^2}}$ has NO critical points. :(')

ax[0].set_title('Function')
ax[1].set_title('Derivative')
plt.show()

# partial derivatives
# when we have more than one input in a function
# more than one var
# i.e. f(x, y)
# partial der (x) = fx = 2*y**2
# fy = 4*x*y

from sympy.abc import x, y

f = x**2 + x*y**2

print('\\frac{\\partial f}{\\partial x} = %s'%sym.latex(sym.diff(f, x)))
print('\\frac{\\partial f}{\\partial y} = %s'%sym.latex(sym.diff(f, y)))

# exercise with partial derivatives:
# f(x, y) = x**2 + x*y**2
x, y = sym.symbols('x y')
fxy = x**2 + x*(y**2)
partial_x = sym.diff(fxy, x)
partial_y = sym.diff(fxy, y)

p = sym.plotting.plot3d(fxy, (x, -3, 3), (y, -3, 3), title='$(x, y)$ = %s' %(fxy))
p = sym.plotting.plot3d(partial_x, (x, -3, 3), (y, -3, 3), title='$(x, y) $= %s' %(fxy))
p = sym.plotting.plot3d(partial_y, (x, -3, 3), (y, -3, 3), title='$(x, y) $= %s' %(partial_y))

# indefinite and definite integrals
# no idea what integrations are
# adding a lot together of really tiny things!
# what is the area between two lines of a non-linear curve
# the smallest possible width that a bar can take, that is used to cover the area of interest under the curve
# sum up very very tiny bars (values) to get the A
# formula: S between a and b (x and y) of the f(x)*dx
# f(x) height of the bar at each point of x
# dx smallest, thinnest width of quantization of x
# definite integra: between definite points a and b

# indefinite integral of polynomials:
# S bx^a * dx = bx^(a+1) / a+1 + c
# integration and derivation are the opposites of each other

x = sym.symbols('x')


f = x

sym.integrate(f)

sym.integrate(f, (x, 0, 1))  # indefinite integral
p = sym.plotting.plot(f, show=False)
p.xlim = [0, 1]
p.ylim = [0, 1]
p.show()

f = x**3 / (x-2)
intf = sym.integrate(f)
p = sym.plotting.plot(f, show=False)
p.extend(sym.plotting.plot(intf,(x, 2.1, 10), show=False, line_color='r'))
p[0].label = '$f(x) = %s$'%(sym.latex(f))
p[1].label = '$\\int f(x) dx = %s$'%(sym.latex(intf))

p.legend = True
p.ylim = [-200, 200]
p.show()

# exercise:

f = 2*x**3 + sym.sin(x)

deriv = sym.diff(f)

int = sym.integrate(deriv)

# how to calculate the A between two curves
# S from a to b [f(x) - g(x)] * dx

x = sym.symbols('x')
symf = x**2
symg = x

f = sym.lambdify(x, symf)
g = sym.lambdify(x, symg)

x_vals = np.linspace(-2, 2, 55)

# add patch:
xpatch = np.linspace(0, 1, 100)
ypatch = np.vstack((g(xpatch), f(xpatch))).T

fig, ax = plt.subplots()
from matplotlib.patches import Polygon
ax.add_patch(Polygon(ypatch, facecolor='k', alpha=.3))

plt.plot(x_vals, f(x_vals))
plt.plot(x_vals, g(x_vals), 'r')

plt.legend(['$f(x) = %s$'%sym.latex(symf)])
plt.axis([-.25, 1.25, -.5, 1.5])
plt.show()

# let's get A between those 2 curves:
# using intercepts
# At an intersection point, both curves have the same x and the same y value
x = sym.symbols('x')
f = x**2
g = x

# Find intersection points
solutions = sym.solve(sym.Eq(f, g), x)  # gives [0, 1]

# Compute definite integral between intersections
area = sym.integrate(g - f, (x, solutions[0], solutions[1]))

# Area between 2 functions
# get area between 2 intercepts
# get definite integral of functions
x = sym.symbols('x')
fx = x**2
gx = x

f = sym.lambdify(x, fx)
g = sym.lambdify(x, gx)

intercepts = sym.solve(sym.Eq(fx, gx), x)

integral = sym.integrate((f, (x, 0, 1)))

# add patch:
xpatch = np.linspace(intercepts[0], intercepts[1], 100)
ypatch = np.vstack((g(xpatch), f(xpatch))).T

fig, ax = plt.subplots()
from matplotlib.patches import Polygon
ax.add_patch(Polygon(ypatch, facecolor='k', alpha=.3))

plt.plot(x_vals, f(x_vals))
plt.plot(x_vals, g(x_vals), 'r')
plt.show()
