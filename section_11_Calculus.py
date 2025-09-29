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