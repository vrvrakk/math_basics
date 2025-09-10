# ALGEBRA 2
import numpy as np
import sympy as sym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

a = [1, 3, 4, 1, 5]

# SUMMATION: a1+a2+a3+an+1
sum = np.sum(a)
# PRODUCT:
product = np.prod(a)
# a1*a2*a3*a4*an+1

# CUMULATIVE SUM
# y1 = a1
# y2 = a1 + a2
# y3 = a1 + a2 + a3

cum_sum = np.cumsum(a)

plt.plot(cum_sum, 'rs-')
plt.plot(a, 'bo-')
plt.legend(['cumulative sum', 'original order'])
plt.show()

# Exercises

ex1 = 1 / sum
e2 = 1 / product

# Exercise
num_list = [1, 2, 3, 4, 5]
num_list_mult = []
for i, num in enumerate(num_list):
    multiplicator = i+1
    res = num * multiplicator
    num_list_mult.append(res)


# DIFFERENCES
x = [1, 2, 4]
np.diff(x) # 2-1 and 4-2

y = np.arange(0, 11)
np.diff(y) # output will always be one number less than the original function
np.diff(y, 2) # second-odrer difference
# it's like saying:
np.diff(np.diff(y))

# Exercise: plot the function f(x) = x**2
x = sym.Symbol('x')
function = x**2
df = sym.diff(function)
dx = sym.diff(x)
equation = df / dx
val_range = np.linspace(-2, 2, 101)

y1 = [function.subs(x, i) for i in val_range]
y2 = [equation.subs(x, i) for i in val_range]

plt.plot(val_range, y1, 'b-', label='f')
plt.plot(val_range, y2, 'r-', label='df')


plt.xlim(-2, 2)
plt.ylim(-1, 2)
plt.legend()
plt.grid()
plt.show()

# ROOTS OF POLYNOMIALS
# a0 + a1x + a2x**2 + ... + anx**n = 0
# set to 0, solve for x -> roots
x = sym.Symbol('x')
polynomial = 3* (x**2) + 2*x - 1
# define the coefficients:
coeffs = [3, 2, -1] # need to be in descending order
roots = np.roots(coeffs) # two solutions to setting the polynomial to 0

for i in roots:
    print('$At x=%g, %s = %g$' %(i, sym.latex(polynomial), polynomial.subs(x, i)))

# Exercise on polynomial roots:
# generate a degree-N polynomial and count the number of roots
for i in range(1, 11):
    coeffs = np.arange(1, i+1)
    print('A degree-%s polynomial has %s roots' %(len(coeffs)-1, len(np.roots(coeffs))))

# The quadratic equation:
# ax**2 + bx + c = 0
# solution:
# x = -b +- sqrt(b**2 - 4ac) / 2a

a, b, c = 2, 7, 5

quadeqP= (-b + np.sqrt(b**2 - 4*a*c)) / (2 * a)
quadeqN= (-b - np.sqrt(b**2 - 4*a*c)) / (2 * a)

import scipy as sp
def quadeq(a, b, c):
    # initialize
    out = np.zeros(2)
    out[0] = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    out[1] = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return  out

out = quadeq(a, b, c)

# exercise
# compute the quadratic equation for:
a = 1
b = np.arange(-5, 6)
c = np.arange(-2, 11)
M = np.zeros((len(b), len(c)))
for bi, b_val in enumerate(b):
    for ci, c_val in enumerate(c):
        out = quadeq(a, b_val, c_val)
        M[bi, ci] = out[0]
plt.imshow(M, extent=[c[0], c[-1], b[0], b[-1]],cmap='viridis')
plt.xlabel('c')
plt.ylabel('b')
plt.show()

# IMAGINARY OPERATOR
# x**2 + 1 = 0
# x**2 = -1
# x = +- sqrt(-1)
# therefore: x = i -> a special operator for such cases..

# z = a * bi
# z complex number
# a real part
# b imaginary part
# i imaginary operator

# Addition and subtraction with complex numbers
# z = a bi
# w = c di
# z + w = (a + c) (b + d)i
# z - w = (a - c) (b - d)i

# examples
print(1j) # only j works. not any other letter
np.sqrt(-1)
# you can force the dtype to be complex
print(np.sqrt(-1, dtype='complex'))

real_part = 4
imag_part = -5

# two ways to create complex numbers
# np.complex deprecated
cn1 = complex(real_part, imag_part)
cn2 = real_part + 1j* imag_part

z1 = complex(4, 5)
z2 = complex(5, 6)

print(cn1, cn2)

np.real(z1)
np.imag(z1)

# Exercise on imaginary nums:
w_real = 2
w_imag = 4

z_real = 5
z_imag = 6

w = complex(w_real, w_imag)
z = complex(z_real, z_imag)

eq1 = w + z
eq2 = complex(np.real(w) + np.real(z), np.imag(w)+np.imag(z))

assert eq1 == eq2

# IMAGINARY NUMBERS CONNJUGATE & MULTIPLICATIONS ETC
# z = a bi
# w = c di

# z * w = (a bi) * (c di) = ac + adi + cbi + bdi^2 tbh what I already guessed
# i^2 = -1 !!
# ac + adi + cbi - bd

# Complex conjugate
# z*
# if z = a + bi
# then z* = a - bi (complex conjugate)

# and if z = a - bi, then the z* = a + bi -> essentially flip the imaginary number
# z = a bi
# then apparently... z = a - bi ? whyyyyyyy I think it's wrong

z1 = complex(4, 5)
z2 = complex(6, -2)

# basically z1 * z2
w = np.real(z1) * np.real(z2)  + np.real(z1) * np.imag(z2)*1j + np.imag(z1)*1j*np.real(z2) + np.imag(z1)*1j*np.imag(z2)*1j
print(w)
print(z1 * z2)
sym.sympify(z1 * z2)
np.conj(z1)

# exercise: use sympy to show that z * z* = a^2 + b^2
# standard approach: z=a+bi
a, b = sym.symbols('a b', real=True) # secret ingredient is real=True

z = a + b*sym.I
z_conj = sym.conjugate(z)
w = sym.expand(z * z_conj)
print(w)

# COMPLEX NUMBERS: DIVISION
# example
# 4 * 2i / 2 = 2 + i

# 4 + 2i / 2 - 3i
# ok apparently the solution is:
# (4 + 2i) * (2 + 3i) / (2 - 3i) (2 + 3i) = 2 + 16i / 13 -> denominator is always a real value

# z / w = z * w* / w * w*
z = complex(4, 2)

z1 = complex(4, 2)
z2 = complex(2, -3)

print('$\\frac{%s}{%s}$' %(z1, sym.latex(sym.sympify(z2))))

# exercise: simplify equation
z = complex(4, 2)
w = complex(3, 5)

eq1 = sym.simplify((z - 1) * (z*w + w) / w*z - w)
eq2 = sym.simplify((w-1) * (1 - w) / -z*w**2 - w**2 + 2*w*z + 2*w - z - 1)

res = eq1 * eq2