# Section 5: Algebra
import numpy as np
import sympy as sym
from IPython.display import display, Math

x = sym.symbols('x')

expr = 2*x + 4 - 9
solution = sym.solve(expr) # it's in a list, because some equations have more than one solution
# i.e.
sol = sym.solve(x**2 - 4)
type(sol)  # a list!
# now print: the solution is x = solution
print(f'The solution is {sym.latex(expr)} = {solution[0]}')  # not precisely

expr = x**2 - 4
sol = sym.solve(expr)

for i in range(0, len(sol)):
    print(f'Solution #{i} us {sol[i]}')

y = sym.symbols('y')
expr = x/4 - x*y +5
sym.solve(expr, x)
sym.solve(expr, y)

# exercises in Algebra
# 1
# 3*q + 4/q + 3 = 5q + 1/q + 1
# implement in sympy
# simplify
# solve for q
q = sym.symbols('q')
expr1 = 3*q + 4/q + 3
expr2 = 5*q + 1/q + 1
eq = 3*q + 4/q + 3 - 5*q - 1/q - 1 # we need to convert it into a homogenous equation == 0
sym.simplify(eq)
sym.solve(eq)

# 2
# 2q + 3*q^2 - 5/q - 4/q^3
# implement, apply functions simplify and cancel
eq = 2*q + 3*q**2 - 5/q - 4/q**3
simplified_eq = sym.simplify(eq)
# Cancel common factors in a rational function ``f``.
cancelled_eq = sym.cancel(simplified_eq)
sym.solve(cancelled_eq)
# 3
# sqrt(3) + sqrt(15) * q / sqrt(2) + sqrt(10) * q
# implement, simplify, confirm algebra with paper and pencil
# test in python by substituting numbers for q
eq = (np.sqrt(3) + np.sqrt(15) * q) / (np.sqrt(2) + np.sqrt(10) * q)
simplified_eq = sym.simplify(eq)
simplified_eq.subs(q, 10).evalf(2) # two decimels of precision (default 15)
# why is it giving 1.22 no matter what?
# ok I estimated visually, answer is sqrt(6)/2, that's why

# 33. expanding terms:
# example: a(b + c) = ab + ac etc
x = sym.symbols('x')
# NOW A NEW WAY FOR SYMBOLIC VARS
from sympy.abc import x
term1 = (4*x + 5)
term2 = x
print(term1 * term2)
sym.latex(sym.expand(term1 * term2))

from sympy.abc import y
expr = x*(2*y**2 - 5**x/x)
sym.latex(sym.expand(expr))

# expanding exercise

function = (4 + x) * (2 - y)
# expand for when x and y are 0, 1 and 2 respectively
numbers = [0, 1, 2]
# well we didn't actually use expand lol.
for xi in numbers:
    for yi in numbers:
        sub_func = function.subs({x:xi, y:yi})
        print(f'When x = {xi} and y = {yi}, f(x, y) = {sub_func}')

# Matrices - linear algebra
# all about matrix operations
# referred to by uppercase letters
# first rows, then columns (x, y)
A = [ [1, 2], [3, 4] ]
# convert to matrix
A = np.array(A)
sym.sympify(A)

# create a matrix with just zeros
mat = np.zeros([4, 6])  # four rows, six columns of zeros
# assign diff numbers to individual elements within mat
mat[0, 1] = 2  # first row, second column element == 2
mat[2, 4] = 7

numrange = range(0, 4)
for row_i in numrange:
    for column_j in numrange:
        mat[row_i, column_j] = (-1)**(row_i + column_j)

# exercise: create a matrix (3 rows, 4 columns) with the function
function = sym.sympify((4 + x) * (2 - y))
numrange = range(0, 3)
M = np.zeros([3, 3])
# for all combinations of x and y [0, 1, 2]
for xi in numrange:
    for yi in numrange:
        solution = function.subs({x:xi, y:yi})
        M[xi, yi] = solution

# create a multiplication table
x_list = list(range(1, 11))
y_list = x_list.copy()
M = np.zeros([10, 10])
for xi in x_list:
    for yi in y_list:
        solution = xi * yi
        M[xi-1, yi-1] = solution # -1 needs to be added because the Matrix indices start with 0 and end with 9
        # IndexError: index 10 is out of bounds for axis 1 with size 10
M = sym.Matrix(M)

# 36. associative, commutative and distributive properties
# Associative
# 2(3*4) = (2*3)4 -> both 24=24
# a(b*c) = (a*b)c
# Commutative
# ab = ba # order does not matter -> same shit
# abc = bca = cba etc
# Distributive rule
# x(y + z) = xy + xz
# you can distribute the x outside of the parentheses to the elements inside the parentheses
# how to use them with sympy:
x, y = sym.symbols('x y')
exp1 = x * (4*y)
exp2 = (x * 4) * y
# associative: if == 0, then associative property true
exp1 - exp2
# commutative
e1 = 4 * x * y
e2 = x * 4* y
e3 = y * x * 4
e1.subs({x:3, y:2})
print(e1.subs({x:3, y:2}))
print(e2.subs({x:3, y:2}))
print(e3.subs({x:3, y:2}))
# all equal!

# distributive rule:
a, b, c, d = sym.symbols('a b c d')
expr = (a+b) * (c + d)
sym.expand(expr)

# exercises on the three rules: Association, Commutation, Distribution
z, w, x, y = sym.symbols('z w x y')
x = w*(4 - w) + 1 / w**2*(1+w)
f1 = x*(y + z)
f2 = 3/x + x**2

# display and simplify f1 * f2
# show that the commutative property holds (f1 = f2)
multiplication = f1 * f2
mult_simplified = sym.simplify(multiplication)
# f1 * f2 == f2 * f1
# if these two are equal, then subtractring from each other should be 0
sym.simplify(f1*f2 - f2*f1)

# List
# exercise on lists
x = sym.symbols('x')
e1 = 2*x + x*(4 - 6*x) + x
e2 = -x * (2/x + 4/x**2) + 4 + x / 4*x
e3 = (x + 3)*(x - 3)*x*(1/9*x)
list_ = [e1, e2, e3]
# implement these as a list??
# expand and print
# for a for loop
for e in list_:
    print(sym.latex(sym.expand(e)))

# indexing and slicing
# create a vector
vec = list(range(10, 21))
# access an individual item within list
vec[0:5:2]  # from 0 to 5 but in steps of 2
vec[::2]

# greatest common denominator: GCD
# the largest integer that divides into two numbers with no remainder
# i.e. gcd(2, 4) = 2
# gcd(6, 15) = 3
# 6, 17 = 1
# good for reducing fractions

import math
math.gcd(6, 15) # gcd only defined for integers
a, b = 16, 88
fact = math.gcd(a, b)
print(f"{a}/{b} = ({a} * {fact})/({b} * {fact})")

# Greatest common denominator exercises
# illustrate the following property using symbolic variables, then show an example with nums
#gcd(ca, cb) = c * gcd(a, b)
a, b, c = sym.symbols('a b c')
e1 = sym.gcd(c*a, c*b)
e2 = c * sym.gcd(a, b)

a = 15
b = 6
c = 3
print(math.gcd(a*c, c*b))
print(c * math.gcd(a, b))

# GCD 2:
# create a matrix that contains 10 rows and 15 columns, where each element is equal to 99
# populate the matrix such that the ith row and jth column contains gcd(i+1, j+1)
M = np.zeros([10, 15])
M.fill(99)
for xi in range(0, 10): # ok just had to change the range for x and y respectivley
    # to iterate over the 10 rows and 15 columns respectively
    for yi in range(0, 15):
        M[xi, yi] = math.gcd(xi + 1, yi + 1)

# dictionary
# dictionary exercises
x, y = sym.symbols('x y')
e1 = 4*x
a1 = 6
e2 = sym.sin(y)
a2 = 0
e3 = x**2
a3 = 9

eq_dict = {e1:a1, e2:a2, e3:a3}
for keys, values in eq_dict.items():
    equation = keys - values
    solution = sym.solve(equation)
    print(solution)
    # im so smart woo


# PRIME NUMBERS
# only nums divided by themselves and 1
# prime factorization: breaking down a number into prime multiplicands
number = 48
fact_dict = sym.factorint(number)
# {2: 4, 3:1}
# 2 * 2 * 2 * 2 + 3 * 1
print(fact_dict.keys())
# 48 not a prime number has more than 1 key, and more than 1 repetition of a number
# exercise: loop through integers from 2 to 50
# use factorint, list, len, if/else and print
# n is a prime number # is a composite number with prime factors [k:v]
int_list = np.arange(2, 51)
for integer in int_list:
    solution = sym.factorint(integer)
    repetition = 1
    if len(list(solution.keys())) == 1:
        if repetition in solution.values():
            print(f'{integer} is a primer number')
        elif repetition not in solution.values():
            print(f'{integer} is a composite number with prime factors {list(solution.keys())}')
    else:
        print(f'{integer} is a composite number with prime factors {list(solution.keys())}')
# success :)

# 43: SOLVING INEQUALITIES
# how to write inequalities in python
# using sym.solve
# how to indicate infinities in sympy
x = sym.symbols('x')
expr = 4 * x > 8
expr1 = (x - 1) * (x + 3) > 0
sym.solve(expr)
sym.solve(expr1)

a, b, c = sym.symbols('a b c')
ex = a*x > b**2/c
sym.solve(ex, x)

# exercise:
eq = 3*x / 2 + ((4 - 5*x) / 3) <= 2 - (5*(2 - x)/4)
sym.solve(eq)  # i got 22/17

# POLYNOMIALS
# a never-ending expression, kind of like a polymer
# a0 + a1*x + a2*x^2 + ... + an*x^n
from sympy.abc import x
p1 = 2*x**3 + x**2 - x
p2 = x**3 - x**4 - 4*x**2

p1 = sym.Poly(2*x**3 + x**2 -x) # domain=ZZ means all coefficients are integers
# what can we do with a poly class?
p1.eval(0)
p1.coeffs() # what are the 'x's multiplied with

# polynomial exercises
# 1
p1 = sym.Poly(x**2 + 2*x)
p2 = sym.Poly(-x**3 + 4*x)
p3 = sym.Poly(x**5 - x**4 + x/4 + 4)
p_list = [p1, p2, p3]

for p in p_list:
    degree = p.degree()
    if degree % 2 == 0:
        coeff_sum = np.sum(p.coeffs())
        print(f'The degree is even and the coefficient is sum {coeff_sum}')
    elif degree %2 != 0:
        coeff_count = len(p.coeffs())
        print(f'The degree is odd, and the coefficient count is {coeff_count}')

# multiplying POLYNOMIALS
p_mult = p1 * p2

# exercises:
y = sym.symbols('y')
p1 = sym.Poly(4*x**4 - 3*x**2 + x*y**2 - 9*y**3)
p2 = sym.Poly(-x**3 + 6*x**2*y + 8*y**3)
p_mult = p1 * p2
p_mult.subs({x:5, y:2})

# dividing polynomials:
div_p = p1/p2

# exercise:
p = sym.Poly(x**6 + 2*x**4 + 6*x - y / (x**3 + 3))
pNum = x**6 + 2*x**4 + 6*x - y
pDen = x**3 + 3
y_list = np.arange(5, 16)
for y_value in y_list:
    p_sub = pNum.subs({y:y_value})
    if sym.fraction(sym.simplify(p_sub/pDen))[1] == 1:
        right_answer = y_value
        print(f'The answer that satisfies our goal is {y_value}')

# Factoring polynomials: i.e. 4 = 2 * 2
# 16 = 4 * 4 -> factoring integers
# 8 = 2 * 2 * 2
# for polynomials:
p = x**2 + 4*x + 3
# is also equal to
# p = (x + 1) * (x + 3)
# how to solve it:
sym.factor(p)
# exercise: determine whether the following polynomials can be factored
p1 = x**2 + 4*x + 3
p2 = 2*y**2 - 1
p3 = 3*y**2 + 12*y
p_list = [p1, p2, p3]
for p in p_list:
    solution = sym.factor(p)
    if p == solution:
        print('Not factorable!')
    else:
        print(f'{p} factor is {solution}')

# BUG HUNT
# 1
from sympy.abc import x2
# OBVIOUSLY X2 IS NOT A LETTER

#2
a, b, c = sym.symbols('a b c')
expr = 4*b + 5*a*a - c**3 + 5*d # obviously d is not included and the a*a is weird -> a**2

# 3
import math
gcd(30, 50) # WRONG, it is math.gcd()

# 4
expr = 4*x - 8
solve(expr) # again: sym.solve

# 5:
import numpy as np
A = np.array([[1, 2], [3, 4]]) # make it look nice:
A = np.array([[1, 2],
             [3, 4]])

# 6:
fact_dict = sym.factorint(44)
allkeys = fact_dict.keys() # mistake is that it needs to be a list prior to looping
# list(fact_dict.keys())
for i in range(0, len(allkeys)):
    print('%g was present %g times.' %(i, allkeys[i]))

# 7:
x, y = sym.symbols('x y')
expr = 4* x - 5*y**2
expr.subs({x=5}) # it's :, not =

# 8:
f = 5/9
display(Math(sym.latex('\\frac{5}{9}')))
# or f=  sym.sympify(5)/9
# display(Math(sym.latex(f))

# 9:
from sympy.abc import x, y
expr = 2*x + 4*y
# solve for y
sym.solve(expr, y) # not without y. needs to be specified

# 10:
import numpy as np
A = np.array([[1, 2],[3, 4]])
# set the element in the second row, second column to 9
A[2, 2] = 9
print(A)
# it's A[1, 1] = 9, since indexing starts at 0. not 1.