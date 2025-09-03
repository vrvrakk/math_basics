# Sympy and LATEX
# symbolic vs numerical math
import sympy
import sympy as sym
sym.init_printing() # works only in jupyter lol

x = sym.symbols('x')
# or
x, y = sym.symbols('x y')

x = y + 4

# compare symbolic nums to arithmetic: sympy vs numpy
import numpy as np
ar_solution = np.sqrt(2)
sym_solution = sym.sqrt(2)

# sympy exercises
# 1
y = sym.symbols('y')
x = sym.symbols('x')

sym_func = y * (x**2)

# 2
sqr_four = sym.sqrt(4) * x


# 3
sym.sqrt(x) * sym.sqrt(x)

# now off to LaTeX

from IPython.display import Math, display
func = display(Math('3 + 4 = 7'))
display(Math(('\\sigma = \\mu \\times \\sqrt{5}'))) # LATEX CODE
display(Math('x_{mm} + y^{n + 2k - 15}'))

# fractions
display(Math('\\text {the answer to this equation is }'
             '\\frac{1+x}{2v-s^{t+4r}}')) # first {} the numerator, 2nd denominator

# §\frac{1+x}{2v-s^{t+4r}}§ -> indicates the start LaTeX coding

# exercises
# 1
display(Math('4x + 5y - 8z = 17'))
# 2
display(Math('sin(2\\pi f t + \\theta'))
'''
$\sin{(2\pi \times ft + \theta)}$\\
$e = mc^2$\\
$\frac{4+5x^2}{(1+x)\times(1-x)}$
'''

# integrating latx and sympy
mu, alpha, sigma = sym.symbols('mu, alpha, sigma')
expression = sym.exp((mu-alpha)**2 / (2*sigma**2)) # gaussian bell curve
print(expression)
sympy.init_printing()
display(expression) # so that in jupyter the printed expression would be cuter

hello = sym.symbols('hello')
hello/3

x = sym.symbols('x')
expression = x + 4
expression.subs(x, 100) # substitutes the symbolic variable with sth else
expression.subs(x, 'kolos') # works with strings as well

x, y = sym.symbols('x y')
expression2 = x + y
expression2.subs({x:-4, y:100})

expr = 3/x
sym.latex(expr)
sym.latex(sym.sympify(expr)) # code that can convert this into sym variable

# sympy and latex challenge
# x^2 + 4 , and substitute x with whatever n number is given
# then print out the latex form of the expression
x = sym.symbols('x')
expr = x**2 + 4
for i in range(-2, 3, 1):
    ans = (x + 4).subs(x, i**2)
    display(Math('\\text{With :} x = %g, x^2+4 \\quad \\Rightarrow \\quad %g^2+4 = %g' %(i, i, ans)))

# example use of sympy to understand the law of exponents
x, y, z = sym.symbols('x y z')
ex = x**y * x**z
sym.simplify(ex) # determines whether it's possible to simplify this expression
ex1 = x**y * x**z
ex2 = x**y / x**z
ex3 = x**y * y**z

'%s = %s' %(sym.latex(ex1), sym.latex(sym.simplify(ex1)))
# equation
lhs = 4
rhs = 6-2
sym.Eq(lhs, rhs)
sym.powsimp(ex1)
sym.Eq(sym.expand(ex1-sym.simplify(ex1)))
sym.expand((x + 1)**2)
sym.factor(x**3 - x**2 + x - 1)
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
sym.powsimp(x**y*x**z)

# final exercise: debugging
# 1
# need to insert the modules: import sympy as sym
import sympy as sym # this and the line below were missing
from IPython.display import display, Math
mu, alpha = sym.symbols('mu, alpha')
expr = 2 * sym.exp(mu**2/alpha)
display(Math(expr)) # this also needs to be converted into latex code
display(Math(sym.latex(expr)))

# 2
Math('1234 + \frac{3x}{\sin(2\pi t+\theta)}') # need to use double slashes for latex code in python
Math('1234 + \\frac{3x}{\\sin(2\\pi t+\\theta)}')

# 3
a = '3'
b = '4'

print(sym.sympify(a+b)) # this will give 34 instead of 7 bc they are strings

print(sym.sympify(int(a+b)))

# 4
sym.Eq(4*x = 2) # obvi the x needs to be defined
x  = sym.sympify('x')
# it's lhs and rhs whatever they mean
sym.Eq(4*x,  2)
# fun fact: left hand side, right hand side lol

# 5
q = x^2 # FALSE
r = x**2
display(q)

q = x**2

# 6
q, r = sym.symbols('q, r')
q = sym.sympify('x^2') # if you use sympify you can use this ^
r = sym.sympify('x**2')
display(q)
display(r)

sym.Eq(q, r)

# 7
x = sym.symbols('x')
equation = (4*x**2 - 5*x + 10)**(1/2)
display(equation)
sym.subs(equation, x, 3)
# I think this is supposed to be substitution?
# system, symbols, result, known symbols, exclude
sym.substitution(equation, x, 3) # it also crashes and says 'pow' not iterable
# nevermind also wrong..goal is:
equation.subs(x, 3) # get output if we replace x with 3

# 8
x, y = sym.symbols('x y')
equation = 1/4*x*y**2 - x*(5*x + 10*Y**2)**(3) # OBVIOUSLY IT'S THE Y
equation = 1/4*x*y**2 - x*(5*x + 10*y**2)**(3)
display(equation)