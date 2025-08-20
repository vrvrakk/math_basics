'''
git remote add origin https://github.com/vrvrakk/math_basics.git
git branch -M main
git push -u origin main
'''

# correction: Section 2: Arithmetic
# %g
base = 2
for num in range(10):
    solution = base ** num
    print('%g to the power of %g is equal to %g' % (num, base, solution))
# if statements
# exercise:
# loop over numbers 0-3 (variable i),
# loop over 0-4 (var j)
# print i to the power -j where i > 0, j> 0
for i in range(4):
    for j in range(5):
        if i > 0 and j > 0:
            solution = i**-j
            print('%g^%g = %g' % (i, -j, solution))

# absolute value
# the distance of the number to the origin of the number line
# I guess without direction taken into account

# exercise: print the absolute value of each of a set of numbers
# only if each number is < -5 or > 2
nums = [-4, 6, -1, 43, -18, 2, 0]
# expected output: -4 was not tested. absolute value of 6 is 6.
for num in nums:
    if num < -5 or num > 2:
        solution = abs(num)
        print('Absolute value of %g is %g' % (num, solution))
    else:
        print('%g was not tested.' % (num))

# division remainder (modulus)
a = 10
b = 3
# integer division
c = int(a/b)
d = a%b # basically: how many times dos b fit into a? what remainds?
print('%g goes into %g, %g times with a remainder of %g.' % (b, a, c, d))

for i in range (-5, 6, 1):
    if i % 2 == 0: # if we divide said number into two, there is 0 remainders
        print('%g is an even number.' % i)
    else:
        print('%g is an odd number.' % i)

# interactive math functions
def myfunction():
    print('hello world')

myfunction()

def computeremainder(x, y):
    divis = x/y
    remainder = x%y
    print('%g goes into %g, %g times with a remainder of %g' %(y, x, divis, remainder))

computeremainder(5, 2)

def divisionWithInput():
    x = int(input('numerator: '))
    y = int(input('denominator: '))
    divis = x/y
    remainder =  x%y
    print('%g goes into %g, %g times with a remainder of %g' %(y, x, divis, remainder))

divisionWithInput()

# function challenge:
# two function that will compute x^y and y/y
# input 3 nums: x, y and function switch
# call requested function
def exponent(x, y):
    solution = x ** y
    return solution
def division(x, y):
    solution = x / y
    return solution

def function_switch():
    x_input = int(input('numerator: '))
    y_input = int(input('exponent: '))
    func_switch = input('Press "1" to compute %g^%g or "2" to compute %g/%g: ' % (x_input, y_input, x_input, y_input))
    if func_switch == '1':
        solution = exponent(x_input, y_input)
        print(solution)
    elif func_switch == '2':
        solution = division(x_input, y_input)
        print(solution)
    else:
        print('Invalid function switch!')

function_switch()

# guess the number game
import random
def guessTheNumber():
    random_number = random.randint(1, 100)
    user_input = int(input('Guess a number between 1 and 100: '))
    while True:
        if user_input == random_number:
            print('Got it! The right number was %g and your final guess was %g.' % (random_number, user_input))
            break
        elif user_input < random_number:
            print('Guess higher!')
            user_input = int(input('Guess again: '))
        elif user_input > random_number:
            print('Guess lower!')
            user_input = int(input('Guess again: '))

# arithmetic bug hunt:
# 1
# ok obviously if he wants to print x+y, x and y need to be ints not str
# therefore:
x = '1'
y = '2'
x_int = int(x)
y_int = int(y)
print(x+y)

# 2
x, y = 4, 5
print('%g + %g = %g' % (x, y)) # obviously the solution %g is missing
# therefore:
print('%g + %g = %g' % (x, y, x+y)) # obviously the solution %g is missing

# 3
# I think it's the %i? yep

# 4
3 ** 2 # not 3^2

# 5: just needed a colon at the for loop end (:)

# 6:
a = 10
b = 20
result = 2*a <= b # changed the >
print(result)

# 7. if a+b*2 > 40 then: obviously then is wrong

# 8.
# var = input(input a number!) # lol
var = input('input a number! ')

# 9
4/10 # not 4\10

# 10
w = int(input('Input a num: '))
z = int(input('Input another num: '))
print('The sum of %s and %s is %s' % (w, z, w+z)) # int was missing before input
# if I used input and %s it would give no error but the end result would be false:
# i.e. 3+3 = 33 instead of 6

# 11: uncompleted parenthesis EOF while parsin error (end of file)
# also 9/3 not 9/4 on both sides
9/3 == 9/4
# 12
t = 1
while t < 11:
    print(t) # t will always be smaller, therefore it will get stuck in the loop
