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