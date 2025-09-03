import numpy as np
# obviously integers and floats..
x = 7
y = 7.0

print(x, y)

print(type(x))
print(type(y))

for i in range(x): # from 0 to c-1
    print(i)

for i in range(y): # from 0 to d-1
    print(i) # ERROR: CANNOT BE INTERPRETED AS INTEGER: TYPE ERROR
x+y # turns into float the output

# strings
firstName= 'Mike'
lastName= 'Johannson'

firstName + lastName
firstName * lastName # not possible babes
firstName * 3

%whos # get info on all the vars!

# Exercises:
# convert across data types
s1 = '4'
s2 = '4.7'
n1 = 5
n2 = 5.8

# convert str to int and str to floats
# int to str
import numpy as np
s1_to_n1 = int(s1)
s2_to_n2 = int(np.round(float(s2)))

s1_to_f1 = float(s1)
s2_to_f2 = float(s2)

n1_to_s1 = str(n1)
n2_to_s2 = str(n2)

# lists and numpy arrays
lolix = [0, 1, 2, 3, 4, 5]
type(lolix)
listlist = [3,
            lolix,
            5.5]
slist = ['1', '2', '3', '4', '5']

lolix + slist

lolix * 3

# numpy
list_np  = np.array(lolix)
print(lolix)
print(list_np) # no commas

list_np * 3 # it multiplies each element

lolix + 3 # cannot concatenate int in list, only list (TypeError)
lolix + [3]

# exercise
for i, element in enumerate(listlist):
    element_type = type(element)
    # print('List element %g is %g and is of type %s.' % (i, element, (element_type))) # nevermind
    print(f'List element {i} is {element} and is of type {element_type}.')

# how to remove the '' from the element type
for i, item in enumerate(listlist):
    t = str(type(item))[8:]
    apost = t.find("'") # todo: remember this shite: find
    t_trimmed = t[:apost]
    print(f'List element {i} is {element} and is of type {t_trimmed}.')



s = ('Mike')
for i, letter in enumerate(s):
    print(i, letter)