import numpy
import cv2

print(cv2.__version__)
print(numpy.__version__)

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
print(" ")  

s = "hello"
print(s.capitalize())  # Capitalize a string
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces
print(s.center(7))     # Center a string, padding with spaces
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
print('  world '.strip())  # Strip leading and trailing whitespace
print(" ")

xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
# -1 means the end of list
print(xs[-1])     # Negative indices count from the end of the list; prints "2"

xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)

xs.append('bar') # Add a new element to the end of the list
print(xs)

x = xs.pop()     # Remove and return the last element of the list
print(x, xs)