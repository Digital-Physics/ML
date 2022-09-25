import numpy as np

a = np.array([[1.1, 2.8, 3.5],
              [2.4, 3.4, 5.6]])

print("round every entry in the array w/ np.rint()")
b = np.rint(a)

print(b)