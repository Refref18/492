import numpy as np

# Create a 3D array with dimensions (2, 3, 4)
arr = np.arange(24).reshape((2, 3, 4))
print(arr)
# Output:
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

# Transpose the array using transpose(1, 0, 2)
transposed_arr = arr.transpose(1, 0, 2)
print(transposed_arr)
# Output:
# [[[ 0  1  2  3]
#   [12 13 14 15]]
#
#  [[ 4  5  6  7]
#   [16 17 18 19]]
#
#  [[ 8  9 10 11]
#   [20 21 22 23]]]
