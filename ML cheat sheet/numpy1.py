import numpy as np
# lesson one
a = range(5)  # create a list of 0 to 4
b = np.arange(1, 5)  # create array 1 to 4
b = np.arange(1, 5, 2)  # create array 1 to 5 with step of 2
c = np.linspace(1, 5, 10)  # create array of 10 values between 1 and 5 with same amount of space between them
d = np.array([[1, 2], [3, 4], [5, 6]])  # create array
d.shape  # shape of the array which is here 3 rows and 2 columns it won't change the original array
d.reshape(2, 3)  # reshape the array into 2 rows and 3 columns it won't change the original array
d.ravel()  # it will flat the array into 1 column or 1 dimension it won't change the original array

d.min()  # return minimum argument it's 1 here
d.max()  # return maximum argument , it's 6 here
d.sum()  # return sum of all the arguments
d.sum(axis=0)  # return sum of the every column
d.sum(axis=1)  # return sum of the every rows
np.sqrt(d)  # return (jazr 2) every argument in the array
np.std(d)  # return standard deviation of all arguments( enheraf meyar )

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a + b  # it will add row 1 and column 1 of b and a and so on, output will be [[6,8],[10,12]]
# a-b or a*b or a/b # it will do every action for argument in same row and column
c = a.ndim  # shows dimension of the array
c = a.itemsize  # shows byte size of the elements here we have digits then it will be 4
c = a.dtype  # show data type of the elements
a = np.array([[1,2],[3,4],[5,6]], dtype=np.float64) # changing the type of elements to float64
c= a.size  # shows the size of the array , the number of elements
c= np.zeros((3,4)) # create array of zeroes , and we enter the shape
c= np.ones((3,4)) # create array of ones
