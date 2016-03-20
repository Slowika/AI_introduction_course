import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

input_data = np.array([
    [1,1],
    [-1,1],
    [-1,0],
    [1,-1],]
)

output_data = np.array([[0,0,0,1]]).T

b = 10 #ustalone

def sigmoid(x1, x2, w1, w2):
    return 1/(1+np.exp(-b*(x1*w1+x2*w2)))

#def derivative()

def e(w1, w2, x1, x2):
    sum = 0
    i = 0
    while i<4:
        sum += (sigmoid(x1, x2, w1, w2) - output_data[i])/2
        i = i+1
    return sum

w1 = np.linspace(-20, 20, num=200)
w2 = np.linspace(-20, 20, num=200)

def countError(w1, w2):
    error = 0
    for i in input_data:
        error += e(w1, w2, i[0], i[1])

    return error

totalError = [[countError(i, j)[0] for j in w2] for i in w1]

fig = plt.figure()
ax = fig.gca(projection='3d')
w1, w2 = np.meshgrid(w1, w2)

surf = ax.plot_surface(w1, w2, totalError, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

plt.show()

