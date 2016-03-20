import numpy as np
import matplotlib.pyplot as plt

def f(v):
    return [
        v[0]+v[2],
        v[1],
        v[2]+v[0],
        v[3]*v[2],
        v[4]+v[5]
    ]


def sigmoid(s):
    b = 1
    return 1 / (1 + np.exp(-b * s))


def deriv(s):
    return s * (1 - s)


input_data = np.random.random((500, 6))

output_data = np.array([f(v) for v in input_data])


np.random.seed(1)

syn0 = 2 * np.random.random((6, 1)) - 1
syn1 = 2 * np.random.random((1, 5)) - 1

iterations = []
errors = []

for i in range(100000):
    l0 = input_data
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l2_error = output_data - l2

    if i%100 ==0:
        print("Error: %f" % np.mean(np.abs(l2_error)))
        iterations.append(i)
        errors.append( float(np.mean(np.abs(l2_error))))


    l2_delta = l2_error * deriv(l2)
  
    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * deriv(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

fig = plt.figure()
print (iterations)
print (errors)
plt.plot(iterations, errors)
plt.show()
