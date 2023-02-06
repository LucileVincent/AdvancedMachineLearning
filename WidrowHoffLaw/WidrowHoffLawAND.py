# AND

# Learning rate
lr = [0.002, 0.02, 0.2, 2]

# Input vector (x0, x1)
X = [[0, 0], [1, 0], [0, 1], [1, 1]]

# Output
c = [0, 0, 0, 1]

# set of weights
w = [[-1, 1], [-0.8, 0.1]]
b = [0.5, 0]  # Bias

def train(w, x, c, b, lr):
    for _ in range(1000):
        for i, entry in enumerate(x):  # enumerate : parcourt tt
            theta = prediction(w,entry,b)
            for j, weight in enumerate(w):
                weight = weight+lr*(c[i]-theta)*entry[j]
                w[j] = weight
            b = b+lr*(c[i]-theta)
    return w, b


def prediction(w, x, b):
    TF = 0
    for weight, xi in zip(w,x):  # zip : met 2 arrays ensemble de meme taille
        TF += weight*xi
    TF += b
    return TF > 0


for weight in w:
    for bias in b:
        for learning_rate in lr:
            new_weight = weight.copy()
            print("Weight : ", weight, ", Bias : ", bias, ", Learning rate : ", learning_rate)
            print(train(new_weight, X, c, bias, learning_rate))