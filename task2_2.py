class KNNRegrssion:
    n_neighbors = 0 
    
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    def fit(self,x,y):
        self.x = x
        self.y = y
    
    def predict(self,X_eval):
        res = [0]*len(X_eval)
        c= 0
        for i in X_eval:
            left_lim = max(0,i[0]-self.n_neighbors)
            right_lim = (i+self.n_neighbors)[0]
            #print("left_lim=",left_lim,"right = ",right_lim)
            sum = 0
            count = 0
            for j in range(0,len(self.x)):
                if (self.x[j][0]>=left_lim and self.x[j][0]<right_lim):
                    sum += self.y[j]
                    count+= 1
            #print(c,sum/count)
            res[c] = (sum/count)
            print(res[c])
            c += 1
        return res
        

import numpy as np
import matplotlib.pyplot as plt

X = [[0], [1], [2], [3], [4]]
y = [0, 0.3, 0.75, 1, 2]

neigh = KNNRegrssion(n_neighbors=2)
neigh.fit(X, y)

X_eval = np.linspace(0,4,1000)
X_eval = X_eval.reshape(-1,1)

plt.figure()
res = neigh.predict(X_eval)
#print(res)
plt.plot(X_eval,res, label="kNN regression predictor")
plt.plot(X,y, 'rs', markersize=12, label="trainin set")
plt.title("Simplistic test of kNN regression")
plt.show()