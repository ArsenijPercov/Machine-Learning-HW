import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

class Linear_Regrssion: 

    def fit(self,x,y):
        self.x = x
        self.y = y

    def make_X(self):
        temp = np.array([[1]*len(self.x)])
        self.bigx = np.transpose(np.append(temp,[self.x], axis =0 ))
        return self.bigx

    def find_b(self):
        self.X_t = np.transpose(self.bigx)
        self.b = np.matmul(self.X_t,self.y)
        return self.b

    def find_a(self):
        self.A = np.matmul(self.X_t,self.bigx)
        return self.A

    def cholesky(self):
        n = len (self.A)
        L = [[0.0]*n for i in range(n)]
        for i in range(n):
            for k in range(i+1):
                tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                if (i == k):
                    L[i][k] = sqrt(self.A[k][k]-tmp_sum)
                else:
                    L[i][k] = (1.0/L[k][k])*(self.A[i][k]-tmp_sum)
        self.L = L
        return L

    def solve1(self):
        n = len(self.L)
        res = [0.0]*n
        for i in range(n):
            temp = 0
            for j in range(i+1):
                if (i==j):
                    res[i] = (self.b[i]-temp)/self.L[i][j]
                else:
                    temp += res[j]*self.L[i][j]
        self.fact_mid = res
        return res
    
    def solve2(self):
        
        L_t = np.transpose(self.L)
        print(L_t)
        n = len(L_t)
        res = [0.0]*n
        for i in reversed(range(n)):
            temp = 0
            for j in reversed(range(i,n)):
                if (i==j):
                    res[i] = (self.fact_mid[i]-temp)/L_t[i][j]
                else:
                    temp += res[j]*L_t[i][j]
        self.fact = res
        return res
    
    def predict(self,x):
        self.make_X()
        self.find_b()
        self.find_a()
        self.cholesky()
        self.solve1()
        self.solve2()
        res = [0]*len(x)
        count = 0
        for i in x:
            z = [[1],[i]]
            res[count] = np.matmul(np.transpose(z),self.fact)
            count += 1
        return res


csv_file = pd.read_csv(r'salary_data.csv')
data_years = np.array(csv_file['YearsExperience'].tolist())
data_salary = np.array(csv_file['Salary'].tolist())
plt.scatter(data_years,data_salary)


lr = Linear_Regrssion()
lr.fit(data_years,data_salary)

pred_data = np.linspace(0,11,1000)
pred_res = []
pred_res = lr.predict(pred_data)
plt.plot(pred_data,pred_res, label="lin regression predictor")
plt.show()