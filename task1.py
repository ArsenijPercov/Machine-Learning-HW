import math  
def inner_product(a,b):
    if len(a) != len(b):
        return 0
    sum = 0
    for i in range(len(a)):
        sum+=a[i]*b[i]
    return sum

def mat_product(a,b):
    result = [[0 for x in range(len(b[0]))]for y in range(len(a))]
    for i in range(len(result)):
        for j in range(len(result[0])):
            cellres = 0
            for q in range(len(a[0])):
                #print("a",i,q,a[i][q]," b",q,j,b[q][j],"=",cellres)
                cellres+=a[i][q]*b[q][j]
            result[i][j] = cellres
            #print(i,j,"=",cellres)
    return result


def i2_norm(x):
    sum = 0
    for i in x:
        sum+=i*i
    return math.sqrt(sum)


a = [[10,20],[20,30],[30,10]]
b = [20,30,40]
print(i2_norm(b))