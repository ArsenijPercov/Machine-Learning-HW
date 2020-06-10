print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
target_names = iris.target
data = iris.data[:,0:2]
print(data)
plt.scatter(data[:, 0],data[:,1],c=target_names)
plt.show()


x = np.linspace(0,2*np.pi)
plt.plot(x,np.sin(x))
plt.show()