
import numpy as np
import matplotlib.pyplot  as plt

# это коллекция функций, которые делают Matplotlib похожим на MATLAB. Это удобный высокоуровневый интерфейс 
# для быстрого создания различных типов графиков.
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def devil(N,D=2,K=3):
    N=100
    D=2
    K=3
    X=np.zeros((N*K,D))
    y=np.zeros(N*K, dtype="uint8")

    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N)
        t=np.linspace(j*4,(j+1)*4,N)
        X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
        y[ix]=j
    return X,y

X,y = devil(300)
X[:, 0]+=np.random.normal(loc=0,scale=0.15,size=300)
X[:,1]+=np.random.normal(loc=0,scale=0.15,size=300)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

x_min,x_max = (-1,1)
y_min,y_max = (-1,1)

h=0.05

plt.figure(figsize=(12,12))
plt.scatter(X[:,0],X[:,1],c=y)

plt.xlim((x_min,x_max))
plt.ylim((y_min,y_max))

plt.grid(True)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.6)

model = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)


plt.figure(figsize=(12,12))
plt.xlim((x_min,x_max))
plt.ylim=((y_min,y_max))
xx,yy =  np.meshgrid(np.linspace(x_min, x_max,100),np.linspace(y_min,y_max,100))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,label = "test points")

plt.legend()

plt.show()