from cmath import sqrt
import numpy as np
import math
import matplotlib.pyplot as plt

def npdf(x,mean,stdd):
    return (1/math.sqrt(2*math.pi)*stdd)*math.exp((-1/2)*((x-mean)**2/stdd**2))

def p(x):
    return 0.3*npdf(x,2,1) + 0.4*npdf(x,5,2) + 0.3*npdf(x,9,1)
    
def uniform_aux(x):
    return p(x)/(1/15)

def normal_aux(x):
    return p(x)/npdf(x,5,4)
  

def sample1():
    samples = np.random.uniform(0,15,200)
    return samples

def sample2():
    samples = np.random.normal(5,4,200)
    return samples

def u_weight(samples):
    new_samples = np.array(list(map(uniform_aux,samples)))
    return new_samples

def n_weight(samples):
    new_samples = np.array(list(map(normal_aux,samples)))
    return new_samples
    
def normalize_weights(weights):
    normalized = []
    for w in weights:
        normalized.append(w/(sum(weights)))
    return normalized

def resample(samples,n_weights):
    resamples = np.random.choice(samples,200,p=n_weights,replace=True)
    return resamples
    
    

samples = sample2()
weights = n_weight(samples)
n_weights = normalize_weights(weights)
resamples = resample(samples,n_weights)

x1 = list(range(0,15,1))
x = []
for i in x1:
    x.append(p(i))
x2 = []
for i in x:
    x2.append(i*10) 


plt.subplot(1,2,1)
plt.plot(x2,'r')
plt.hist(resamples)
plt.scatter(samples,n_weights,color='black')
plt.title("resamples")

plt.subplot(1,2,2)
plt.hist(samples)
plt.scatter(samples,n_weights,color ='black')
plt.title("samples")
plt.plot(x2,'r')
plt.show()
