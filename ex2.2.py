from cmath import sqrt
from pickle import TRUE
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
    samples = np.random.uniform(0,15,1000)
    return samples

def sample2():
    samples = np.random.normal(5,4,1000)
    return samples

def weight1(samples):
    new_samples = np.array(list(map(uniform_aux,samples)))
    return new_samples

def weight2(samples):
    new_samples = np.array(list(map(normal_aux,samples)))
    return new_samples
    
def normalize_weights(weights):
    normalized = []
    for w in weights:
        normalized.append(w/(sum(weights)))
    return normalized

def resample(samples,n_weights):
    resamples = np.random.choice(samples,1000,p=n_weights,replace=True)
    return resamples
    
    

samples_uni = sample1()
samples_norm = sample2()
weights_uni = weight1(samples_uni)
weights_norm = weight2(samples_norm)
n_weights_uni = normalize_weights(weights_uni)
n_weights_norm = normalize_weights(weights_norm)
resamples_uni = resample(samples_uni,n_weights_uni)
resamples_norm = resample(samples_norm, n_weights_norm)

x1 = list(range(0,15,1))
x = []
for i in x1:
    x.append(p(i))
x2 = []
for i in x:
    x2.append(i) 


plt.subplot(2,2,1)
plt.plot(x2,'r')
plt.hist(samples_uni,density=TRUE)
plt.title("uniform samples")

plt.subplot(2,2,2)
plt.plot(x2, 'r')
plt.hist(resamples_uni,density=TRUE)
plt.title("uniform resamples")


plt.subplot(2,2,3)
plt.hist(samples_norm,density=TRUE)
plt.title("normal samples")
plt.plot(x2,'r')

plt.subplot(2,2,4)
plt.hist(resamples_norm,density=TRUE)
plt.title("normal resamples")
plt.plot(x2,'r')

plt.show()
