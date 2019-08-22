import numpy as np
import sys
import truth as t
import model as m
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import check_grad


n = 60

Tinf = np.array([20.0,30.0,40.0,50.0])
#Tinf = np.array([40.0,50.0])
Tinf_holdout = np.array([25.0,35.0,45.0,55.0,-1])
#Tinf_holdout = np.array([45.0,55.0])
#zt = range(0,n)
zt = np.linspace(0,1,n)
Tt_avg = np.zeros((n,1))
beta = np.ones((n,1))

zt = np.linspace(0,1,n)
Tt_avg = np.zeros((len(Tinf),n))
Tm = np.zeros((len(Tinf),n))
beta_best = np.zeros((len(Tinf),n))
beta = np.ones((n,))
Ttemp = np.zeros((n,))
num_samp = 100
stop_resid = 10.0E-12

import truth as t

for j in range(0,len(Tinf)) :
	for i in range(0,num_samp) :
		output = t.truth(Tinf[j],n,stop_resid)
		#sys.stdout.write(str(output)+'\n')
		Tt_avg[j,:] += np.reshape(output,(n,))
Tt_avg /= np.float(num_samp)

import model as m

#Run baseline model for training data temps
of = 0.0
for i in range(0,len(Tinf)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	Tm_out, of_out = m.model(np.reshape(beta,(n,1)), Tt_avg[i,:],Tinf[i],stop_resid)
	#Tm, of = m.model(beta,Tt_avg,Tinf,resid_stop)
	sys.stdout.write('Tm_out = ' + str(Tm) +'\n')
	of += of_out
	Tm[i,:] += np.reshape(Tm_out,(n,))
Tm_baseline = Tm#sys.stdout.write('Objective Function for Model = ' + str(of) + '\n')

#Read in training data
#f = open("Classic_Training_Data/"+str(Tinf)+"_n"+str(n)+".dat","w+")
#for i in range(0,n) :
#	f.write("%f %f %f\n" % (Tinf, Tm_best[i], beta_best[i]))

#for i in range(0,len(Tinf)) :
#    f = open("Classic_Training_Data/"+str(Tinf[i])+"_n"+str(n)+".dat","w+")
#    temp = 0
#    for j in range(0,n) :
#        f.read("%f %f %f\n" % (temp, Tm[i,j], beta_best[i,j]))
#    f.close()

#https://stackoverflow.com/questions/6583573/how-to-read-numbers-from-file-in-python
train_data = np.zeros((n*len(Tinf),3))
li = 0
for j in range(0,len(Tinf)) :
    start = 1
    with open("Classic_Training_Data/"+str(Tinf[j])+"_n"+str(n)+".dat") as f:
        #w, h = [np.float(x) for x in next(f).split()] # read first line
        sys.stdout.write('train_data='+str(train_data)+'\n')
        if start == 1 :
            train_data[li,:] = [np.float(x) for x in next(f).split()]
            li+=1
            start = 0
        for line in f: # read rest of lines
            train_data[li,:] = [np.float(x) for x in line.split()]
            li+=1
X = train_data[:,0:2:1]
sys.stdout.write('X = '+str(X) +'\n')
#y = np.zeros((n*len(Tinf),1))
y = train_data[:,2]
y = np.reshape(y,(n*len(Tinf),1))
sys.stdout.write('y = '+str(y) +'\n')

sys.stdout.write('Training Data:\n'+str(train_data))

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

print("Total Dataset Samples: " + str(y.size))
print("Train Dataset Samples: " + str(y_train.size) + " = " + str(y_train.size*100.0/y.size) + " %")
print("Test  Dataset Samples: " + str(y_test.size) + " = " + str(y_test.size*100.0/y.size) + " %")

print('Standardizing Inputs...')

import sklearn.preprocessing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_std = sc.transform(X)

from sklearn.neural_network import MLPRegressor
print("Using Multi Layer Perceptron Regression")
y_train = np.reshape(y_train,-1)
y_test = np.reshape(y_test,-1)
y = np.reshape(y,-1)
mod = MLPRegressor(hidden_layer_sizes=(20,),activation='tanh',solver='lbfgs',verbose=False,learning_rate_init=0.001, learning_rate='constant',alpha=0.000,max_iter = 200000,tol=1e-15)
mod.fit(X_train_std,y_train)

y_train_pred = mod.predict(X_train_std)
y_test_pred = mod.predict(X_test_std)

sys.stdout.write('X_train_std = '+str(X_train_std)+'\n')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('Baseline (All Features)')
print('MSE train: %.10f, test: %.10f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.10f, test: %.10f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

#hist = np.zeros((1,2))
#evaluation = 0
#best = 0
#of_best = of
#beta_best = np.zeros((n,1))
#Tm_best = Tm

import model_aug as ma
T_aug = np.zeros((len(Tinf),n))
T_holdout_aug = np.zeros((len(Tinf),n))
OF = np.zeros((len(Tinf)),)
beta_train = np.zeros((len(Tinf),n))
for j in range(0,len(Tinf)) :
	Temp, OF[j], beta_temp = ma.model(n,mod,sc,Tt_avg[:,j],Tinf[j],stop_resid)
	T_aug[j,:] = np.reshape(Temp,(n,))
	beta_train[j,:] += np.reshape(beta_temp,(n,))
#sys.stdout.write('Augmented model result: '+str(T_aug)+'\n')
of = np.sum(OF)
sys.stdout.write('Augmented model composite objective function '+str(of)+'\n')

Tm_holdout = np.zeros((len(Tinf_holdout),n))
Tm_holdout_baseline = np.zeros((len(Tinf_holdout),n))
beta_holdout = np.zeros((len(Tinf_holdout),n))

sys.stdout.write('Optimization done! Running holdout cases\n')

Tt_avg_holdout = np.zeros((len(Tinf_holdout),n))
Tinf_var = zt*40.0+20.0
for j in range(0,len(Tinf_holdout)) :
	for i in range(0,num_samp) :
		if Tinf_holdout[j] > 0.0 :
			output = t.truth(Tinf_holdout[j],n,stop_resid)
		else :
			output = t.truth(Tinf_var,n,stop_resid)
		#sys.stdout.write(str(output)+'\n')
		Tt_avg_holdout[j,:] += np.reshape(output,(n,))
Tt_avg_holdout /= np.float(num_samp)

for i in range(0,len(Tinf_holdout)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	if Tinf_holdout[i] > 0.0 :
		Tm_out, of_out = m.model(np.reshape(beta,(n,1))*1.0, Tt_avg_holdout[i,:],Tinf_holdout[i],stop_resid)
	else :
		Tm_out, of_out = m.model(np.reshape(beta,(n,1))*1.0, Tt_avg_holdout[i,:],Tinf_var,stop_resid)
	Tm_holdout_baseline[i,:] += np.reshape(Tm_out,(n,))

of = 0.0
for i in range(0,len(Tinf_holdout)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	if Tinf_holdout[i] > 0.0 :
		Tm_out, of_out, beta_temp = ma.model(n,mod,sc,Tt_avg_holdout[i,:], Tinf_holdout[i],stop_resid)
	else :
		Tm_out, of_out, beta_temp = ma.model(n,mod,sc,Tt_avg_holdout[i,:], Tinf_var,stop_resid)
	Tm_holdout[i,:] += np.reshape(Tm_out,(n,))
	beta_holdout[i,:] += np.reshape(beta_temp,(n,))



font = {'size'   : 14}
plt.rc('font', **font)

plt.figure(1)
plt.subplot(1,2,1)
for i in range(0,len(Tinf)):
	plt.plot(zt, Tt_avg[i,:],'-',label=r'Truth, $T_\infty='+str(Tinf[i])+'$')
	plt.plot(zt,Tm_baseline[i,:],'-.',label=r'Model, $T_\infty='+str(Tinf[i])+'$')
	plt.plot(zt,T_aug[i,:],':+',label=r'Augmented, $T_\infty='+str(Tinf[i])+'$')
plt.legend(fontsize=8)
plt.xlabel('z',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()


plt.subplot(1,2,2)
for i in range(0,len(Tinf)) :
	plt.plot(zt,Tt_avg[i,:]-Tm_baseline[i,:],label=r'Model, $T_\infty='+str(Tinf[i])+'$')
for i in range(0,len(Tinf)) :
	plt.plot(zt,Tt_avg[i,:]-T_aug[i,:],':',label=r'Augmented, $T_\infty='+str(Tinf[i])+'$')
plt.xlabel('z',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.legend(fontsize=8)
plt.grid()

#plt.subplot(2,2,3)
plt.figure(2)
for i in range(0,len(Tinf)) :
	plt.plot(zt[1:n-1],beta_train[i,1:n-1],label=r'$\beta$, $T_\infty='+str(Tinf[i])+'$')
plt.ylabel(r'$\beta$',fontsize=14)
plt.xlabel('z',fontsize=18)
plt.legend(fontsize=8)
plt.grid()

#plt.subplot(2,2,4)
plt.figure(3)
plt.plot(y_train,y_train-y_train_pred,'o',label='Train Data')
plt.plot(y_test,y_test-y_test_pred,'r+',label='Test Data')
plt.xlabel(r'$\beta$',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.legend()
plt.grid()

plt.suptitle(r'Training Conditions: $T_{\infty} $ = '+str(Tinf)+', n = ' +str(n), fontsize=18)

#plt.show()

#HOLDOUT PLOTS

plt.figure(4)
plt.subplot(1,2,1)
for i in range(0,len(Tinf_holdout)):
	if Tinf_holdout[i] > 0.0 :
		plt.plot(zt, Tt_avg_holdout[i,:],'-',label=r'Truth, $T_\infty='+str(Tinf_holdout[i])+'$')
		plt.plot(zt,Tm_holdout_baseline[i,:],'-.',label=r'Model, $T_\infty='+str(Tinf_holdout[i])+'$')
		plt.plot(zt,Tm_holdout[i,:],':+',label=r'Augmented, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(zt, Tt_avg_holdout[i,:],'-',label=r'Truth, $T_\infty=40z+20$')
		plt.plot(zt,Tm_holdout_baseline[i,:],'-.',label=r'Model, $T_\infty=40z+20$')
		plt.plot(zt,Tm_holdout[i,:],':+',label=r'Augmented, $T_\infty=40z+20$')
plt.legend(fontsize=8)
plt.xlabel('z',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()


plt.subplot(1,2,2)
for i in range(0,len(Tinf_holdout)) :
	if Tinf_holdout[i] > 0.0 :
		plt.plot(zt,Tt_avg_holdout[i,:]-Tm_holdout_baseline[i,:],label=r'Model, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(zt,Tt_avg_holdout[i,:]-Tm_holdout_baseline[i,:],label=r'Model, $T_\infty=40z+20$')
for i in range(0,len(Tinf_holdout)) :
	if Tinf_holdout[i] > 0.0 :
		plt.plot(zt,Tt_avg_holdout[i,:]-Tm_holdout[i,:],':',label=r'Augmented, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(zt,Tt_avg_holdout[i,:]-Tm_holdout[i,:],':',label=r'Augmented, $T_\infty=40z+20$')
plt.xlabel('z',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.legend(fontsize=8)
plt.grid()

#plt.subplot(2,2,3)
plt.figure(5)
for i in range(0,len(Tinf_holdout)) :
	if Tinf_holdout[i] > 0.0 :
		plt.plot(zt[1:n-1],beta_holdout[i,1:n-1],label=r'$\beta$, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(zt[1:n-1],beta_holdout[i,1:n-1],label=r'$\beta$, $T_\infty=40z+20$')
plt.ylabel(r'$\beta$',fontsize=14)
plt.xlabel('z',fontsize=18)
plt.legend(fontsize=8)
plt.grid()

#plt.suptitle(r'Holdout Conditions', fontsize=18)
#plt.show()

#Plot in feature space
plt.figure(6)
#plt.subplot(2,2,1)
for i in range(0,len(Tinf)):
	plt.plot(np.ones((n,))*Tinf[i],T_aug[i,:],'o',markersize=8,label=r'Training, $T_\infty='+str(Tinf[i])+'$')
for i in range(0,len(Tinf_holdout)):
	if Tinf_holdout[i] > 0.0 :
		plt.plot(np.ones((n,))*Tinf_holdout[i],Tm_holdout[i,:],'+',markersize=8,label=r'Holdout, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(Tinf_var,Tm_holdout[i,:],'+',markersize=8,label=r'Holdout, $T_\infty=40z+20$')
plt.legend(fontsize=8)
plt.xlabel(r'$T_\infty$',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()
#plt.suptitle('Feature Space',fontsize=14)
plt.show()