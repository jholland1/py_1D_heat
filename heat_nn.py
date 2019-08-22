import numpy as np
import sys
import truth as t
import model_nn as m
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import check_grad
import random
#from scipy import optimize


sci_opt = 1

#500 for stochastic, 150 for BFGS
max_iter = 4000
report_of = 100

n = 20
Tinf = np.array([20.0,30.0,40.0,50.0])
Tinf_holdout = ([25.0,35.0,45.0,55.0,-1])
# Tinf = ([50.0])
# Tinf_holdout = ([45.0, 55.0])

#Tinf = np.array([50.0])
#Tinf_holdout = ([45.0,55.0,-1])
#Tinf = np.array([20.0])
stop_resid = 10**-12.0

num_opt_sims = 0
num_opt_adjoint_sims = 0

#Set up neural network
inputlayersize = 2
hiddenlayersize = 5
nLayers = 1
#syn0 = 2*np.rand(inputlayersize+1,hiddenlayersize)-1.0
#syn1 = 2*np.rand(hiddenlayersize,1)-1.0

if nLayers == 1 : beta_length = inputlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize+1
else : beta_length = inputlayersize*hiddenlayersize+hiddenlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize*2+1
# weights = (2*np.random.rand(beta_length,)-1.0)/10.0

weights = np.zeros(beta_length)
for i in range(0,inputlayersize*hiddenlayersize) :
	weights[i] = np.random.randn()*np.sqrt(1.0/inputlayersize)
for i in range(inputlayersize*hiddenlayersize,beta_length) :
	weights[i] = np.random.randn()*np.sqrt(1.0/hiddenlayersize)

#zt = range(0,n)
zt = np.linspace(0,1,n)
Tt_avg = np.zeros((len(Tinf),n))
Tm = np.zeros((len(Tinf),n))
beta = np.ones((n,))
Ttemp = np.zeros((n,))
num_samp = 100
for j in range(0,len(Tinf)) :
	for i in range(0,num_samp) :
		output = t.truth(Tinf[j],n,stop_resid)
		#sys.stdout.write(str(output)+'\n')
		Tt_avg[j,:] += np.reshape(output,(n,))
Tt_avg /= np.float(num_samp)
	

import model_nn as mnn
#Tm_baseline, of = m.model(beta,Tt_avg,Tinf,10.0**-12.0)
of = 0.0
for i in range(0,len(Tinf)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	Tm_out, of_out, beta_temp = m.model(weights*0.0, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
	of += of_out
	Tm[i,:] += np.reshape(Tm_out,(n,))
Tm_baseline = Tm
sys.stdout.write('Objective Function for Model = ' + str(of) + '\n')

of = 0.0
Tm = np.zeros((len(Tinf),n))
for i in range(0,len(Tinf)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	Tm_out, of_out, beta_temp = m.model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
	of += of_out
	Tm[i,:] += np.reshape(Tm_out,(n,))
sys.stdout.write('Objective Function for Initial Weights = ' + str(of) + '\n')




#grad = m.adj_model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,Tm[i,:])
#sys.stdout.write('Adjoint Gradient  = ' + str(grad) + '\n')

hist = np.zeros((1,2))
evaluation = 0
best = 0
of_best = of
weights_best = np.zeros((beta_length,1))
beta_best = np.zeros((len(Tinf),n))
Tm_best = Tm
current_case = 0
hist_count = 0
def model_wrap(weights, Tt_avg, Tinf, resid_stop = 10.0**-12.0,inputlayersize=0,hiddenlayersize=0,nLayers = 1, extra = None) :
	global evaluation
	global hist_count
	global hist
	global beta_best
	global Tm_best
	global Tm
	global best
	global of_best
	global beta_length
	global weights_best
	global n
	global current_case
	global num_opt_sims
	weights = np.reshape(weights,(beta_length,1))
	#weights = weights
	#sys.stdout.write(str(weights)+'\n')
	
	of = 0.0
	Tm = np.zeros((len(Tinf),n))
	beta = np.zeros((len(Tinf),n))
	if sci_opt == 1 or evaluation == 0 or np.mod(evaluation,report_of) == 0 :
		hist_count += 1
		current_case = random.randint(0,len(Tinf)-1)
		for i in range(0,len(Tinf)) :
			Tm_out, of_out, beta_temp = m.model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
			Tm[i,:] += np.reshape(Tm_out,(n,))
			beta[i,:] += np.reshape(beta_temp,(n,))
			of += of_out
			num_opt_sims += 1
	
		#Tm, of, beta = m.model(weights,Tt_avg,Tinf,resid_stop,inputlayersize,hiddenlayersize)
		if evaluation == 0 :
			hist[0][1] = of
		else :
			hist = np.append(hist,np.array([[evaluation, of]]),axis=0)
		if of < of_best :
			of_best = of
			Tm_best = Tm
			weights_best = np.reshape(weights,(beta_length,1))
			beta_best = beta
		sys.stdout.write('Iteration #'+str(evaluation)+' Obj Fun = ' + str(of)+'\n')
	else : #Only run direct case for a single case
		current_case = random.randint(0,len(Tinf)-1)
		i = current_case
		Tm_out, of_out, beta_temp = m.model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
		Tm[i,:] = np.reshape(Tm_out,(n,))
		beta[i,:] += np.reshape(beta_temp,(n,))
		sys.stdout.write('Direct Solution Done for Tinf = ' + str(Tinf[i]) + '\n')
		num_opt_sims += 1
		#Tm, of, beta = m.model(weights,Tt_avg,Tinf,resid_stop,inputlayersize,hiddenlayersize)

	evaluation += 1
	
	#sys.stdout.write('Tm in Direct solve is: '+str(Tm)+'\n')
	return of

def model_wrap_fd(weights, Tt_avg, Tinf, resid_stop = 10.0**-12.0,inputlayersize=0,hiddenlayersize=0,nlayers = 1, extra = None) :
	global evaluation
	global hist
	global beta_best
	global Tm_best
	#global Tm
	global best
	global of_best
	global beta_length
	global weights_best
	global current_case
	
	weights = np.reshape(weights,(beta_length,1))
	weights = weights
	#sys.stdout.write(str(weights)+'\n')
	Tmt, of, beta = m.model(weights,Tt_avg,Tinf,resid_stop,inputlayersize,hiddenlayersize,nLayers)
	#sys.stdout.write('Iteration #'+str(evaluation)+' Obj Fun = ' + str(of)+'\n')
	of = of
	return of
	
	
def grad_wrap(weights, Tt_avg, Tinf, resid_stop =stop_resid,inputlayersize=0,hiddenlayersize=0, nLayers = 1 ,extra = None) :
	global beta_length
	global Tm
	global sci_opt
	global current_case
	global num_opt_adjoint_sims
	#grad = opt.approx_fprime(weights,model_wrap_fd,0.000001,Tt_avg,Tinf,resid_stop,inputlayersize,hiddenlayersize)



	#Tm = np.zeros((len(Tinf),n))
	#for i in range(0,len(Tinf)) :
	#	Tm_out, of_out, beta_temp = m.model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
	#	Tm[i,:] += np.reshape(Tm_out,(n,))
	#	#beta[i,:] += np.reshape(beta_temp,(n,))
	#	#of += of_out
	


	grad = np.zeros((np.size(weights,)))
	#Stochastic Gradient
	if sci_opt == 0 :
		#i = random.randint(0,len(Tinf)-1)
		i = current_case
		grad += m.adj_model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers,Tm[i,:])
		num_opt_adjoint_sims += 1
		#grad += opt.approx_fprime(weights,model_wrap_fd,0.000001,Tt_avg[i,:],Tinf[i],resid_stop,inputlayersize,hiddenlayersize,nLayers,Tm[i,:])
	else :
#		Full Gradient
		for i in range(0,len(Tinf)) :
			grad += m.adj_model(weights, Tt_avg[i,:],Tinf[i],stop_resid,inputlayersize,hiddenlayersize,nLayers,Tm[i,:])
			num_opt_adjoint_sims += 1
			#grad += opt.approx_fprime(weights,model_wrap_fd,0.000001,Tt_avg[i,:],Tinf[i],resid_stop,inputlayersize,hiddenlayersize,nLayers,Tm[i,:])
		
	#sys.stdout.write('Adjoint Gradient = ' + str(grad) +'\n')
	#sys.stdout.write('Tm in Adjoint solve is: '+str(Tm)+'\n')
	return grad


xb_low = [-10.0]*beta_length
xb_high = [10.0]*beta_length
bound = zip(xb_low,xb_high)

#sysl_wrap,grad_wrap,np.reshape(weights,(beta_length,)),Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers,Tm)
#sys.stdout.write('Error in Gradient Computation vs FD is: '+str(err)+'\n').stdout.write('Checking Gradient Using check_grad()\n')
##err = check_grad(mode

if sci_opt :
	#ONLY DO THIS FOR BIAS NODES?
	sys.stdout.write('Running a few steepest descent iters to converge bias weights\n')
	steep_iters = 0
	alpha = 0.0001
	#alpha = 0.001 if not running Tinf = 20.0
	weights = np.reshape(weights,(beta_length,))
	weights_old = weights
	of = 9999999999.9
	iteration = 0
	for i in range(0,steep_iters) :
		of_old = of #NEED TO UPDATE Tm FOR THIS OR YOU WILL GET THE WRONG GRADIENT
		of = model_wrap(weights,Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers)
		grad = grad_wrap(weights,Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers)
		weights_old = weights
		weights = weights-grad*alpha
		iteration +=1
	sys.stdout.write('Starting SciPy Optimization\n')
	res = opt.minimize(model_wrap, np.reshape(weights,(beta_length,)), args=(Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers), method='BFGS', jac=grad_wrap,hess=None,hessp=None,bounds=bound,constraints=(), tol=None, callback=None,options={'gtol': 1.0E-20,'maxiter':2000})
	weights_opt = res.x
	sys.stdout.write('Optimal Weights: ' + str(weights_opt) +'\n')
else :
	################################################################################################
	alpha = 0.0001
	min_alpha = 0.1
	weights = np.reshape(weights,(beta_length,))
	weights_old = weights
	of = 9999999999.9
	iteration = 0
	for i in range(0,max_iter) :
		of_old = of
		of = model_wrap(weights,Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers)
		#if iteration == 10 : alpha = alpha*1.5
	#	if of > of_old and alpha > min_alpha :
	#		alpha /= 2.0
	#		weights = weights_old
	#		of = of_old
	#	else :
	#		grad = grad_wrap(weights,Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,Tm)
		grad = grad_wrap(weights,Tt_avg, Tinf, stop_resid,inputlayersize,hiddenlayersize,nLayers)
		weights_old = weights
		weights = weights-grad*alpha
		iteration +=1
		#sys.stdout.write(str(weights)+'\n')
	weights_opt = weights
	################################################################################################

	sys.stdout.write('Optimal Weights: ' + str(weights_opt) +'\n')

sys.stdout.write('Number of Direct Simulations in Optimization: ' + str(num_opt_sims) + '\n')
sys.stdout.write('Number of Adjoint Simulations in Optimization: ' + str(num_opt_adjoint_sims) + '\n')
sys.stdout.write('Total Number of Simulations in Optimization: ' + str(num_opt_sims+num_opt_adjoint_sims) + '\n')

Tm_holdout = np.zeros((len(Tinf_holdout),n))
Tm_holdout_baseline = np.zeros((len(Tinf_holdout),n))
beta_holdout = np.zeros((len(Tinf_holdout),n))

sys.stdout.write('Optimization done! Running holdout cases\n')

Tinf_var = zt*40.0+20.0
Tt_avg_holdout = np.zeros((len(Tinf_holdout),n))
for j in range(0,len(Tinf_holdout)) :
	for i in range(0,num_samp) :
		if Tinf_holdout[j] > 0.0 :
			output = t.truth(Tinf_holdout[j],n,stop_resid)
		else : #Run the variable Tinf case
			output = t.truth(Tinf_var,n,stop_resid)
		#sys.stdout.write(str(output)+'\n')
		Tt_avg_holdout[j,:] += np.reshape(output,(n,))
Tt_avg_holdout /= np.float(num_samp)

for i in range(0,len(Tinf_holdout)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	if Tinf_holdout[i] > 0.0 :
		Tm_out, of_out, beta_temp = m.model(weights*0.0, Tt_avg_holdout[i,:],Tinf_holdout[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
	else:
		Tm_out, of_out, beta_temp = m.model(weights*0.0, Tt_avg_holdout[i,:],Tinf_var,stop_resid,inputlayersize,hiddenlayersize,nLayers)
	Tm_holdout_baseline[i,:] += np.reshape(Tm_out,(n,))

of = 0.0
for i in range(0,len(Tinf_holdout)) :
	#sys.stdout.write(str(Tt_avg[i,:])+'\n')
	if Tinf_holdout[i] > 0.0 :
		Tm_out, of_out, beta_temp = m.model(weights_opt, Tt_avg_holdout[i,:],Tinf_holdout[i],stop_resid,inputlayersize,hiddenlayersize,nLayers)
	else :
		Tm_out, of_out, beta_temp = m.model(weights_opt, Tt_avg_holdout[i,:],Tinf_var,stop_resid,inputlayersize,hiddenlayersize,nLayers)
	Tm_holdout[i,:] += np.reshape(Tm_out,(n,))
	beta_holdout[i,:] += np.reshape(beta_temp,(n,))



#import shelve

#filename='/home/jonholland07/SD_Card/py_1D_Heat/heat_nn.out'
#my_shelf = shelve.open(filename,'n') # 'n' for new

#for key in dir():
#    try:
#        my_shelf[key] = globals()[key]
#    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
#        print('ERROR shelving: {0}'.format(key))
#my_shelf.close()

#https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session
#To recover this data:
#my_shelf = shelve.open(filename)
#for key in my_shelf:
#    globals()[key]=my_shelf[key]
#my_shelf.close()

import pickle
# Saving the objects:
with open('heat_nn.pkl', 'w') as f:  # Python 3: open(..., 'wb')
	    pickle.dump([hist,Tinf], f)
	
# Getting back the objects:
#with open('plotting_logs/heat.pkl') as f:  # Python 3: open(..., 'rb')
#	hist,Tinf = pickle.load(f)

#font = {'size'   : 14}
#plt.rc('font', **font)
eps = 5.0*10.0**-4.0

beta_truth = 1.0/eps*(1.0+5.0*np.sin(3.0*np.pi/200.0*Tt_avg[0,:])+np.exp(0.02*Tt_avg[0,:]))*10.0**-4.0+0.5/eps*(Tinf[0]-Tt_avg[0,:])/(Tinf[0]**4.0-Tt_avg[0,:]**4.0)


plt.figure(1)
plt.subplot(1,2,1)
for i in range(0,len(Tinf)):
	plt.plot(zt, Tt_avg[i,:],'-',label=r'Truth, $T_\infty='+str(Tinf[i])+'$')
	plt.plot(zt,Tm_baseline[i,:],'-.',label=r'Model, $T_\infty='+str(Tinf[i])+'$')
	plt.plot(zt,Tm_best[i,:],':+',label=r'Augmented, $T_\infty='+str(Tinf[i])+'$')
plt.legend(loc='best')
plt.xlabel('z',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()


plt.subplot(1,2,2)
for i in range(0,len(Tinf)) :
	plt.plot(zt,Tt_avg[i,:]-Tm_baseline[i,:],label=r'Model, $T_\infty='+str(Tinf[i])+'$')
for i in range(0,len(Tinf)) :
	plt.plot(zt,Tt_avg[i,:]-Tm_best[i,:],':',label=r'Augmented, $T_\infty='+str(Tinf[i])+'$')
plt.xlabel('z',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.legend(loc='best')
plt.grid()

plt.figure(2)
#plt.subplot(1,2,1)
for i in range(0,len(Tinf)) :
	plt.plot(zt,beta_best[i,:],'-+',label=r'$\beta$, $T_\infty='+str(Tinf[i])+'$')
if len(Tinf) == 1 : plt.plot(zt,beta_truth,label=r'$\beta_{truth}$')
plt.ylabel(r'$\beta$',fontsize=14)
plt.xlabel('z',fontsize=18)
plt.legend(loc='best')
plt.grid()

#plt.subplot(2,2,4)
plt.figure(3)
plt.semilogy(hist[:,0],hist[:,1],label='OF')
plt.xlabel('Evaluation',fontsize=14)
plt.ylabel('Objective Function',fontsize=14)
plt.grid()

#plt.suptitle(r'Training Conditions: $T_{\infty} $ = '+str(Tinf)+', n = ' +str(n), fontsize=18)

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
plt.legend(loc='best')
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
plt.legend(loc='best')
plt.grid()

plt.figure(5)
#plt.subplot(2,2,3)
for i in range(0,len(Tinf_holdout)) :
	if Tinf_holdout[i] > 0.0 :
		plt.plot(zt,beta_holdout[i,:],label=r'$\beta$, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(zt,beta_holdout[i,:],label=r'$\beta$, $T_\infty=40z+20$')
plt.ylabel(r'$\beta$',fontsize=14)
plt.xlabel('z',fontsize=14)
plt.legend(loc='best')
plt.grid()

#plt.suptitle(r'Holdout Conditions', fontsize=18)
#plt.show()

#Plot in feature space
plt.figure(6)
#plt.subplot(2,2,1)
for i in range(0,len(Tinf)):
	plt.plot(np.ones((n,))*Tinf[i],Tm_best[i,:],'o',markersize=8,label=r'Training, $T_\infty='+str(Tinf[i])+'$')
for i in range(0,len(Tinf_holdout)):
	if Tinf_holdout[i] > 0.0 :
		plt.plot(np.ones((n,))*Tinf_holdout[i],Tm_holdout[i,:],'+',markersize=8,label=r'Holdout, $T_\infty='+str(Tinf_holdout[i])+'$')
	else :
		plt.plot(Tinf_var,Tm_holdout[i,:],'+',markersize=8,label=r'Holdout, $T_\infty=40z+20$')
plt.legend(loc='best')
plt.xlabel(r'$T_\infty$',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()
plt.suptitle('Feature Space',fontsize=14)
plt.show()

