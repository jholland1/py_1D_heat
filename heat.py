import numpy as np
import sys
import truth as t
import model as m
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import check_grad

n = 20
Tinf = 50.0

#zt = range(0,n)
zt = np.linspace(0,1,n)
Tt_avg = np.zeros((n,1))
beta = np.ones((n,1))

for i in range(0,100) :
	Tt_avg += t.truth(Tinf,n,10.0**-10.0)

Tt_avg /= 100.0

Tm, of = m.model(beta,Tt_avg,Tinf,10.0**-12.0)
Tm_base = Tm
sys.stdout.write('Objective Function = ' + str(of) + '\n')

hist = np.zeros((1,2))
evaluation = 0
best = 0
of_best = of
beta_best = np.zeros((n,1))
Tm_best = Tm
def model_wrap(beta, Tt_avg, Tinf, resid_stop = 10.0**-12.0, extra = None) :
	global evaluation
	global hist
	global beta_best
	global Tm_best
	global best
	global of_best
	global Tm
	beta = np.reshape(beta,(n,1))
	Tm, of = m.model(beta,Tt_avg,Tinf,resid_stop)
	if evaluation == 0 :
		hist[0][1] = of
	else :
		hist = np.append(hist,np.array([[evaluation, of]]),axis=0)
	if of < of_best :
		of_best = of
		Tm_best = np.reshape(Tm,(n,1))
		beta_best = np.reshape(beta,(n,1))
	evaluation += 1
	sys.stdout.write('Iteration #'+str(evaluation)+' Obj Fun = ' + str(of)+'\n')
	return of
	
def adj_wrap(beta, Tt_avg, Tinf, resid_stop = 10.0**-12.0 ,extra = None) :
	global Tm
	beta = np.reshape(beta,(n,1))
	Tm, of = m.model(beta,Tt_avg,Tinf,resid_stop)
	grad = m.adj_model(beta,Tt_avg,Tinf,resid_stop,Tm)
	return grad

#def adjoint_wrap(beta) :

#sys.stdout.write('Checking Gradient Using check_grad()\n')
#err = check_grad(model_wrap,adj_wrap,np.reshape(beta,(n,)),Tt_avg, Tinf, 10.0**-12.0,Tm)
#sys.stdout.write('Error in Gradient Computation vs FD is: '+str(err)+'\n')


res = opt.minimize(model_wrap, np.reshape(beta,(n,)), args=(Tt_avg, Tinf, 10.0**-10.0,Tm), method='BFGS', jac=adj_wrap,hess=None,hessp=None,bounds=None,constraints=(), tol=None, callback=None,options={'gtol': 1.0E-20,'maxiter':2000})

beta_inv = res.x

font = {'size'   : 14}
plt.rc('font', **font)

#Write out variables for training data
#f = open(data_dir+"beta_fiml.dat","w+")
#for curr_beta in y_tec_mod :
#    write_beta = np.float(curr_beta)-1.0
#    f.write("%f\n" % write_beta)
#f.close()

f = open("Classic_Training_Data/"+str(Tinf)+"_n"+str(n)+".dat","w+")
for i in range(0,n) :
	f.write("%f %f %f\n" % (Tinf, Tm_best[i], beta_best[i]))

f.close()

import pickle
# Saving the objects:
with open('heat.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([hist,Tinf], f)

eps = 5.0*10.0**-4.0
#beta_truth = 1.0/eps*(1+5.0*np.sin(3.0*np.pi/200.0*Tm_base)+np.exp(0.02*Tm_base))*10**-4.0+0.5/eps*(Tinf-Tm_base)/(Tinf**4.0-Tm_base**4.0)
beta_truth = 1.0/eps*(1.0+5.0*np.sin(3.0*np.pi/200.0*Tt_avg)+np.exp(0.02*Tt_avg))*10.0**-4.0+0.5/eps*(Tinf-Tt_avg)/(Tinf**4.0-Tt_avg**4.0)


plt.figure(1)
plt.subplot(1,2,1)
plt.plot(zt, Tt_avg,'-+',label='Truth')
plt.plot(zt,Tm_base,label='Model')
plt.plot(zt,Tm_best,label='Inverse')
plt.legend(loc='best')
plt.xlabel('z',fontsize=14)
plt.ylabel('T',fontsize=14)
plt.grid()


plt.subplot(1,2,2)
plt.plot(zt,Tt_avg-Tm_base,label='Model')
plt.plot(zt,Tt_avg-Tm_best,'--',label='Inverse')
plt.xlabel('z',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.grid()
plt.legend()


plt.figure(2)
#plt.subplot(1,2,1)
plt.plot(zt[1:n-1],beta_best[1:n-1],'-+',label=r'$\beta$')
plt.plot(zt[1:n-1],beta_truth[1:n-1],'-o',label=r'$\beta_{truth}$')
plt.ylabel(r'$\beta$',fontsize=14)
plt.xlabel('z',fontsize=18)
plt.legend()
plt.grid()

plt.figure(3)
#plt.subplot(1,2,1)
plt.semilogy(hist[:,0],hist[:,1],'-+',label='OF')
plt.xlabel('Evaluation',fontsize=14)
plt.ylabel('Objective Function',fontsize=14)
plt.grid()

#plt.suptitle(r'$ T_{\infty} $ = '+str(Tinf)+', n = ' +str(n), fontsize=18)

plt.show()
