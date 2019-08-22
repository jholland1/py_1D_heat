import numpy as np
import sys
#from ad import adnumber
#from ad.admath import *
#import autograd.numpy as np
#from autograd import grad

def model(Tinf = 50, n = 30, stop_resid = 10**-12,beta=1.0,Tt = 0.0) :
	resid = 1
	i = 1
	T = np.zeros((n,1))
	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.25*dz**2.0
	#random_var = 0.1*np.random.rand(n,1);
	
	for i in range(1,n-1) :
		A[i][i-1] = 1.0
		A[i][i] = -2.0
		A[i][i+1] = 1.0
	A[0][0] = 2.0
    	A[0][1] = -5.0
    	A[0][2] = 4.0
    	A[0][3] = -1.0
	
	A[n-1][n-1] = 2.0
    	A[n-1][n-2] = -5.0
    	A[n-1][n-3] = 4.0
    	A[n-1][n-4] = -1.0
    	j = 1
    	eps = 5.0*10**-4.0
    	OF = 0.0
    	
    	def next_iter(T, A, Tinf, beta, dz, dt) :
    		g = -eps*beta*(Tinf**4.0-T**4.0)
    		dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
    		dT[0] = 0.0
    		dT[n-1] = 0.0
    		T = T+dT
    		resid =np.sum(np.sqrt(dT*dT))
    		sys.stdout.write(str(resid)+'\n')
    		OF = np.sqrt(np.sum(Tt-T)**2.0)
    		
    		return T, resid, OF
    	
    	while resid > stop_resid :
    		j = j+1
    		T, resid, OF = next_iter(T, A, Tinf,beta, dz, dt)
    	
    	return T, OF
    	
    	#Iterate Until Residual Drops Enough
    	#while resid > stop_resid :
    	#	#eps = (1.0+5.0*np.sin(3.0*np.pi/200.0*T)+np.exp(0.02*T)+random_var)*10.0**-4.0
    	#	g = -eps*beta*(Tinf**4.0-T**4.0);
    	#	dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
    	#	#dT[0] = 0.0
    	#	#dT[n-1] = 0.0
    	#	
    	#	T[2:n-1] = T[2:n-1]+dT[2:n-1]
    	#	resid =np.sum(np.sqrt(dT*dT))
    	#	j = j+1
    	#	OF = np.sqrt(np.sum(Tt-T)**2.0)
	#return T, OF

	
#T = truth(50,50)
#print T
	
	
