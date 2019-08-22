import numpy as np
import sys

def truth(Tinf=50.0, n = 30, stop_resid = 10**-12) :
	resid = 1
	i = 1
	T = np.zeros((n,1))
	Tinf = np.asarray(Tinf)
	if np.size(Tinf) > 2 :
		Tinf = np.reshape(Tinf,(n,1))
	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.25*dz**2.0
	random_var = 0.1*np.random.rand(n,1);
	
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
    #Iterate Until Residual Drops Enough
	while resid > stop_resid :
		eps = (1.0+5.0*np.sin(3.0*np.pi/200.0*T)+np.exp(0.02*T)+random_var)*10.0**-4.0
		g = -eps*(Tinf**4.0-T**4.0)+h*(T-Tinf);
		dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
		dT[0] = 0.0
		dT[n-1] = 0.0
		
		T = T+dT
		resid =np.sum(np.sqrt(dT*dT))
		j = j+1
		#sys.stdout.write(str(j)+' Resid: ' + str(resid)+ '\n')
	return T
	
#T = truth(50,50)
#print T
	
	
