import numpy as np
import sys
import sklearn

def model(n,mod, sc, Tt = 0.0, Tinf = 50, stop_resid = 10**-12, extra = None) :
	resid = 1
	i = 1
	#n = np.size(Tt)
	T = np.zeros((n,1))
	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.30*dz**2.0
	
	Tinf = np.asarray(Tinf)
	if np.size(Tinf) > 2 :
		Tinf = np.reshape(Tinf,(n,1))
	if np.size(Tinf) > 2 :
		Tinf = np.reshape(Tinf,(n,1))
		
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
	features = np.zeros((n,2))
	def next_iter(T, A, Tinf, dz, dt, sc, mod) :
		features = np.zeros((n,2))
		if np.size(Tinf) > 2 :
			for i in range(0,n) : features[i,0] = Tinf[i]
		else : features[:,0] = Tinf
		features[:,1] = np.transpose(T)
		#sys.stdout.write('Features: ' + str(features) +'\n')
		features_std = sc.transform(features)
		#sys.stdout.write('Features Scaled: ' + str(features_std) +'\n')
		beta = mod.predict(features_std)
		beta = np.reshape(beta,(n,1))
		#sys.stdout.write('Beta: ' + str(beta) +'\n')
		g = -eps*beta*(Tinf**4.0-T**4.0)
		dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
		dT[0] = 0.0
		dT[n-1] = 0.0
		T = T+dT
		resid =np.sum(np.sqrt(dT*dT))
		OF = 0.5*np.sum((Tt-T)**2.0)
		return T, resid, OF, beta
    	
	while resid > stop_resid :
		j = j+1
		T, resid, OF, beta = next_iter(T, A, Tinf, dz, dt, sc, mod)
    	
	return T, OF, beta