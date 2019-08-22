import numpy as np
import sys

def model(beta = 1.0, Tt = 0.0, Tinf = 50, stop_resid = 10**-12, extra = None) :
	resid = 1
	i = 1
	n = np.size(Tt)
	T = np.zeros((n,1))
	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.05*dz**2.0
	
	Tinf = np.asarray(Tinf)
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
	def next_iter(T, A, Tinf, beta, dz, dt) :

		g = -eps*beta*(Tinf**4.0-T**4.0)
		#g = beta*eps*4.0*T**3.0
		dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
		dT[0] = 0.0
		dT[n-1] = 0.0
		T = T+dT
		resid =np.sum(np.sqrt(dT*dT))
		OF = 0.5*np.sum((Tt-T)**2.0)
		
		return T, resid, OF
		
	while resid > stop_resid :
		j = j+1
		T, resid, OF = next_iter(T, A, Tinf,beta, dz, dt)
	return T, OF
	


def adj_model(beta, Ttruth, Tinf = 50, stop_resid = 10**-12, Tmodel = 0.0, extra = None) :
# Compute the adjoint of the 1D heat equation
# Notation:
# f denotes physical equation and its derivatives (fx, etc)
# g denotes objective function and its derivatives
	
	Tmodel = np.array(Tmodel)
	n = np.size(Ttruth)
	beta = np.reshape(beta,(n,1))

	resid = 1
	i = 1
	lambd = np.zeros((n,1))
	g = np.zeros((n,1))

	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.25*dz**2.0

	#Spatial Discretization (A Matrix)
	for i in range(1,n-1) :
		A[i][i-1] = 1.0
		A[i][i] = -2.0
		A[i][i+1] = 1.0
		
	#One-sized 2nd Order on Boundaries
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
    	
    	# Partial derivative of objective function wrt T
	gx = -(Ttruth-Tmodel)
    	
	At = np.transpose(A)
	#At = A
    #Iterate until residual drops enough
	while resid > stop_resid :
    #while j < 2 :
		j = j+1

    	#Partial derivative of RHS of physics wrt Tmodel
    	
    	#LINEARIZED RIGHT HAND SIDE?? RHS Derivative wrt Temperature?
		fx = -beta*eps*4.0*Tmodel**3.0
		#OR JUST RHS??
		#fx = eps*beta*(Tinf**4.0-Tmodel**4.0)
		
		#sys.stdout.write(str(fx)+'\n')
		gx = -(Ttruth-Tmodel)
		dlambda = np.reshape((dt/dz**2.0)*np.dot(At,lambd),(n,1))+dt*gx+dt*fx*lambd
		dlambda[0] = 0.0
		dlambda[n-1] = 0.0
		lambd = lambd + dlambda
		#sys.stdout.write('lambda='+str(lambd)+'\n')
		resid = np.sum(np.sqrt(dlambda**2.0))
		
	grad = -lambd*(-eps*(Tinf**4.0-Tmodel**4.0))
	#grad = -lambd*(-beta*eps*4.0*Tmodel**3.0)
	grad = np.reshape(grad,(n,))
	return grad
    	

