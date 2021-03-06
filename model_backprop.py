import sys
import numpy as np
import scipy.optimize as opt

#BIAS NODES ARE SCREWING WITH GRADIENT - RECODE WITH SEPARATE BIAS WEIGHTS AND NODES?

def model(beta_t = 1.0, Tt = 0.0, Tinf = 50, stop_resid = 10**-12,inputlayersize=0,hiddenlayersize=0,nLayers=1, is_complex=0, extra = None) :
	alpha = 0.005
	n = np.size(Tt)
	#sys.stdout.write(str(n)+'\n')
	Tt = np.reshape(Tt,(n,1))
	#sys.stdout.write(str(Tt)+'\n')
	
	#sys.stdout.write(str(weights)+'\n')
	#weights = np.reshape(n,0)
	#Set up neural networks
	#beta = np.zeeros((n,1))
	if nLayers == 1 :
		num_weights = inputlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize+1
		#weights = np.reshape(weights,(num_weights,1))
	
		inputlayersize = 3
		syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
		syn1 = 2*np.zeros((hiddenlayersize,1))-1.0
		
		l1_bias_weights = np.zeros((hiddenlayersize,))
		output_bias = 0.0
	
		r = 0
		c = 0
		count = 0
		#sys.stdout.write(str(syn0[2,0])+'\n')
		for r in range(0,inputlayersize) :
			for c in range(0,hiddenlayersize) :
				#sys.stdout.write(str(syn0[r,c])+'\n')
				#sys.stdout.write(str(weights[count])+'\n')
				#sys.stdout.write(str(r)+' '+str(c)+'\n')
				syn0[r,c] = np.random.randn()/100.0
				count+=1
	
		for r in range(0,hiddenlayersize) :
			c = 0
			syn1[r,c] = np.random.randn()/100.0
			count+=1
		
		for r in range(0,hiddenlayersize) :
			l1_bias_weights[r] = np.random.randn()/100.0
			count+=1
		output_bias = np.random.randn()/100.0
		
	else :
		num_weights = inputlayersize*hiddenlayersize+hiddenlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize*2+1
		#weights = np.reshape(weights,(num_weights,1))
	
		inputlayersize = 3
		syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
		syn1 = 2*np.zeros((hiddenlayersize,hiddenlayersize))-1.0
		syn2 = 2*np.zeros((hiddenlayersize,1))-1.0
	
		l1_bias_weights = np.zeros((hiddenlayersize,))
		l2_bias_weights = np.zeros((hiddenlayersize,))
		output_bias = 0.0
		
		r = 0
		c = 0
		count = 0
		#sys.stdout.write(str(syn0[2,0])+'\n')
		for r in range(0,inputlayersize) :
			for c in range(0,hiddenlayersize) :
				#sys.stdout.write(str(syn0[r,c])+'\n')
				#sys.stdout.write(str(weights[count])+'\n')
				#sys.stdout.write(str(r)+' '+str(c)+'\n')
				syn0[r,c] = np.random.randn()/100.0
				count+=1
		
		for r in range(0,hiddenlayersize) :
			for c in range(0,hiddenlayersize) :
				syn1[r,c] = np.random.randn()/100.0
				count+=1		
		for r in range(0,hiddenlayersize) :
			c = 0
			syn2[r,c] = np.random.randn()/100.0
			count+=1
		for r in range(0,hiddenlayersize) :
			l1_bias_weights[r] = np.random.randn()/100.0
			count+=1
		for r in range(0,hiddenlayersize) :
			l2_bias_weights[r] = np.random.randn()/100.0
			count+=1
		output_bias = np.random.randn()/100.0
	
	#Set up heat solver
	resid = 1
	i = 1
	
	T = np.zeros((n,1))
	h = 0.5
	A = np.zeros((n,n))
	z = np.linspace(0,1,n)
	dz = 1.0/(np.float(n)-1.0)
	dt = 0.25*dz**2.0
	
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
	

    	
    #Set up training data
   	#sys.stdout.write('value of n is ' +str(n)+'\n')
   	trainingdata_inputs = np.ones((n,3))
   	for i in range(0,n) :
		#sys.stdout.write('T[i] = ' + str(T[i])+' training_data_inputs[i,1] = ' +str(trainingdata_inputs[i,1])+' Tinf='+str(Tinf)+'\n')
   		trainingdata_inputs[i,1] = Tinf/50.0
		#trainingdata_inputs[0:n:1,1] *= Tinf/50.0
   	nn_cost = 1.0
	
	if is_complex == 1 :
		syn1 = syn1.astype(complex)
		syn0 = syn0.astype(complex)
		if nLayers == 2 : syn2 = syn2.astype(complex)
		A = A.astype(complex)
		T = T.astype(complex)
		trainingdata_inputs = trainingdata_inputs.astype(complex)
	
	while resid > stop_resid and j < 10000 :
		j = j+1
		
   		#sys.stdout.write('Iteration = '+str(j)+' resid = '+str(resid)+' loss is: '+str(nn_cost)+'\n')
   		#Update training data
		meanT = np.mean(T)
		#sys.stdout.write(str(T)+'\n')
		for i in range(0,n) : 
			#sys.stdout.write('T[i] = ' + str(T[i])+' training_data_inputs[i,1] = ' +str(trainingdata_inputs[i,1])+' Tinf='+str(Tinf)+'\n')
			trainingdata_inputs[i,2] = T[i]/Tinf
		#feedforward training data to find betas
		layer0 = trainingdata_inputs
		layer1 = np.dot(layer0,syn0)
		layer1[:,1] = 1.0
#		for i in range(0,n) : 
#			layer1[i,:] += l1_bias_weights
		layer1 = np.tanh(layer1)
		
		if nLayers == 1 : 
			#beta = np.dot(layer1,syn1)+1.0 + output_bias
			beta = np.dot(layer1,syn1)
		else : 
			layer2 = np.dot(layer1,syn1)
			layer2[:,1] = 1.0
#			for i in range(0,n) : 
#				layer2[i,:] += l2_bias_weights
			layer2 = np.tanh(layer2)
			#beta = np.dot(layer2,syn2)+1.0+output_bias
			beta = np.dot(layer2,syn2)
 ##############BACKRPOPAGATION#############################   		
		#sys.stdout.write('Beta = '+str(beta)+'\n')
		if nLayers == 1 : #1 Layer
			layer2_error=(beta-beta_t)
			#layer2_bias_delta = layer2_error
			layer2_delta = layer2_error
			nn_cost = np.mean(abs(layer2_error))
			layer1_error = np.dot(layer2_delta,np.transpose(syn1))
			#layer1_bias_delta=np.dot(layer2_delta,np.transpose(l1_bias_weights))
		else : #Two layers THIS IS NOT CORRECT!!!
			layer2_error=np.dot((beta-beta_t),np.transpose(syn2))
			#layer2_bias_delta = layer2_error
			layer2_delta = layer2_error*(1.0-np.tanh(layer2)**2.0)
			layer2_delta[:,1] = layer2_error[:,1]
			nn_cost = np.mean(abs(beta-beta_t))
			layer1_error = np.dot(layer2_delta,np.transpose(syn1))
			#layer1_bias_delta=layer2_delta*np.transpose(l1_bias_weights)
			
		#sys.stdout.write('Current loss is: '+str(nn_cost)+'\n')		
		layer1_delta = layer1_error*(1.0-np.tanh(layer1)**2.0)
		layer1_delta[:,1] = layer1_error[:,1]
		
		syn1 = syn1 - alpha*np.dot(np.transpose(layer1),layer2_delta);
		syn0 = syn0 - alpha*np.dot(np.transpose(layer0),layer1_delta);
		#l1_bias_weights = l1_bias_weights - alpha*np.dot(np.transpose(l1_bias_weights),layer1_delta)
		#output_bias = output_bias - alpha*(np.sum(layer2_error))
			
		if nLayers == 2 :
			syn2 = syn2 - alpha*np.dot(np.transpose(layer2),(beta-beta_t))
			#l2_bias_weights = l2_bias_weights - alpha*(np.transpose(l2_bias_weights)*layer2_bias_delta)
													  
#############################################			
		
		g = -eps*beta*(Tinf**4.0-T**4.0)
   		dT = (dt/dz**2.0)*np.dot(A,T)-dt*g
   		dT[0] = 0.0
   		dT[n-1] = 0.0
   		T = T+dT
   		resid =np.sum(np.sqrt(dT*dT))
   		OF = 0.5*np.sum((Tt-T)**2.0)
   		#sys.stdout.write('Iter '+str(j)+' resid ' +str(resid)+'\n')
    	
   	return T, OF, beta
    	


#************************************************************************************************************************

#************************************************************************************************************************

#************************************************************************************************************************

#************************************************************************************************************************    	
    	
def adj_model(weights, Ttruth, Tinf = 50, stop_resid = 10**-12,inputlayersize=0,hiddenlayersize=0,nLayers = 1, Tmodel = 0.0, extra = None) :
# Compute the adjoint of the 1D heat equation
# Notation:
# f denotes physical equation and its derivatives (fx, etc)
# g denotes objective function and its derivatives
	n = np.size(Ttruth)
	#sys.stdout.write(str(n)+'\n')
	Ttruth = np.reshape(Ttruth,(n,1))
	#sys.stdout.write(str(Tt)+'\n')
	
	Tmodel = np.array(Tmodel)
	Tmodel = np.reshape(Tmodel,(n,1))
	n = np.size(Ttruth)
	if nLayers == 1 :
		num_weights = inputlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize+1
		weights = np.reshape(weights,(num_weights,1))
	
		inputlayersize = 2
		syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
		syn1 = 2*np.zeros((hiddenlayersize,1))-1.0
		
		l1_bias_weights = np.zeros((hiddenlayersize,))
		output_bias = 0.0
	
		r = 0
		c = 0
		count = 0
		#sys.stdout.write(str(syn0[2,0])+'\n')
		for r in range(0,inputlayersize) :
			for c in range(0,hiddenlayersize) :
				#sys.stdout.write(str(syn0[r,c])+'\n')
				#sys.stdout.write(str(weights[count])+'\n')
				#sys.stdout.write(str(r)+' '+str(c)+'\n')
				syn0[r,c] = weights[count]
				count+=1
	
		for r in range(0,hiddenlayersize) :
			c = 0
			syn1[r,c] = weights[count]
			count+=1
		
		for r in range(0,hiddenlayersize) :
			l1_bias_weights[r] = weights[count]
			count+=1
		output_bias = weights[count]
		
	else :
		num_weights = inputlayersize*hiddenlayersize+hiddenlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize*2+1
		weights = np.reshape(weights,(num_weights,1))
	
		inputlayersize = 2
		syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
		syn1 = 2*np.zeros((hiddenlayersize,hiddenlayersize))-1.0
		syn2 = 2*np.zeros((hiddenlayersize,1))-1.0
	
		l1_bias_weights = np.zeros((hiddenlayersize,))
		l2_bias_weights = np.zeros((hiddenlayersize,))
		output_bias = 0.0
		
		r = 0
		c = 0
		count = 0
		#sys.stdout.write(str(syn0[2,0])+'\n')
		for r in range(0,inputlayersize) :
			for c in range(0,hiddenlayersize) :
				#sys.stdout.write(str(syn0[r,c])+'\n')
				#sys.stdout.write(str(weights[count])+'\n')
				#sys.stdout.write(str(r)+' '+str(c)+'\n')
				syn0[r,c] = weights[count]
				count+=1
		
		for r in range(0,hiddenlayersize) :
			for c in range(0,hiddenlayersize) :
				syn1[r,c] = weights[count]
				count+=1		
		for r in range(0,hiddenlayersize) :
			c = 0
			syn2[r,c] = weights[count]
			count+=1
		for r in range(0,hiddenlayersize) :
			l1_bias_weights[r] = weights[count]
			count+=1
		for r in range(0,hiddenlayersize) :
			l2_bias_weights[r] = weights[count]
			count+=1
		output_bias = weights[count]

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
    	#gx = -(Ttruth-Tmodel)
    	
    	At = np.transpose(A)
	#At = A
	
	#Set up training data
    	#sys.stdout.write('value of n is ' +str(n)+'\n')
    	trainingdata_inputs = np.ones((n,2))
    	for i in range(0,n) :
    		trainingdata_inputs[i,0] = Tinf/50.0
    	#trainingdata_inputs[0:n:1,1] *= Tinf/50.0
	
	#Update training data\
    	meanT = np.mean(Tmodel)
    	
    	for i in range(0,n) :
    		trainingdata_inputs[i,1] = Tmodel[i]/Tinf
   	
    	#feedforward training data to find betas
    	layer0 = trainingdata_inputs
    	layer1 = np.dot(layer0,syn0)
    	for i in range(0,n) : layer1[i,:] += l1_bias_weights
    	layer1 = np.tanh(layer1)
    	if nLayers == 1 : beta = np.dot(layer1,syn1)+1.0 + output_bias
    	else : 
    		layer2 = np.dot(layer1,syn1)
    		for i in range(0,n) : layer2[i,:] += l2_bias_weights
    		layer2 = np.tanh(layer2)
    		beta = np.dot(layer2,syn2)+1.0+output_bias
	
	gx = -(Ttruth-Tmodel)
	
    	#Iterate until residual drops enough
    	while resid > stop_resid :
    	#while j < 2 :
    		j = j+1
	    	#for i in range(0,n) :
	    	#	trainingdata_inputs[i,2] = Tmodel[i]/Tinf
	    	#feedforward training data to find betas
	    	#layer0 = trainingdata_inputs
	    	#layer1 = np.tanh(np.dot(layer0,syn0))
	    	#layer1[:,0] = 1.0 #bias nodes
	    	#beta = np.dot(layer1,syn1)+1.0
	    	
    		#Partial derivative of RHS of physics wrt Tmodel
    		fx = beta*eps*4.0*Tmodel**3.0
    		dlambda = np.reshape((dt/dz**2.0)*np.dot(At,lambd),(n,1))+dt*gx-dt*fx*lambd
    		dlambda[0] = 0.0
    		dlambda[n-1] = 0.0
    		lambd = lambd + dlambda
    		resid = np.sum(np.sqrt(dlambda**2.0))
    	
    	
    	
    	
    	def forward_prop(weights,inputlayersize,hiddenlayersize,trainingdata_inputs,num_weights) :
    		#global inputlayersize
    		#global hiddenlayersize
    		#global weights
    		#weights[iWeight] = weight_value
    		weights = np.reshape(weights,(num_weights,1))
		if nLayers == 1 :
			num_weights = inputlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize+1
			weights = np.reshape(weights,(num_weights,1))
	
			inputlayersize = 2
			syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
			syn1 = 2*np.zeros((hiddenlayersize,1))-1.0
			
			syn0 = syn0.astype(complex)
			syn1 = syn1.astype(complex)
			syn2 = syn0.astype(complex)
		
			l1_bias_weights = np.zeros((hiddenlayersize,))
			l1_bias_weights = l1_bias_weights.astype(complex)
			
			output_bias = 0.0
	
			r = 0
			c = 0
			count = 0
			#sys.stdout.write(str(syn0[2,0])+'\n')
			for r in range(0,inputlayersize) :
				for c in range(0,hiddenlayersize) :
					#sys.stdout.write(str(syn0[r,c])+'\n')
					#sys.stdout.write(str(weights[count])+'\n')
					#sys.stdout.write(str(r)+' '+str(c)+'\n')
					syn0[r,c] = weights[count]
					count+=1
	
			for r in range(0,hiddenlayersize) :
				c = 0
				syn1[r,c] = weights[count]
				count+=1
		
			for r in range(0,hiddenlayersize) :
				l1_bias_weights[r] = weights[count]
				count+=1
			output_bias = weights[count]
		
		else :
			num_weights = inputlayersize*hiddenlayersize+hiddenlayersize*hiddenlayersize+hiddenlayersize+hiddenlayersize*2+1
			weights = np.reshape(weights,(num_weights,1))
	
			inputlayersize = 2
			syn0 = 2*np.zeros((inputlayersize,hiddenlayersize))-1.0
			syn1 = 2*np.zeros((hiddenlayersize,hiddenlayersize))-1.0
			syn2 = 2*np.zeros((hiddenlayersize,1))-1.0
			
			syn0 = syn0.astype(complex)
			syn1 = syn1.astype(complex)
			syn2 = syn2.astype(complex)
	
			l1_bias_weights = np.zeros((hiddenlayersize,))
			l2_bias_weights = np.zeros((hiddenlayersize,))
			
			l1_bias_weights = l1_bias_weights.astype(complex)
			l2_bias_weights = l2_bias_weights.astype(complex)
			
			output_bias = 0.0
		
			r = 0
			c = 0
			count = 0
			#sys.stdout.write(str(syn0[2,0])+'\n')
			for r in range(0,inputlayersize) :
				for c in range(0,hiddenlayersize) :
					#sys.stdout.write(str(syn0[r,c])+'\n')
					#sys.stdout.write(str(weights[count])+'\n')
					#sys.stdout.write(str(r)+' '+str(c)+'\n')
					syn0[r,c] = weights[count]
					count+=1
		
			for r in range(0,hiddenlayersize) :
				for c in range(0,hiddenlayersize) :
					syn1[r,c] = weights[count]
					count+=1		
			for r in range(0,hiddenlayersize) :
				c = 0
				syn2[r,c] = weights[count]
				count+=1
			for r in range(0,hiddenlayersize) :
				l1_bias_weights[r] = weights[count]
				count+=1
			for r in range(0,hiddenlayersize) :
				l2_bias_weights[r] = weights[count]
				count+=1
			output_bias = weights[count]
			
	    	#feedforward training data to find betas
	    	layer0 = trainingdata_inputs
	    	layer1 = np.dot(layer0,syn0)
	    	for i in range(0,n) : layer1[i,:] += l1_bias_weights
	    	layer1 = np.tanh(layer1)
	    	if nLayers == 1 : beta = np.dot(layer1,syn1)+1.0 + output_bias
	    	else : 
	    		layer2 = np.dot(layer1,syn1)
	    		for i in range(0,n) : layer2[i,:] += l2_bias_weights
	    		layer2 = np.tanh(layer2)
	    		beta = np.dot(layer2,syn2)+1.0+output_bias
		
    		return beta 
    		
    	#Compute Jacobian Matrix by Complex Step Autodifferentiation (Forward Mode Complex Step Differentiation)
    	jac = np.zeros((num_weights,n))
    	eps_fd = 0.000000000000000000000001j
    	for i in range(0,num_weights) :
    		beta = np.reshape(beta,(n,))
    		weights_delt = weights.astype(complex)
    		weights_delt[i] += eps_fd
    		
    		beta_delt = forward_prop(weights_delt,inputlayersize,hiddenlayersize,trainingdata_inputs,num_weights)
    		beta_delt = np.reshape(beta_delt,(n,))
    		#jac[i,0:n:1] = (beta_delt-beta)/eps_fd
    		jac[i,0:n:1] = np.real(beta_delt/eps_fd)
    		
    	#sys.stdout.write('Jacobian is: ' +str(jac) +'\n')
    	
    	grad = weights*0.0
    	#weights = np.reshape(weights,(num_weights,))
    	#dweights = opt.approx_fprime(weights,forward_prop,0.000001,inputlayersize,hiddenlayersize,trainingdata_inputs,num_weights)
    	
    	#for i in range(0,num_weights) :	
    	#	for j in range(0,n) :    		
    	#		grad[i] += eps*lambd[j]*jac[i,j]*(Tinf**4.0-Tmodel[j]**4.0)
    	grad_JbB = eps*lambd*(Tinf**4.0-Tmodel**4.0)
    	grad = np.dot(jac,grad_JbB)
    	grad = np.reshape(grad,(np.size(weights),))
    	return grad
    	

