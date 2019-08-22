# py_1D_heat
1D Heat Equation Model Problem for Field Inversion and Machine Learning Demonstration

Python scripts used for the 1D heat equation model problem. Numerical methodology is detailed in my dissertation
and other publications available on my researchgate page:
https://www.researchgate.net/profile/Jonathan_Holland5

-------------------------------------------------------
FI-Classic
-------------------------------------------------------
Command to execute
python heat.py

number of spatial points controled by "n" in script. This performs the field inversion portion. Truth equation solved in truth.py,
the imperfect model and adjoint of imperfect model solved in model.py

-------------------------------------------------------
FIML-Embedded
-------------------------------------------------------
Command to execute:
python heat_backprop.py

Note that the gradient simply uses complex step differentiation (forward differentiation) so it is much more costly than the 
adjoint implementation in FI-Classic. This was needed because of the lack of robust autodiff tools for python. Some exist but
have difficulty with numpy arrays. Therefore the forward differentiation was parallelized, and the number of processors used 
is controlled by nprocs variable in heat_backprop.py.

-------------------------------------------------------
FIML-Direct
-------------------------------------------------------
Command to execute:
python heat_nn.py

Adjoint is implemented in this application so runs just as efficiently as FI-Classic. Also note that a major advantage of FIML-Direct
is that numerous cases (multiple Tinf) cases can be considered simultaneously in the a single inversion. This is demonstrated 
by defining Tinf as a vector. A "-1" executes the variable Tinf case.
