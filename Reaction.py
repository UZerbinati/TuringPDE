import numpy as np
from math import sqrt
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import sparse
from scipy.sparse.linalg import eigs
from scikits.umfpack import spsolve #UMFPACK If you work in serial.
from tqdm import trange, tqdm
from nodepy.runge_kutta_method import *

class ODESol:
    def __init__(self,timesteps,timestep,U):
        self.t = timesteps;
        self.h = timestep;
        self.y = U;
def RK4(F,T,U0,arg,N):
    tt = np.linspace(T[0],T[1],N);
    h = tt[1]-tt[0];
    U = np.zeros([len(U0),N]);
    U[:,0] = U0;
    for i in trange(0,N-1):
        Y1 = U[:,i];
        Y2 = U[:,i] + 0.5*h*F(tt[i],Y1,arg);
        Y3 = U[:,i] + 0.5*h*F(tt[i]+0.5*h,Y2,arg);
        Y4 = U[:,i] + h*F(tt[i]+0.5*h,Y3,arg);
        U[:,i+1] = U[:,i]+(h/6)*(F(tt[i],Y1,arg)+2*F(tt[i]+ 0.5*h,Y2,arg)+2*F(tt[i]+ 0.5*h,Y3,arg)+F(tt[i]+h,Y4,arg))
    sol = ODESol(tt,h,U);
    return sol;

Table = []
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02,"tau2":0.2,"alpha":0.899, "beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.001,"delta2":0.0045,"tau1":0.02,"tau2":0.2,"alpha":0.899, "beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02,"tau2":0.2,"alpha":1.9, "beta":-0.91,"gamma":-1.9}]

Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":2.02,"tau2":0.0,"alpha":2.0, "beta":-0.91,"gamma":-2}]
Table = Table + [{"delta1":0.00105,"delta2":0.0021,"tau1":3.5,"tau2":0.0,"alpha":0.899, "beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02,"tau2":0.2,"alpha":1.9, "beta":-0.85,"gamma":-1.9}]
Table = Table + [{"delta1":0.00225,"delta2":0.0005,"tau1":2.02,"tau2":0.0,"alpha":2.0, "beta":-0.91,"gamma":-2}]

def Reaction(t,x,parameters):
    #The ODE is autonomus so we don't really need
    #the depende on time.
    u = x[0]; v = x[1]; #We grab the useful quantity.
    d1 = parameters["delta1"]; d2 = parameters["delta2"];
    t1 = parameters["tau1"]; t2 = parameters["tau2"];
    a = parameters["alpha"]; b = parameters["beta"]; g = parameters["gamma"];
    du = a*u*(1-t1*(v**2))+v*(1-t2*u);
    dv = b*v*(1+(a*t1/b)*(u*v))+u*(g+t2*v);
    return np.array([du,dv])

t0 = 0.                  # Initial time
u0 = np.array([0.5,0.5])# Initial values
tfinal = 100.              # Final time
dt_output=0.1# Interval between output for plotting
N=int(tfinal/dt_output)       # Number of output times
print(N)
tt=np.linspace(t0,tfinal,N)  # Output times
ODE = RK4(Reaction,[t0,tfinal],u0,Table[6],N);
uu=ODE.y
plt.plot(tt,uu[0,:],tt,uu[1,:])
plt.legend(["u","v"])
plt.show();

