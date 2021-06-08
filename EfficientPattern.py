import numpy as np
from math import sqrt
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import sparse
from scipy.sparse.linalg import eigs
#from scikits.umfpack import spsolve #UMFPACK If you work in serial.
from scipy.sparse.linalg import spsolve
from tqdm import trange, tqdm
from nodepy.runge_kutta_method import *

class ODESol:
    def __init__(self,timesteps,timestep,U):
        self.t = timesteps;
        self.h = timestep;
        self.y = U;
def laplacian_1D(m):
    em = np.ones(m)
    e1=np.ones(m-1)
    A = (sparse.diags(-2*em,0)+sparse.diags(e1,-1)+sparse.diags(e1,1))/((2/(m+1))**2);
    A[0,-1]=1/((2/(m+1))**2);
    A[-1,0]=1/((2/(m+1))**2);
    return A;
def laplacian_2D(m):
    I = np.eye(m)
    A = laplacian_1D(m)
    return sparse.kron(A,I) + sparse.kron(I,A)
def Reaction(t,x,parameters):
    u = x[0:m**2]; v = x[m**2:]; #We grab the useful quantity.
    d1 = parameters["delta1"]; d2 = parameters["delta2"];
    t1 = parameters["tau1"]; t2 = parameters["tau2"];
    a = parameters["alpha"]; b = parameters["beta"]; g = parameters["gamma"];
    #Reaction
    du = a*u*(1-t1*(v**2))+v*(1-t2*u);
    dv = b*v*(1+(a*t1/b)*(u*v))+u*(g+t2*v);
    b = np.append(du,dv,axis=0);
    return b;
def SplitSolver(F,T,U0,arg,N):
    tt = np.linspace(T[0],T[1],N);
    h = tt[1]-tt[0];
    U = np.zeros([len(U0),N]);
    U[:,0] = U0;
    for i in trange(0,N-1):
        B = sparse.bmat([[arg["delta1"]*A,None],[None,arg["delta2"]*A]]);
        UStar = spsolve((sparse.identity(B.shape[0])-h*B),U[:,i])
        Y1 = UStar;
        Y2 = UStar + 0.5*h*F(tt[i],Y1,arg);
        Y3 = UStar + 0.5*h*F(tt[i]+0.5*h,Y2,arg);
        Y4 = UStar + h*F(tt[i]+0.5*h,Y3,arg);
        U[:,i+1] = UStar+(h/6)*(F(tt[i],Y1,arg)+2*F(tt[i]+ 0.5*h,Y2,arg)+2*F(tt[i]+ 0.5*h,Y3,arg)+F(tt[i]+h,Y4,arg))
    sol = ODESol(tt,h,U);
    return sol;
Table = []
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02, "tau2":0.2,"alpha":0.899,"beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.001,"delta2":0.0045,"tau1":0.02, "tau2":0.2,"alpha":0.899,"beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02, "tau2":0.2,"alpha":1.9,"beta":-0.91,"gamma":-1.9}]

Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":2.02, "tau2":0.0,"alpha":2.0,"beta":-0.91,"gamma":-2}]
Table = Table + [{"delta1":0.00105,"delta2":0.0021,"tau1":3.5, "tau2":0.0,"alpha":0.899,"beta":-0.91,"gamma":-0.899}]
Table = Table + [{"delta1":0.00225,"delta2":0.0045,"tau1":0.02, "tau2":0.2,"alpha":1.9,"beta":-0.85,"gamma":-1.9}]
Table = Table + [{"delta1":0.00225,"delta2":0.0005,"tau1":2.02, "tau2":0.0,"alpha":2.0,"beta":-0.91,"gamma":-2}]

TIndex = int(input("What pattern you want to simulate ? "))
m=100
x=np.linspace(0,1,m+2); x=x[1:-1]
y=np.linspace(0,1,m+2); y=y[1:-1]
X,Y=np.meshgrid(x,y)
A=laplacian_2D(m)
#Generating the initial data
mu, sigma = 0, 0.5 # mean and standard deviation
u0 = np.random.normal(mu, sigma,m**2);
v0 = np.random.normal(mu, sigma,m**2)
#Plotting the initial data
plt.figure()
U0=u0.reshape([m,m])
plt.pcolor(X,Y,U0)
plt.colorbar();
plt.title("Initial Data u");
plt.figure()
V0=v0.reshape([m,m])
plt.pcolor(X,Y,V0)
plt.colorbar();
plt.title("Initial Data v");

t0 = 0.0                  # Initial time
tfinal = 150            # Final time
dt_output=0.3          # Interval between output for plotting
N=int(tfinal/dt_output)       # Number of output times
ODE = SplitSolver(Reaction,[t0,tfinal],np.append(u0,v0,axis=0),Table[TIndex],N);
uu=ODE.y
ut = uu[0:m**2,-1];
vt = uu[m**2:,-1];
Ut=ut.reshape([m,m])
plt.figure()
plt.pcolor(X,Y,Ut)
plt.colorbar();
plt.figure()
Vt=vt.reshape([m,m])
plt.pcolor(X,Y,Vt)
plt.colorbar();
plt.show();
