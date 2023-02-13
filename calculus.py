import pandas as pd
import numpy as np
import math
import scipy
from scipy.special import erf

from . import misc

#Easy objective functions

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Logistic_grad(pred,act):
 return (pred-act)/(pred*(1-pred))

#Tweedie

def produce_Tweedie_grad(pTweedie=1.5):
 def Tweedie_grad(pred,act):
  return (pred-act)/(pred**pTweedie)
 return Tweedie_grad

#Easy linkages


def Unity_link(x):
 return x

def Unity_link_grad(x):
 return 1

def Unity_delink(x):
 return x


def Root_link(x):
 return x*x

def Root_link_grad(x):
 return 2*x

def Root_delink(x):
 return x**0.5


def Log_link(x):
 return np.exp(x)

def Log_link_grad(x):
 return np.exp(x)

def Log_delink(x):
 return np.log(x)


def Logit_link(x):
 return 1/(1+np.exp(-x))

def Logit_link_grad(x):
 return np.exp(-x)/((1+np.exp(-x))**2)

def Logit_delink(x):
 return np.log(x/(1-x))

links={"Unity":Unity_link,"Root":Root_link,"Log":Log_link,"Logit":Logit_link}
linkgrads={"Unity":Unity_link_grad,"Root":Root_link_grad,"Log":Log_link_grad,"Logit":Logit_link_grad}
delinks = {"Unity":Unity_delink, "Root":Root_delink, "Log":Log_delink, "Logit":Logit_delink}

#Multiple modelling!

def Take0(*args):
 return args[0]
def Take1(*args):
 return args[1]
def Take2(*args):
 return args[2]
def Take3(*args):
 return args[3]
def Take4(*args):
 return args[4]

def js0(*args):
 return pd.Series([0]*len(args[0]))
def js1(*args):
 return pd.Series([1]*len(args[0]))

def nonefunc(*args):
 return None


def Add_mlink(*args):
 return sum(args) #Amazingly, this works!

def Add_mlink_grad(*args):
 return pd.Series([1]*len(args[0]))


def Max_mlink_2(x1, x2):
 return (x1>x2)*x1+(x1<=x2)*x2 #I am aware that this line in particular is an abomination, and refuse to care. Rewrite it at your whim, dear reader.

def Max_mlink_grad_2_A(x1,x2):
 return (x1>=x2).astype(int)

def Max_mlink_grad_2_B(x1,x2):
 return (x2>=x1).astype(int)


def Min_mlink_2(x1, x2):
 return (x1<x2)*x1+(x1>=x2)*x2 #I am aware that this line in particular is an abomination, and refuse to care. Rewrite it at your whim, dear reader.

def Min_mlink_grad_2_A(x1,x2):
 return (x1<=x2).astype(int)

def Min_mlink_grad_2_B(x1,x2):
 return (x2<=x1).astype(int)


def Mult_mlink_2(x1, x2):
 return x1*x2

def Mult_mlink_grad_2_A(x1,x2):
 return x2

def Mult_mlink_grad_2_B(x1,x2):
 return x1


#Learning Rate Adjustments

def default_LRA(*args):
 return 1


def addsmoothing_LRA_0(*args):
 return sum([sum(arg.abs()) for arg in args])/sum(args[0].abs())

def addsmoothing_LRA_1(*args):
 return sum([sum(arg.abs()) for arg in args])/sum(args[1].abs())

def addsmoothing_LRA_2(*args):
 return sum([sum(arg.abs()) for arg in args])/sum(args[2].abs())

def addsmoothing_LRA_3(*args):
 return sum([sum(arg.abs()) for arg in args])/sum(args[3].abs())

def addsmoothing_LRA_4(*args):
 return sum([sum(arg.abs()) for arg in args])/sum(args[4].abs())

addsmoothing_LRAs = [addsmoothing_LRA_0,addsmoothing_LRA_1,addsmoothing_LRA_2,addsmoothing_LRA_3,addsmoothing_LRA_4]

#alongside-error models . . .

def Add_mlink_allbutlast(*args):
 return sum(args[:-1])

def Add_mlink_onlylast(*args):
 return args[-1]

def Add_mlink_grad_void(*args):
 return pd.Series([0]*len(args[0]))



def addsmoothing_LRA_0_erry(*args):
 return sum([sum(arg.abs()) for arg in args[:-1]])/sum(args[0].abs())

def addsmoothing_LRA_1_erry(*args):
 return sum([sum(arg.abs()) for arg in args[:-1]])/sum(args[1].abs())

def addsmoothing_LRA_2_erry(*args):
 return sum([sum(arg.abs()) for arg in args[:-1]])/sum(args[2].abs())

def addsmoothing_LRA_3_erry(*args):
 return sum([sum(arg.abs()) for arg in args[:-1]])/sum(args[3].abs())

def addsmoothing_LRA_4_erry(*args):
 return sum([sum(arg.abs()) for arg in args[:-1]])/sum(args[4].abs())

addsmoothing_LRAs_erry = [addsmoothing_LRA_0_erry, addsmoothing_LRA_1_erry, addsmoothing_LRA_2_erry, addsmoothing_LRA_3_erry, addsmoothing_LRA_4_erry]


#Cratio

def Cratio_mlink(*args):
 return sum(args)/(sum(args)+1)

def Cratio_mlink_grad(*args):
 return 1/((sum(args)+1)*(sum(args)+1))


#Tobit



def gnormal_u_diff(u, p, y):
 return -(y*(y-u)/((p**2)*(u**3)) - 1/u)

def gnormal_p_diff(u, p, y):
 return -((y-u)**2/((p**3)*(u**2)) - 1/p)

def PDF(u, p, y):
 return np.exp(-0.5*((y-u)/(p*u))**2) / (p*u*math.sqrt(2*math.pi))

def CDF(u, p, y):
 return 0.5*(1 - erf((y-u)/(p*u*math.sqrt(2))))

def gnormal_u_diff_censored(u, p, y):
 return -((y/u)*PDF(u,p,y)/CDF(u,p,y))

def gnormal_p_diff_censored(u, p, y):
 return -(((y-u)/p)*PDF(u,p,y)/CDF(u,p,y))
 
#Techne

def techne_mult_eval(pred,act, obj=[[-1,-1],[0,1],[1,-1]]):
 err = (pred-act)/act
 return sum(misc.get_effect_of_this_cont_col(err, obj))/len(err)

def create_techne_mult_grad(obj):
 def techne_grad(pred, act):
  err = (pred-act)/act
  op = pd.Series([0]*len(err))
  op[err<=obj[0][0]] = (obj[1][1]-obj[0][1])/(obj[1][0]-obj[0][0])
  for i in range(len(obj)-1):
   subset = (err>=obj[i][0]) & (err<=obj[i+1][0])
   op[subset] = (obj[i+1][1]-obj[i][1])/(obj[i+1][0]-obj[i][0])
  op[err>=obj[-1][0]] = (obj[-1][1]-obj[-2][1])/(obj[-1][0]-obj[-2][0])
  return op
 return techne_grad
 
