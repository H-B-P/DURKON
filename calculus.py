import pandas as pd
import numpy as np
import math
import scipy
from scipy.special import erf

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

# . . . and make it double!

def gnormal_u_diff_doublecensored(u,p,y,c1,c2):
 return ((-c2/u)*PDF(u,p,c2) - (-c1/u)*PDF(u,p,c1)) / (CDF(u,p,c2) - CDF(u,p,c1))

def gnormal_p_diff_doublecensored(u,p,y,c1,c2):
 return ((-(c2-u)/p)*PDF(u,p,c2) - (-(c1-u)/p)*PDF(u,p,c1)) / (CDF(u,p,c2) - CDF(u,p,c1))

def gnormal_u_diff_doubleuncensored(u,p,y,c1,c2):
 return gnormal_u_diff(u,p,y)

def gnormal_p_diff_doubleuncensored(u,p,y,c1,c2):
 return gnormal_p_diff(u,p,y)

#Suvec

def suvec_normal_upper(pred,act):
 return (pred-act)*(pred>act)

def suvec_normal_lower(pred,act):
 return (pred-act)*(pred<act)

def suvec_gamma_upper(pred,act):
 return ((pred-act)/(pred*pred))*(pred>act)

def suvec_gamma_lower(pred,act):
 return ((pred-act)/(pred*pred))*(pred<act)

#Techne

def get_effect_of_this_cont_col(ser, cont):
 x = ser
 effectOfCol = pd.Series([1]*len(ser))
 effectOfCol.loc[(x<=cont[0][0])] = cont[0][1] #Everything too early gets with the program
 for i in range(len(cont)-1):
  x1 = cont[i][0]
  x2 = cont[i+1][0]
  y1 = cont[i][1]
  y2 = cont[i+1][1]
  effectOfCol.loc[(x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[x>=cont[-1][0]] = cont[-1][1] #Everything too late gets with the program
 return effectOfCol

def techne_mult_eval(pred,act, obj=[[-1,-1],[0,1],[1,-1]]):
 err = (pred-act)/act
 return sum(get_effect_of_this_cont_col(err, obj))/len(err)

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

#Dodel Template (realistically you'll have to customize this for each model you're optimizing a dodel for)

#profit = (pri-lia)*prob
#d(profit)/d(price) = prob + (pri-lia) * d(prob)/d(price)
#d(prob)/d(price) = d(prob)/d(comb) * d(comb)/d(pricefeateff) * d(pricefeateff)/d(pricefeat) * d(pricefeat)/d(price)
#d(prob)/d(comb) is link of the comb
#d(comb)/d(pricefeateff) is, for an additive-combed model, just 1
#d(pricefeateff)/d(feat) is the model's line gradient
#d(feat)/d(price) is 1/mar because we assume a percentage

def get_dodel_lossgrad(model, df, pfname):
 def dodel_lossgrad(pri, lia, mar):
  df[pfname] = pri/mar
  
  #prob = misc.predict(model,df,"Logit")
  comb = misc.predict(model,df)
  prob = calculus.Logit_link(comb)
  
  dpdc = calculus.Logit_link_grad(comb)
  dcde = 1
  dedf = get_gradient_of_this_cont_col(pri/mar, model["conts"][pfname])
  dfdp = 1/mar
  
  dpdp = dpdc*dcde*dedf*dfdp
  
  return prob + (pri-lia) * dpdp
 return dodel_lossgrad

def get_gradient_of_this_cont_col(ser, cont):
 x = ser
 gradOfCol = pd.Series([0.0]*len(ser)) # at the points themselves, and everywhere not covered, grad is 0
 for i in range(len(cont)-1):
  x1 = cont[i][0]
  x2 = cont[i+1][0]
  y1 = cont[i][1]
  y2 = cont[i+1][1]
  gradOfCol.loc[(x>x1)&(x<x2)] = (y2 - y1)/(x2 - x1)
 return gradOfCol

#Gradient Zero Gimmick

def get_gzg_grad(obj, lb=None, ub=None):
 def gzg_grad(pred, act):
  grad = obj(pred,act)
  if ub is not None:
   grad[pred>ub]=0
  if lb is not None:
   grad[pred<lb]=0
  return grad
 return gzg_grad

