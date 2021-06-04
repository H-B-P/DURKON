import pandas as pd
import numpy as np

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Gauss_hess(pred,act):
 return 2

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Poisson_hess(pred,act):
 return act/(pred*pred)

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Gamma_hess(pred,act):
 return (2*act-pred)/(pred*pred*pred)
