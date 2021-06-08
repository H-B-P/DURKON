import pandas as pd
import numpy as np
import math

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def get_gradient(x, y):
 aveX = float(sum(x))/len(x)
 aveY = float(sum(y))/len(y)
 
 numerator=sum((x-aveX)*(y-aveY))
 denominator=sum((x-aveX)*(x-aveX))
 
 return numerator/denominator

def target_input_with_output(func, desiredOutput, startingUB, startingLB, iters=10): #this assumes monotonicity
 ub=startingUB
 lb=startingLB
 for i in range(iters):
  mp = (ub+lb)/2.0
  mpOP = func(mp)
  if mpOP<desiredOutput:
   ub=mp
  else:
   lb=mp
 return (ub+lb)/2.0

if __name__ == "__main__":
 print(round_to_sf(12.3456,3))
 
 print(get_gradient(np.array([1,2,3,4]),np.array([2,4,6,8])))
 
 def sq(a):
  return a*a
 print(target_input_with_output(sq, 9, 0, 100, 20))
