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
