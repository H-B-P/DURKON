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


def get_sorted_keys(d):
 keys = [c for c in d]
 keys.sort(key=str)
 return keys


def denump(obj):
 if isinstance(obj, dict):
  return {denump(k):denump(v) for k, v in obj.items()}
 if isinstance(obj, list):
  return [denump(x) for x in obj]
 if isinstance(obj, np.int64):
  return int(obj)
 if isinstance(obj, np.float64):
  return float(obj)
 return obj