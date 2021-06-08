import pandas as pd
import numpy as np
import math
import util

def gimme_pseudo_winsors(inputDf, col, pw=0.05):
 return util.round_to_sf(inputDf[col].quantile(pw),3), util.round_to_sf(inputDf[col].quantile(1-pw),3)

def default_seg(inputDf, col, targetNofpts=5, pw=0.02):
 addl = (1-2*pw)/(targetNofpts-1)
 segs=[]
 for i in range(targetNofpts):
  segs.append(util.round_to_sf(inputDf[col].quantile(pw+addl*i),3))
 return segs

if __name__ == "__main__":
 dyct = {"x":list(range(101))}
 df=pd.DataFrame(dyct)
 print(default_seg(df, "x", 5, 0.02))

