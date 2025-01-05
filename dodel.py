import misc
import calculus

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
