import numpy as np
import pandas as pd

import random

random.seed(0)

def roll_dX(X):
 return random.choice(list(range(X)))+1

flavCoeffs = {"marzipan":7, "coffee":11, "chocolate":8, "vanilla":7, "strawberry": 7, "raspberry":6, "lemon":9}
fancyCoeffs = {"No":3, "Somewhat":4, "Very":3}
fancyWeddingCoeffs = {"No":2, "Somewhat":6, "Very":9}


def gen(nrow):
 
 widths = []
 heights = []
 thicknesses = []
 
 flavours = []
 weddings = []
 fancys = []
 
 prices = []
 
 for i in range(nrow):
  bigness = roll_dX(4)+roll_dX(4)
  width = bigness+roll_dX(4)+roll_dX(4)
  height = bigness+roll_dX(4)
  volume = width*width*height
  thickness = 7+roll_dX(6)-roll_dX(4)
  
  flavour = random.choice(["marzipan"]*11 + ["coffee"]*2 + ["chocolate"]*107 + ["vanilla"]*92 + ["strawberry"]*33 + ["raspberry"]*1 + ["lemon"]*37)
  wedding = random.choice(["Yes"]*12+["No"]*86)
  fancy = random.choice(["No"]*23+["Somewhat"]*51+["Very"]*30)
  
  price = 1000 + (volume+roll_dX(20)-roll_dX(20)) * (flavCoeffs[flavour]+roll_dX(4)) * (10- abs(thickness-7))
  
  if wedding=="Yes":
   price = price*(fancyWeddingCoeffs[fancy]+roll_dX(4))
  else:
   price = price*(fancyCoeffs[fancy]+roll_dX(4))
  
  price = round(price/100)/100
  
  widths.append(width)
  heights.append(height)
  thicknesses.append(thickness)
  flavours.append(flavour)
  weddings.append(wedding)
  fancys.append(fancy)
  prices.append(price)
  
 dictForDf = {"Width":widths, "Height":heights, "Icing Thickness":thicknesses, "Flavour":flavours, "Wedding?":weddings, "Fancy?":fancys,"Price":prices}

 df = pd.DataFrame(dictForDf)
 
 return df


trainDf = gen(12345)
print(trainDf)
trainDf.to_csv('train.csv', index=False)

testDf = gen(2345)
print(testDf)
testDf.to_csv('test.csv', index=False) 