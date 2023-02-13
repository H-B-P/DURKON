import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus
import wraps

trainDf = pd.read_csv("train.csv")

print(trainDf)

cats=["Flavour","Wedding?","Fancy?"]
conts=["Width","Height","Icing Thickness"]

model = wraps.prep_model(trainDf, "Price", cats, conts)

print(model)

model = wraps.train_gamma_model(trainDf, "Price", 200, 0.1, model)

print(model)



model = wraps.prep_model(trainDf, "Price", cats, conts)

print(model)

model = wraps.train_poisson_model(trainDf, "Price", 200, 0.001, model)

print(model)



model = wraps.prep_model(trainDf, "Price", cats, conts)

print(model)

model = wraps.train_tweedie_model(trainDf, "Price", 200, 0.01, model)

print(model)
