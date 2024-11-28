import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus
import wraps
import export

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

#

model = wraps.prep_model(trainDf, "Price", cats, conts, contEdge=0.1, bandConts=True) #Need more generous bucketing if you're planning to band.
bandedTrainDf = prep.band_df(trainDf, model)
print(model)

model = wraps.train_gamma_model(bandedTrainDf, "Price", 200, 0.1, model)

export.export_model(model, "banded.csv")
print(model)

#

model = wraps.prep_model(trainDf, "Price", cats, conts, contTargetPts=3, contEdge=0.2, bandConts=True) #Need much more generous bucketing if you're planning to interact features.
bandedTrainDf = prep.band_df(trainDf, model)
print(model)
model = prep.add_catcat_to_model(model, bandedTrainDf, "Width", "Height", replace=True, merget=True)
print(model)

model = wraps.train_gamma_model(bandedTrainDf, "Price", 200, 0.1, model)

export.export_model(model, "banded_interacted.csv")
print(model)