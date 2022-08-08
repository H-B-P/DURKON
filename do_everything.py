import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus

import wraps

#Logistic Proof of Concept 

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,2], "cat1":['a','a','a','a','b','b','b','b'], "cat2":['c','c','d','d','c','d','e','d'], "y":[0,0,0,1,0,0,0,1]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_classifier_model(df, "y", cats, conts)
model = wraps.train_classifier_model(df, "y", 50, 0.1, model)

pred = misc.predict(df,model, "Logit")
print(pred)

wraps.interxhunt_classifier_model(df, "y", cats, conts, model)

model = prep.add_catcont_to_model(model, df, 'cat1','cont1')
model = prep.add_catcat_to_model(model, df, 'cat1', 'cat2')
model = prep.add_contcont_to_model(model, df, 'cont1', 'cont2')

wraps.viz_logistic_model(model)

#Gamma Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

models = wraps.prep_gamma_models(df, 'y', cats, conts, 2)
models = wraps.train_gamma_models(df, 'y', 50, [0.2,0.3], models)

pred = misc.predict_models(df, models)
print(pred)

wraps.interxhunt_gamma_models(df, "y", cats, conts, models)

models[0] = prep.add_catcont_to_model(models[0], df, 'cat1','cont1')
models[0] = prep.add_catcat_to_model(models[0], df, 'cat1', 'cat2')
models[0] = prep.add_contcont_to_model(models[0], df, 'cont1', 'cont2')

wraps.viz_gamma_models(models)

#Gnormal Proof of Concept

models = wraps.gnormalize_gamma_models(models, df, "y", cats, conts, 10)

models = wraps.train_gnormal_models(df, 'y', 1000, [0.001,0.002,0.001], models)
pred = wraps.predict_from_gnormal(df, models)
predErrPct = wraps.predict_error_from_gnormal(df, models)

print(pred, predErrPct)

wraps.interxhunt_gnormal_models(df,'y',cats,conts,models)

wraps.viz_gnormal_models(models)

#Additive Proof of Concept

model = wraps.prep_additive_model(df,'y',cats, conts)

model = wraps.train_additive_model(df,'y', 50, 0.2, model)
pred = misc.predict(df,model)

print(pred)

wraps.interxhunt_additive_model(df, 'y', cats, conts, model)

wraps.viz_additive_model(model)

#Adjustment Proof of Concept

df["start"] = [1,2,3,4,2.5,3,3.5,8] #We are trying to figure out how best to get from the 'start' column to the 'y' column.

model = wraps.prep_adjustment_model(df,'y', 'start', cats, conts)

print(model)

model = wraps.train_adjustment_model(df, 'y', 'start', 50, 0.2, model)

pred = misc.predict(df,model)

print(pred)

wraps.interxhunt_adjustment_model(df, 'y', cats, conts, model)

wraps.viz_adjustment_model(model)

