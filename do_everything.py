import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus
import wraps
import viz
import impose
import radify


#

#Weighted Poisson proof of concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "exposure":[1,2,1,2,1,2,1,2], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_model(df, "y", cats, conts)
print(model)
model = wraps.train_poisson_model(df, 'y', 50, 0.1, model, weightCol="exposure")
print(model)

#JPAB proof of concept (still for weighted Poisson; still only for weighted Poisson)

model = wraps.prep_model(df, "y", cats, conts)
print(model)
model = wraps.jpab_interacted_poisson_model(df, 'y',50,0.03, model, weightCol="exposure", N=2)
print(model)

models = wraps.prep_models(df,"y",cats,conts,N=2)
print(models)
models = wraps.jpab_parallelized_poisson_models(df, 'y',50,0.1,models, weightCol="exposure")
print(models)

#Logistic Proof of Concept 

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,2], "cat1":['a','a','a','a','b','b','b','b'], "cat2":['c','c','d','d','c','d','e','d'], "y":[0,0,0,1,0,0,0,1]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_classifier_model(df, "y", cats, conts)
model = wraps.train_classifier_model(df, "y", 50, 0.1, model)

pred = misc.predict(df,model, "Logit")
print(pred)

wraps.interxhunt_classifier_model(df, "y", cats, conts, model, filename="suggestions_logistic")

model = prep.add_catcont_to_model(model, df, 'cat1','cont1')
model = prep.add_catcat_to_model(model, df, 'cat1', 'cat2')
model = prep.add_contcont_to_model(model, df, 'cont1', 'cont2')

wraps.viz_logistic_model(model, "Logistic")

#Saving and Loading

misc.save_model(model)

misc.save_model(model, name="q", timing=False)
m = misc.load_model("models/q.txt")

print(m)
print(model)

#Mark rows where model may not reliably apply

markedDf = misc.mark_anomalous_rows(df, model)

print(markedDf)

#AvE viz

df["Predicted"] = pred
df["Actual"] = df['y']

viz.draw_AvE(df, 'cat1', 'Predicted', 'Actual')

#===

#Gamma Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_model(df,"y",cats,conts)
model = wraps.train_gamma_model(df,"y",50, 0.01, model)
wraps.interxhunt_gamma_model(df, "y", cats, conts, model)

model = prep.add_catcont_to_model(model, df, 'cat1','cont1')
model = prep.add_catcat_to_model(model, df, 'cat1', 'cat2')
model = prep.add_contcont_to_model(model, df, 'cont1', 'cont2')

#Tweedie Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_model(df,"y",cats,conts)
model = wraps.train_tweedie_model(df,"y",50, 0.01, model)
wraps.interxhunt_tweedie_model(df, "y", cats, conts, model)

model = prep.add_catcont_to_model(model, df, 'cat1','cont1')
model = prep.add_catcat_to_model(model, df, 'cat1', 'cat2')
model = prep.add_contcont_to_model(model, df, 'cont1', 'cont2')

#Multimodel Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

models = wraps.prep_models(df, 'y', cats, conts, 2)
models = wraps.train_gamma_models(df, 'y', 50, [0.2,0.3], models)

pred1 = misc.predict(df, models[0])
pred2 = misc.predict(df, models[1])
pred = misc.predict(df, models)

print(pred1, pred2, pred)

wraps.interxhunt_gamma_models(df, "y", cats, conts, models, filename="suggestions_gamma")

models[0] = prep.add_catcont_to_model(models[0], df, 'cat1','cont1')
models[0] = prep.add_catcat_to_model(models[0], df, 'cat1', 'cat2')
models[0] = prep.add_contcont_to_model(models[0], df, 'cont1', 'cont2')

wraps.viz_gamma_models(models, "Gamma")

#Gnormal Proof of Concept

models = wraps.gnormalize_gamma_models(models, df, "y", cats, conts, 11) #I picked 11 because the MPE of the gamma model was a little less than 11%.

models = wraps.train_gnormal_models(df, 'y', 1000, [0.001,0.002,0.001], models)
pred = wraps.predict_from_gnormal(df, models)
predErrPct = wraps.predict_error_from_gnormal(df, models)

print(pred, predErrPct)

wraps.interxhunt_gnormal_models(df,'y',cats,conts,models, filename="suggestions_gnormal")

wraps.viz_gnormal_models(models, "Gnormal")

#Tobit proof of concept (removed for taking too long and requiring gen)

if False:
	df = pd.read_csv('gnormal.csv')
	cdf = df[df['censored']].reset_index()
	udf = df[~df['censored']].reset_index()

	cats=[]
	conts=["x"]

	models = wraps.prep_models(udf, 'y', cats, conts, 1)
	models = wraps.train_gamma_models(udf, 'y', 1000, [0.1], models)

	print(models)

	models = wraps.gnormalize_gamma_models(models, udf, "y", cats, conts, 20)
	models = wraps.train_gnormal_models([udf,cdf], 'y', 1000, [0.1,0.005], models)

	pred = wraps.predict_from_gnormal(df, models)
	predErrPct = wraps.predict_error_from_gnormal(df, models)

	df["PREDICTED"]=pred

	print(df[["PREDICTED","true_y"]])

	print(df["PREDICTED"].mean())
	print(df["true_y"].mean())

	print(models)

	wraps.viz_gnormal_models(models, "Tobit")

	#Twosided Tobit proof of concept

	df = pd.read_csv('gnormal.csv')
	cdf = df[df['censored']].reset_index()
	udf = df[~df['censored']].reset_index()

	cats=[]
	conts=["x"]

	models = wraps.prep_models(udf, 'y', cats, conts, 1)
	models = wraps.train_gamma_models(udf, 'y', 1000, [0.1], models)

	print(models)

	models = wraps.gnormalize_gamma_models(models, udf, "y", cats, conts, 20)
	models = wraps.train_doublecensored_gnormal_models([udf,cdf], ['y','censor1_y','censor2_y'], 3000, [0.01,0.005], models)

	pred = wraps.predict_from_gnormal(df, models)
	predErrPct = wraps.predict_error_from_gnormal(df, models)

	df["PREDICTED"]=pred

	print(df[["PREDICTED","true_y"]])

	print(df["PREDICTED"].mean())
	print(df["true_y"].mean())

	print(models)

	wraps.viz_gnormal_models(models, "Tobit2")

#SuVec proof of concept (removed for requiring gen)

if False:
 df = pd.read_csv('suvec.csv')
 ldf = df[df['over']].reset_index()
 udf = df[~df['over']].reset_index()
 
 cats=[]
 conts=["x"]
 
 models = wraps.prep_models(df, 'censor_y', cats, conts, 1)
 models[0]["BASE_VALUE"]=100
 models = wraps.train_gamma_models(df, 'censor_y', 100, [0.1], models)
 
 print(models)
 
 models = wraps.train_gamma_suvec_models(ldf, udf, "censor_y", 100, [0.5], models)
 
 print(models)

#Additive Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_additive_model(df,'y',cats, conts)

model = wraps.train_normal_model(df,'y', 50, 0.2, model)
pred = misc.predict(df,model)

print(pred)

wraps.interxhunt_normal_model(df, 'y', cats, conts, model, filename="suggestions_additive")

wraps.viz_additive_model(model, "Additive")

#---

#Adjustment Proof of Concept

df["start"] = [1,2,3,4,2.5,3,3.5,8] #We are trying to figure out how best to get from the 'start' column to the 'y' column.

model = wraps.prep_adjustment_model(df,'y', 'start', cats, conts)

print(model)

model = wraps.train_adjustment_model(df, 'y', 'start', 50, 0.2, model)

print(model)

pred = misc.predict(df,model)

print(pred)

wraps.interxhunt_adjustment_model(df, 'y', cats, conts, model, filename="suggestions_adj")

wraps.viz_adjustment_model(model, "Adjustment")

#Gnormalized Adjustment Proof of Concept

models = wraps.gnormalize_gamma_models([model], df, "y", cats, conts, 4)

print(pred, predErrPct)

models = wraps.train_gnormal_models(df, 'y', 2000, [0.0001,0.0001], models, staticFeats=["start"])

print(models)
pred = wraps.predict_from_gnormal(df, models)
predErrPct = wraps.predict_error_from_gnormal(df, models)

print(pred, predErrPct)

wraps.interxhunt_gnormal_models(df,'y',cats,conts,models, filename="suggestions_gnormadj")

wraps.viz_gnormal_models(models, "Gnormalized Adjustment")

#---

#Continuous Complexity Penalization Proof of Concept

df = pd.DataFrame({"cont":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], "y":[1,2,3,4,5,6.2,7,7.8,9,10.2,11,12.4,12.6,14,15.4,16,16.2,18.8,19,20]})
#df = pd.DataFrame({"cont":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], "y":[1,2,3,4,5,4.01,2.99,2,3,4.1,5.1,6,4,2,0,1,2,3,2,1]})
cats=[]
conts=["cont"]

#The intended workflow is as follows:
model = wraps.prep_additive_model(df,'y',cats,conts, contTargetPts=20) #Prep a model with an excessive amount of pts per feat . . .
model["BASE_VALUE"]=0 #(Included to make sensemaking easier)
print(model)
model = wraps.train_normal_model(df,'y',100, 0.2, model, imposns = [impose.get_penalize_conts_complexity(2*0.2)]) # . . . impose a high penalty on polyjointedness in conts . . .
print(model)
model = misc.simplify_conts(model, 0.5) # . . . remove any redundant points . . .
print(model)
model = wraps.train_normal_model(df,'y',100, 0.2, model) # . . . then retrain model sans penalties.
print(model)

#Penalization Proof of Concept

df = pd.DataFrame({"cont1":[1,4,3,2,6,7,5,9,8], "cont2":[1,2,3,1,2,3,1,2,3], "cont3":[1,2,3,4,5,4,3,2,1]})
df["y"]= df["cont1"] + df["cont2"]*0.04 + df["cont3"]*0.01

cats=[]
conts=["cont1","cont2", "cont3"]


#The intended workflow is as follows:
model = wraps.prep_additive_model(df,'y',cats, conts) #Prep a model as per usual . . .
model = wraps.train_normal_model(df,'y', 50, 0.2, model, imposns = [impose.get_penalize_nondefault(0.8*0.2,0)])# . . . train with high penalization (here 0.8, modulo the 0.2 lr) . . .
model = misc.de_feat(model, 0) # . . . then, purge all features that LASSO pulled to the default . . . 
model = wraps.train_normal_model(df,'y', 50, 0.2, model) # . . . and re-train on remaining features with lower (or zero!) penalization.

#(NOTE #1: for a multiplicative model the middle line would be "misc.de_feat(model, 1)", as the default value there is 1.)
#(NOTE #2: de_feat doesn't affect interactions because if you're using penalization to remove interaction effects then you have made poor life choices.)

print(model)

print(misc.what_conts(model)) 
print(misc.what_cats(model))
print(misc.how_many_conts(model))
print(misc.how_many_cats(model))
#"Oh hey, looks like only one feature survived!" (You can vary penalization until you have roughly the number of features you want, of the kinds you want.)

pred = misc.predict(df,model)
print(pred)

wraps.viz_additive_model(model, "Penalized")

#---

#Cratio Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,5,6,7,8,9], "cont2":[1,2,3,1,2,3,1,2,3], "cont3":[1,2,3,4,5,4,3,2,1]})
df["y"]= ((df["cont3"]>3)|((df["cont2"]>2)&(df["cont1"]>4))).astype(int)

cats=[]
conts=["cont1","cont2", "cont3"]

models = wraps.prep_cratio_models(df, 'y', cats, conts, 1)

models = wraps.train_cratio_models(df, 'y', 100, [0.2], models)

print(models)

pred = misc.predict(df,models, calculus.Cratio_mlink)

print(pred)

predRat = misc.predict(df,models)

print(predRat)

print(df["y"])


wraps.viz_cratio_models(models, "Cratio")

#---

#Flink Proof of Concept

df = pd.DataFrame({"cont1":[0,1,0,1],"cat1":[0,0,1,1],"y":[0,1,1,4]})
model = wraps.prep_additive_model(df, "y", ["cat1"], ["cont1"])
model = wraps.train_normal_model(df, "y", 100, 0.1, model)
print(model)
print(misc.predict(df, model))
model = wraps.flinkify_additive_model(df,model)
model = wraps.train_normal_model(df, "y", 100, 0.1, model)
print(model)
print(misc.predict(df, model))
wraps.viz_additive_model(model, "flink")

#Flink also works for multiples, in both senses of the word!

#---

#GZG proof of concept

df = pd.DataFrame({"cont1":[1,2,3,1,2,3,1,2,3,4],"cont2":[1,2,3,4,5,1,2,3,4,5], "cont3":[4,3,2,1,4,3,2,1,4,3], "cont4":[2,3,4,5,6,7,1,2,3,4]})
df["y"] = df["cont1"]*df["cont2"]*df["cont3"]*df["cont4"]
df["y"] = df['y'].clip(lower=20)

cats = []
conts = ["cont1", "cont2","cont3","cont4"]

model = wraps.prep_model(df, "y", cats, conts)
model["conts"]["cont1"]=[[1,1],[4,1]] #These four lines are just to reduce degrees of freedom btw
model["conts"]["cont2"]=[[1,1],[5,1]]
model["conts"]["cont3"]=[[1,1],[4,1]]
model["conts"]["cont4"]=[[1,1],[7,1]]

#model = wraps.train_gamma_model(df, 'y', 500, 0.1, model)
model = wraps.train_gzg_gamma_model(df, 'y', 500, 0.1, model,lb=20)

print(model)

#Note: The entire point of this trick is reassuring the model that you'll apply appropriate censorship to its predictions post hoc; so if you're using it remember to actually do that!

#---


#(I put the banding demo in do_everything_cake.py)
