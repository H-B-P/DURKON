import pandas as pd
import numpy as np

import copy
import os

from . import actual_modelling
from . import prep
from . import misc
from . import calculus
from . import viz

ALPHABET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def prep_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
 return model


def train_gamma_model(inputDf, resp, nrounds, lr, model, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=pen, specificPens={}, lossgrad=calculus.Gamma_grad, prints=prints)
 return model

def train_poisson_model(inputDf, resp, nrounds, lr, model, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=pen, specificPens={}, lossgrad=calculus.Poisson_grad, prints=prints)
 return model

def train_tweedie_model(inputDf, resp, nrounds, lr, model, pTweedie=1.5, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 Tweedie_grad = calculus.produce_Tweedie_grad(pTweedie) 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=pen, specificPens={}, lossgrad=Tweedie_grad, prints=prints)
 return model





def interxhunt_gamma_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")


def interxhunt_poisson_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")




def interxhunt_tweedie_model(inputDf, resp, cats, conts, model, pTweedie=1.5, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 Tweedie_grad = calculus.produce_Tweedie_grad(pTweedie) 
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")




def prep_gamma_models(inputDf, resp, cats, conts, N=1, fractions=None, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 if fractions==None:
  denom = N*(N+1)/2
  fractions = [(N-x)/denom for x in range(N)]
 models = []
 for fraction in fractions:
  model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
  model["BASE_VALUE"]*=fraction
  models.append(model)
 return models

def train_gamma_models(inputDf, resp, nrounds, lrs, models, pens=None, weightCol=None, staticFeats=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Gamma_grad]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], pens=pens, prints=prints)
 return models

def interxhunt_gamma_models(inputDf, resp, cats, conts, models, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 trialModelTemplate = []
 
 for m in range(len(models)):
  df["PredComb_"+str(m)]=misc.predict(df, models[m])
  trialModelTemplate.append({"BASE_VALUE":1, "conts":{"PredComb_"+str(m):[[min(df["PredComb_"+str(m)]),min(df["PredComb_"+str(m)])],[max(df["PredComb_"+str(m)]),max(df["PredComb_"+str(m)])]]}, "featcomb":"mult"})
 
 sugImps=[[] for m in range(len(models))]
 sugFeats=[[] for m in range(len(models))]
 sugTypes=[[] for m in range(len(models))]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   if not silent:
    print(cats[i] + " X " + cats[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcat_to_model(trialModel, df, cats[i], cats[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   print(trialModels)
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+cats[j])
    sugImps[m].append(misc.get_importance_of_this_catcat(df, trialModels[m], cats[i]+" X "+cats[j], defaultValue=0))
    sugTypes[m].append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   if not silent:
    print(cats[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcont_to_model(trialModel, df, cats[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_catcont(df, trialModels[m], cats[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("catcont")
   
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   if not silent:
    print(conts[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_contcont_to_model(trialModel, df, conts[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(conts[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_contcont(df, trialModels[m], conts[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("contcont")
 
 
 for m in range(len(models)):
  print(sugFeats[m],sugTypes[m],sugImps[m])
  sugDf = pd.DataFrame({"Interaction":sugFeats[m], "Type":sugTypes[m], "Importance":sugImps[m]})
  sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
  sugDf.to_csv(filename+"_"+ALPHABET[m]+".csv")
  

def gnormalize_gamma_models(models, inputDf, resp, cats, conts, startingErrorPercent=20, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 errModel = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
 errModel["BASE_VALUE"]=startingErrorPercent*1.25/100
 models.append(errModel)
 return models

def train_gnormal_models(inputDfs, resp, nrounds, lrs, models, pens=None, weightCol=None, staticFeats=[], prints="normal"):
 if type(inputDfs)!=list:
  dfs=[inputDfs.reset_index(drop=True)]
 else:
  dfs = [inputDf.reset_index(drop=True) for inputDf in inputDfs]
 
 models = actual_modelling.train_models(dfs, resp, nrounds, lrs, models, weightCol, staticFeats, lras = calculus.addsmoothing_LRAs_erry[:len(models)-1] + [calculus.default_LRA], lossgrads = [[calculus.gnormal_u_diff, calculus.gnormal_p_diff],[calculus.gnormal_u_diff_censored, calculus.gnormal_p_diff_censored]], links=[calculus.Add_mlink_allbutlast, calculus.Add_mlink_onlylast], linkgrads=[[calculus.Add_mlink_grad]*(len(models)-1)+[calculus.Add_mlink_grad_void], [calculus.Add_mlink_grad_void]*(len(models)-1)+[calculus.Add_mlink_grad]], pens=pens, prints=prints)
 
 return models

def interxhunt_gnormal_models(inputDfs, resp, cats, conts, models, silent=False, weightCol=None, filename="suggestions"):
 
 if type(inputDfs)!=list:
  dfs = [inputDfs.reset_index(drop=True)]
 else:
  dfs = [inputDf.reset_index(drop=True) for inputDf in inputDfs]
 
 if len(dfs)==2:
  cdf = dfs[0].append(dfs[1]).reset_index()
 else:
  cdf = dfs[0]
 
 trialModelTemplate = []
 
 
 for m in range(len(models)):
  for df in dfs:
   df["PredComb_"+str(m)]=misc.predict(df, models[m])
  
  minever = min([min(df["PredComb_"+str(m)]) for df in dfs])
  maxever = max([max(df["PredComb_"+str(m)]) for df in dfs])
  trialModelTemplate.append({"BASE_VALUE":1, "conts":{"PredComb_"+str(m):[[minever, minever], [maxever, maxever]]}, "featcomb":"mult"})
 
 
 sugImps=[[] for m in range(len(models))]
 sugFeats=[[] for m in range(len(models))]
 sugTypes=[[] for m in range(len(models))]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   if not silent:
    print(cats[i] + " X " + cats[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcat_to_model(trialModel, cdf, cats[i], cats[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+cats[j])
    sugImps[m].append(misc.get_importance_of_this_catcat(cdf, trialModels[m], cats[i]+" X "+cats[j], defaultValue=0))
    sugTypes[m].append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   if not silent:
    print(cats[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcont_to_model(trialModel, cdf, cats[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_catcont(cdf, trialModels[m], cats[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("catcont")
   
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   if not silent:
    print(conts[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_contcont_to_model(trialModel, cdf, conts[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(conts[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_contcont(cdf, trialModels[m], conts[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("contcont")
 
 
 for m in range(len(models)-1):
  sugDf = pd.DataFrame({"Interaction":sugFeats[m], "Type":sugTypes[m], "Importance":sugImps[m]})
  sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
  sugDf.to_csv(filename+"_"+ALPHABET[m]+".csv")


def predict_from_gnormal(df, model):
 return misc.predict_models(df, model, calculus.Add_mlink_allbutlast)
 
def predict_error_from_gnormal(df, model):
 return misc.predict_models(df, model, calculus.Add_mlink_onlylast)*100/1.25


def prep_additive_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 0, weightCol)
 model["featcomb"] = "addl"
 return model

def train_additive_model(inputDf, resp, nrounds, lr, model, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=pen, specificPens={}, lossgrad=calculus.Gauss_grad, prints=prints)
 return model

def interxhunt_additive_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")


def prep_classifier_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 0, weightCol)
 model["BASE_VALUE"] = calculus.Logit_delink(model["BASE_VALUE"])
 model["featcomb"] = "addl"
 return model

def train_classifier_model(inputDf, resp, nrounds, lr, model, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 
 df = inputDf.reset_index(drop=True)
 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=pen, specificPens={}, lossgrad=calculus.Logistic_grad, link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, prints=prints)
 return model

def interxhunt_classifier_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link,  linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")




def prep_adjustment_model(inputDf, resp, startingPoint, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
 model["BASE_VALUE"]=1
 model["featcomb"]="mult"
 if "conts" not in model:
  model["conts"]={}
 model["conts"][startingPoint] = [[min(df[startingPoint]), min(df[startingPoint])], [max(df[startingPoint]), max(df[startingPoint])]]
 return model

def train_adjustment_model(inputDf, resp, startingPoint, nrounds, lr, model, pen=0, weightCol=None, staticFeats=[], prints="normal"):
 
 df = inputDf.reset_index(drop=True)
 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats=[startingPoint]+staticFeats, pen=pen, specificPens={}, lossgrad=calculus.Gamma_grad, prints=prints)
 return model

def interxhunt_adjustment_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"mult"}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"mult"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"mult"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")





def prep_cratio_models(inputDf, resp, cats, conts, N=1, fractions=None, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 if fractions==None:
  denom = N*(N+1)/2
  fractions = [(N-x)/denom for x in range(N)]
 models = []
 for fraction in fractions:
  model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
  model["BASE_VALUE"] = misc.frac_to_ratio(model["BASE_VALUE"])
  model["BASE_VALUE"]*=fraction
  models.append(model)
 return models

def train_cratio_models(inputDf, resp, nrounds, lrs, models, pens=None, weightCol=None, staticFeats=[], minRela=0.1, prints="normal"):
 df = inputDf.reset_index(drop=True)
 models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Logistic_grad]], links=[calculus.Cratio_mlink], linkgrads=[[calculus.Cratio_mlink_grad]*len(models)], pens=pens, minRela=minRela, prints=prints)
 return models

def interxhunt_cratio_models(inputDf, resp, cats, conts, models, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 trialModelTemplate = []
 
 for m in range(len(models)):
  df["PredComb_"+str(m)]=misc.predict(df, models[m])
  trialModelTemplate.append({"BASE_VALUE":1, "conts":{"PredComb_"+str(m):[[min(df["PredComb_"+str(m)]),min(df["PredComb_"+str(m)])],[max(df["PredComb_"+str(m)]),max(df["PredComb_"+str(m)])]]}, "featcomb":"mult"})
 
 sugImps=[[] for m in range(len(models))]
 sugFeats=[[] for m in range(len(models))]
 sugTypes=[[] for m in range(len(models))]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   if not silent:
    print(cats[i] + " X " + cats[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcat_to_model(trialModel, df, cats[i], cats[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_cratio_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   print(trialModels)
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+cats[j])
    sugImps[m].append(misc.get_importance_of_this_catcat(df, trialModels[m], cats[i]+" X "+cats[j], defaultValue=0))
    sugTypes[m].append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   if not silent:
    print(cats[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcont_to_model(trialModel, df, cats[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_cratio_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_catcont(df, trialModels[m], cats[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("catcont")
   
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   if not silent:
    print(conts[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_contcont_to_model(trialModel, df, conts[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_cratio_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(conts[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_contcont(df, trialModels[m], conts[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("contcont")





def flinkify_additive_model(inputDf, model, contTargetPts=5, edge=0.0, weightCol=None):
 df = inputDf.reset_index(drop=True)
 df["preds"] = misc.predict(df, model)
 model["flink"] = prep.get_cont_feat(df, "preds",contTargetPts, edge, 0, weightCol=weightCol)
 return model
 
def flinkify_multiplicative_model(inputDf, model, contTargetPts=5, edge=0.0, weightCol=None):
 df = inputDf.reset_index(drop=True)
 df["preds"] = misc.predict(df, model)
 model["flink"] = prep.get_cont_feat(df, "preds",contTargetPts, edge, 1, weightCol=weightCol)
 return model


def viz_model(model, modelName=0, targetSpan=0.5, defaultValue=1, ytitle="Relativity", subfolder=None, otherName="OTHER", heatmapDetail=10):
 
 try:
  os.mkdir("graphs")
 except:
  pass
 
 if subfolder!=None:
  try:
   os.mkdir("graphs/"+subfolder)
  except:
   pass
  
  fsf = "graphs/"+subfolder
 else:
  fsf="graphs"
 
 
 if "flink" in model:
  viz.draw_cont_pdp(model["flink"], targetSpan, "Un-Flinked Prediction", model=modelName, defaultValue=defaultValue, ytitle=ytitle, folder=fsf)
 if "conts" in model:
  for col in model["conts"]:
   viz.draw_cont_pdp(model["conts"][col], targetSpan, col, model=modelName, defaultValue=defaultValue, ytitle=ytitle, folder=fsf)
 if "cats" in model:
  for col in model["cats"]:
   viz.draw_cat_pdp(model["cats"][col], targetSpan, col, model=modelName, defaultValue=defaultValue, ytitle=ytitle, folder=fsf, otherName=otherName)
 if "contconts" in model:
  for cols in model["contconts"]:
   c1, c2 = cols.split(" X ")
   viz.draw_contcont_pdp(model["contconts"][cols], targetSpan, cols, model=modelName, cont1=c1, cont2=c2, defaultValue=defaultValue, ytitle=ytitle, folder=fsf)
   viz.draw_contcont_pdp_3D(model["contconts"][cols], targetSpan, cols+", 3D", model=modelName, cont1=c1, defaultValue=defaultValue, cont2=c2, ytitle=ytitle, folder=fsf)
   viz.draw_contcont_pdp_heatmap(model["contconts"][cols], targetSpan, cols+", Heatmap", model=modelName, cont1=c1, defaultValue=defaultValue, cont2=c2, ytitle=ytitle, folder=fsf, detail=heatmapDetail)
 if "catconts" in model:
  for cols in model["catconts"]:
   c1, c2 = cols.split(" X ")
   viz.draw_catcont_pdp(model["catconts"][cols], targetSpan, cols, model=modelName, cat=c1, cont=c2, defaultValue=defaultValue, ytitle=ytitle, folder=fsf, otherName=otherName)
 if "catcats" in model:
  for cols in model["catcats"]:
   c1, c2 = cols.split(" X ")
   viz.draw_catcat_pdp(model["catcats"][cols], targetSpan, cols, model=modelName, cat1=c1, cat2=c2, defaultValue=defaultValue, ytitle=ytitle, folder=fsf, otherName=otherName)

def viz_logistic_model(model, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 viz_model(model, defaultValue=0, ytitle="LPUs", subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)

def viz_gamma_models(models, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 if len(models)==1:
  viz_model(models[0], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
 else:
  for m in range(len(models)):
   viz_model(models[m], modelName=ALPHABET[m], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)

def viz_gnormal_models(models, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 if len(models)==2:
  viz_model(models[0], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
 else:
  for m in range(len(models)-1):
   viz_model(models[m], modelName=ALPHABET[m], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
 viz_model(models[-1], modelName="PercentageError", subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)

def viz_additive_model(model, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 viz_model(model, defaultValue=0, ytitle="Delta", subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)

def viz_multiplicative_model(model, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 viz_model(model, defaultValue=1, ytitle="Multiplier", subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)

def viz_adjustment_model(model, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 viz_model(model, ytitle="Adjustment multiplier", subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
 
def viz_cratio_models(models, subfolder=None, targetSpan=0.5, otherName="OTHER"):
 if len(models)==1:
  viz_model(models[0], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
 else:
  for m in range(len(models)):
   viz_model(models[m], modelName=ALPHABET[m], subfolder=subfolder, targetSpan=targetSpan, otherName=otherName)
