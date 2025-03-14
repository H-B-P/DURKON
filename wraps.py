import pandas as pd
import numpy as np

import copy
import os

import actual_modelling
import prep
import misc
import calculus
import viz
import impose

ALPHABET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def prep_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None, bandConts=False):
 df = inputDf.reset_index(drop=True)
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 1, weightCol, bandConts)
 return model


def train_gamma_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=calculus.Gamma_grad, imposns=imposns, momentum=momentum, prints=prints)
 return model

def train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=calculus.Poisson_grad, imposns=imposns, momentum=momentum, prints=prints)
 return model

def train_tweedie_model(inputDf, resp, nrounds, lr, model, pTweedie=1.5, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 Tweedie_grad = calculus.produce_Tweedie_grad(pTweedie) 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=Tweedie_grad, imposns=imposns, momentum=momentum, prints=prints)
 return model

#

def train_gzg_gamma_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal", lb=None, ub=None):
 df = inputDf.reset_index(drop=True)
 gobj = calculus.get_gzg_grad(calculus.Gamma_grad, lb, ub)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=gobj, imposns=imposns, momentum=momentum, prints=prints)
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
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")


def interxhunt_poisson_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions", save=True):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Poisson_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 if save:
  sugDf.to_csv(filename+".csv")
 else:
  return sugDf




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
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=1))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=Tweedie_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=1))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")

#JPAB, currently Poisson-only, to be upd8ed as needed

def jpab_pared_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal", pen=0.1, pencc=0, pe=0.02):
 if pencc>0:
  print("PC!")
  model = train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats, momentum=momentum, imposns = imposns+[impose.get_penalize_conts_complexity(lr*pencc)], prints=prints)
  model = misc.simplify_conts(model, pe)
 if pen>0:
  print("P!")
  model = train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats, momentum=momentum, imposns = imposns+[impose.get_penalize_nondefault(lr*pen)], prints=prints)
  model = misc.de_feat(model,1)
 
 model = train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats, momentum=momentum, imposns = imposns, prints=prints)
 return model

def jpab_interacted_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal", N=4, n=1, replace=False, pen=0, pencc=0, pe=0.02):
 
 model = jpab_pared_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats, momentum=momentum, imposns=imposns, prints=prints, pen=pen, pencc=pencc, pe=pe)
 
 for I in range(N):
  sDf = interxhunt_poisson_model(inputDf, resp, cats=[c for c in model["cats"]], conts=[c for c in model["conts"]], model=model, silent=False, weightCol=None, filename="suggestions", save=False)
  for i in range(n):
   Type = sDf.loc[i,"Type"]
   feat1, feat2 = sDf.loc[i,"Interaction"].split(" X ")
   if Type=="catcat":
    if feat1 in model["cats"] and feat2 in model["cats"]:
     model = prep.add_catcat_to_model(model, inputDf, feat1, feat2, replace=replace)
   if Type=="catcont":
    if feat1 in model["cats"] and feat2 in model["conts"]:
     model = prep.add_catcont_to_model(model, inputDf, feat1, feat2, replace=replace)
   if Type=="contcont":
    if feat1 in model["conts"] and feat2 in model["conts"]:
     model = prep.add_contcont_to_model(model, inputDf, feat1, feat2, replace=replace)
  if replace:
   model = train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats+cats+conts, momentum=momentum, imposns=imposns, prints=prints)
  else:
   model = train_poisson_model(inputDf, resp, nrounds, lr, model, weightCol=weightCol, staticFeats=staticFeats, momentum=momentum, imposns=imposns, prints=prints)
 
 return model

def jpab_parallelized_poisson_models(inputDf, resp, nrounds, lr, models, weightCol=None, staticFeats=[], momentum=0, imposnLists=[[]], prints="normal", surfrac=0.1):
 df = inputDf.reset_index(drop=True)
 for i in range(len(models)):
  lrs = [0]*len(models)
  lrs[i] = lr
  models = actual_modelling.train_models([df], resp, round(nrounds*surfrac), lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Poisson_grad]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], imposnLists=imposnLists, momentum=momentum, prints=prints)
 
 lrs = [lr]*len(models)
 models = actual_modelling.train_models([df], resp, nrounds - len(models)*round(nrounds*surfrac), lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Poisson_grad]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], imposnLists=imposnLists, momentum=momentum, prints=prints)
 return models
 
#We now return to your regularly scheduled walls of code

def prep_models(inputDf, resp, cats, conts, N=1, fractions=None, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 if fractions==None:
  denom = N*(N+1)/2
  fractions = [(N-x)/denom for x in range(N)]
 models = []
 for fraction in fractions:
  model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 1, weightCol)
  model["BASE_VALUE"]*=fraction
  models.append(model)
 return models

def train_gamma_models(inputDf, resp, nrounds, lrs, models, weightCol=None, staticFeats=[], imposnLists=[[]], momentum=0, prints="normal"):
 df = inputDf.reset_index(drop=True)
 models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Gamma_grad]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], imposnLists=imposnLists, momentum=momentum, prints=prints)
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
  

def gnormalize_gamma_models(models, inputDf, resp, cats, conts, startingErrorPercent=20, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 errModel = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 1, weightCol)
 errModel["BASE_VALUE"]=startingErrorPercent*1.25/100
 models.append(errModel)
 return models

def train_gnormal_models(inputDfs, resp, nrounds, lrs, models, weightCol=None, staticFeats=[], prints="normal", momentum=0, imposnLists=[[]]):
 if type(inputDfs)!=list:
  dfs=[inputDfs.reset_index(drop=True)]
 else:
  dfs = [inputDf.reset_index(drop=True) for inputDf in inputDfs]
 
 models = actual_modelling.train_models(dfs, resp, nrounds, lrs, models, weightCol, staticFeats, lras = calculus.addsmoothing_LRAs_erry[:len(models)-1] + [calculus.default_LRA], lossgrads = [[calculus.gnormal_u_diff, calculus.gnormal_p_diff],[calculus.gnormal_u_diff_censored, calculus.gnormal_p_diff_censored]], links=[calculus.Add_mlink_allbutlast, calculus.Add_mlink_onlylast], linkgrads=[[calculus.Add_mlink_grad]*(len(models)-1)+[calculus.Add_mlink_grad_void], [calculus.Add_mlink_grad_void]*(len(models)-1)+[calculus.Add_mlink_grad]], momentum=momentum, imposnLists=imposnLists, prints=prints)
 
 return models

def train_doublecensored_gnormal_models(inputDfs, resps, nrounds, lrs, models, weightCol=None, staticFeats=[], prints="normal", momentum=0, imposnLists=[[]]):
 if type(inputDfs)!=list:
  dfs=[inputDfs.reset_index(drop=True)]
 else:
  dfs = [inputDf.reset_index(drop=True) for inputDf in inputDfs]
 
 models = actual_modelling.train_models(dfs, resps, nrounds, lrs, models, weightCol, staticFeats, lras = calculus.addsmoothing_LRAs_erry[:len(models)-1] + [calculus.default_LRA], lossgrads = [[calculus.gnormal_u_diff_doubleuncensored, calculus.gnormal_p_diff_doubleuncensored],[calculus.gnormal_u_diff_doublecensored, calculus.gnormal_p_diff_doublecensored]], links=[calculus.Add_mlink_allbutlast, calculus.Add_mlink_onlylast], linkgrads=[[calculus.Add_mlink_grad]*(len(models)-1)+[calculus.Add_mlink_grad_void], [calculus.Add_mlink_grad_void]*(len(models)-1)+[calculus.Add_mlink_grad]], momentum=momentum, imposnLists=imposnLists, prints=prints)
 
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



def interxhunt_doublecensored_gnormal_models(inputDfs, resps, cats, conts, models, silent=False, weightCol=None, filename="suggestions"):
 
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
   trialModels = train_doublecensored_gnormal_models(dfs, resps, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
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
   trialModels = train_doublecensored_gnormal_models(dfs, resps, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
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
   trialModels = train_doublecensored_gnormal_models(dfs, resps, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
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


def prep_additive_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 0, weightCol)
 model["featcomb"] = "addl"
 return model

def train_normal_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal"):
 df = inputDf.reset_index(drop=True)
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=calculus.Gauss_grad, momentum=momentum, imposns=imposns, prints=prints)
 return model

def interxhunt_normal_model(inputDf, resp, cats, conts, model, silent=False, weightCol=None, filename="suggestions"):
 
 df = inputDf.reset_index(drop=True)
 
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad=calculus.Gauss_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")


def prep_classifier_model(inputDf, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 0, weightCol)
 model["BASE_VALUE"] = calculus.Logit_delink(model["BASE_VALUE"])
 model["featcomb"] = "addl"
 return model

def train_classifier_model(inputDf, resp, nrounds, lr, model, weightCol=None, staticFeats=[], prints="normal", momentum=0, imposns=[]):
 
 df = inputDf.reset_index(drop=True)
 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, lossgrad=calculus.Logistic_grad, link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, momentum=momentum, imposns=imposns, prints=prints)
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
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link,  linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")




def prep_adjustment_model(inputDf, resp, startingPoint, cats, conts, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 
 df = inputDf.reset_index(drop=True)
 
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 1, weightCol)
 model["BASE_VALUE"]=1
 model["featcomb"]="mult"
 if "conts" not in model:
  model["conts"]={}
 model["conts"][startingPoint] = [[min(df[startingPoint]), min(df[startingPoint])], [max(df[startingPoint]), max(df[startingPoint])]]
 return model

def train_adjustment_model(inputDf, resp, startingPoint, nrounds, lr, model, weightCol=None, staticFeats=[], momentum=0, imposns=[], prints="normal"):
 
 df = inputDf.reset_index(drop=True)
 
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats=[startingPoint]+staticFeats, lossgrad=calculus.Gamma_grad,  momentum=momentum, imposns=imposns, prints=prints)
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
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"mult"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":1, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"mult"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=1)
   trialmodel = actual_modelling.train_model(df, resp, 1, 0.2, trialmodel, staticFeats=["PredComb"], lossgrad = calculus.Gamma_grad, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv(filename+".csv")




def train_gamma_suvec_models(lowerDf, upperDf, resp, nrounds, lrs, models, weightCol=None, staticFeats=[], imposnLists=[[]], momentum=0, prints="normal"):
 models = actual_modelling.train_models([lowerDf, upperDf], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.suvec_gamma_lower],[calculus.suvec_gamma_upper]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], imposnLists=imposnLists, momentum=momentum, prints=prints)
 return models




def prep_cratio_models(inputDf, resp, cats, conts, N=1, fractions=None, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, weightCol=None):
 df = inputDf.reset_index(drop=True)
 if fractions==None:
  denom = N*(N+1)/2
  fractions = [(N-x)/denom for x in range(N)]
 models = []
 for fraction in fractions:
  model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, contEdge, 1, weightCol)
  model["BASE_VALUE"] = misc.frac_to_ratio(model["BASE_VALUE"])
  model["BASE_VALUE"]*=fraction
  models.append(model)
 return models

def train_cratio_models(inputDf, resp, nrounds, lrs, models, pens=None, weightCol=None, staticFeats=[], minrela=0.01, momentum=0, imposnLists=None, prints="normal"):
 df = inputDf.reset_index(drop=True)
 if imposnLists==None:
  models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Logistic_grad]], links=[calculus.Cratio_mlink], linkgrads=[[calculus.Cratio_mlink_grad]*len(models)], momentum=momentum, imposnLists=[[impose.get_enforce_min_rela(minrela)]]*len(models), prints=prints)
 else:
  models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Logistic_grad]], links=[calculus.Cratio_mlink], linkgrads=[[calculus.Cratio_mlink_grad]*len(models)], momentum=momentum, imposnLists=imposnLists, prints=prints)
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





def flinkify_additive_model(inputDf, model, contTargetPts=5, contEdge=0.0, weightCol=None):
 df = inputDf.reset_index(drop=True)
 df["preds"] = misc.predict(df, model)
 model["flink"] = prep.get_cont_feat(df, "preds",contTargetPts, contEdge, 0, weightCol=weightCol)
 return model
 
def flinkify_multiplicative_model(inputDf, model, contTargetPts=5, contEdge=0.0, weightCol=None):
 df = inputDf.reset_index(drop=True)
 df["preds"] = misc.predict(df, model)
 model["flink"] = prep.get_cont_feat(df, "preds",contTargetPts, contEdge, 1, weightCol=weightCol)
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
