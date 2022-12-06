import pandas as pd
import numpy as np
import math
import copy

def prep_model(df, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, defaultValue=1, weightCol=None):
 
 model={"BASE_VALUE":df[resp].mean(), "featcomb":"mult"}
 
 for cat in cats:
  model = add_cat_to_model(model, df, cat, catMinPrev, defaultValue, weightCol)
 for cont in conts:
  model = add_cont_to_model(model, df, cont, contTargetPts,edge, defaultValue, weightCol)
 
 return model


def get_cat_feat(df, cat, catMinPrev=0.01, defaultValue=1, weightCol=None):
 
 if weightCol==None:
  df["WEIGHT_COL"]=1
 else:
  df["WEIGHT_COL"] = df[weightCol]
 
 sw = sum(df["WEIGHT_COL"])
 
 feat = {"OTHER":defaultValue, "uniques":{}}
 
 for u in df[cat].unique():
  if sum(df[df[cat]==u]["WEIGHT_COL"]/sw)>catMinPrev:
   feat['uniques'][u]=defaultValue
 
 return feat

def add_cat_to_model(model, df, cat, catMinPrev=0.01, defaultValue=1, weightCol=None):
 
 if "cats" not in model:
  model["cats"]={}
 model['cats'][cat] = get_cat_feat(df,cat,catMinPrev, defaultValue, weightCol)
 
 return model



def get_cont_feat(df, cont, contTargetPts=5, edge=0.01, defaultValue=1, weightCol=None):
 
 if weightCol==None:
  df["WEIGHT_COL"]=1
 else:
  df["WEIGHT_COL"] = df[weightCol]
 
 df = df[~df[cont].isna()]
 
 sw = sum(df["WEIGHT_COL"])
 
 inpts = []
 for i in range(contTargetPts):
  inpts.append(edge+(1-edge*2)*i/(contTargetPts-1))
 
 df = df.sort_values(by=cont).reset_index()
 df["CSWC"]=df["WEIGHT_COL"].cumsum() - df["WEIGHT_COL"][0]
 
 pts=[]
 feat=[]
 
 for inpt in inpts:
  newpt = max(df[df["CSWC"]<=(inpt*sw)][cont])
  if newpt not in pts:
   pts.append(newpt)
 
 for pt in pts:
  feat.append([pt,defaultValue])
 
 return feat

def add_cont_to_model(model, df, cont, contTargetPts=5, edge=0.01, defaultValue=1, weightCol=None):

 if "conts" not in model:
  model["conts"]={}
 model['conts'][cont] = get_cont_feat(df,cont,contTargetPts, edge, defaultValue, weightCol)
 
 return model



def get_contcont_feat(df, cont1, cont2, contTargetPts1=3, contTargetPts2=3, edge1=0.05, edge2=0.05, defaultValue=1, weightCol=None):
 
 feat1 = get_cont_feat(df, cont1, contTargetPts1, edge1, defaultValue, weightCol)
 feat2 = get_cont_feat(df, cont2, contTargetPts2, edge2, defaultValue, weightCol)
 
 for i in range(len(feat1)):
  feat1[i][1]=copy.deepcopy(feat2)
 
 return feat1

def add_contcont_to_model(model, df, cont1, cont2, contTargetPts1=3, contTargetPts2=3, edge1=0.05, edge2=0.05, defaultValue=1, weightCol=None, replace=False):
 
 if "contconts" not in model:
  model["contconts"]={}
 model['contconts'][cont1 + " X " + cont2] = get_contcont_feat(df, cont1, cont2, contTargetPts1, contTargetPts2, edge1, edge2, defaultValue, weightCol)
 
 if replace:
  if "conts" in model:
   if cont1 in model["conts"]:
    del model["conts"][cont1]
   if cont2 in model["conts"]:
    del model["conts"][cont2]
 
 return model



def get_catcont_feat(df, cat, cont, catMinPrev=0.1, contTargetPts=3, edge=0.05, defaultValue=1, weightCol=None):
 
 feat1 = get_cat_feat(df, cat, catMinPrev, defaultValue, weightCol)
 feat2 = get_cont_feat(df, cont, contTargetPts, edge, defaultValue, weightCol)
 
 feat1["OTHER"]=copy.deepcopy(feat2)
 
 for u in feat1["uniques"]:
  feat1["uniques"][u]=copy.deepcopy(feat2)
 
 return feat1

def add_catcont_to_model(model, df, cat, cont, catMinPrev=0.1, contTargetPts=3, edge=0.05, defaultValue=1, weightCol=None, replace=False):
 
 if "catconts" not in model:
  model["catconts"]={}
 model['catconts'][cat + " X " + cont] = get_catcont_feat(df, cat, cont, catMinPrev, contTargetPts, edge, defaultValue, weightCol)
 
 if replace:
  if "cats" in model:
   if cat in model["cats"]:
    del model["cats"][cat]
  if "conts" in model:
   if cont in model["conts"]:
    del model["conts"][cont]
 
 return model



def get_catcat_feat(df, cat1, cat2, catMinPrev1=0.1, catMinPrev2=0.1, defaultValue=1, weightCol=None):
 
 feat1 = get_cat_feat(df, cat1, catMinPrev1, defaultValue, weightCol)
 feat2 = get_cat_feat(df, cat2, catMinPrev2, defaultValue, weightCol)
 
 feat1["OTHER"]=copy.deepcopy(feat2)
 
 for u in feat1["uniques"]:
  feat1["uniques"][u]=copy.deepcopy(feat2)
 
 return feat1

def add_catcat_to_model(model, df, cat1, cat2, catMinPrev1=0.1, catMinPrev2=0.1, defaultValue=1, weightCol=None, replace=False):
 
 if "catcats" not in model:
  model["catcats"]={}
 model['catcats'][cat1 + " X " + cat2] = get_catcat_feat(df, cat1, cat2, catMinPrev1, catMinPrev2, defaultValue, weightCol)
 
 if replace:
  if "cats" in model:
   if cat1 in model["cats"]:
    del model["cats"][cat1]
   if cat2 in model["cats"]:
    del model["cats"][cat2]
 
 return model


if __name__ == '__main__':
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 print(get_cont_feat(df, "cont1", 5, 0.001, 1))
 
 