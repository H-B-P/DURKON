import copy
import pandas as pd

import misc
import util

exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}

def uniquify_model(model, df):
 newModel = copy.deepcopy(model)
 
 if "conts" in model:
  for col in model["conts"]:
   uList = df[col].unique()
   uList = [u for u in uList]
   uList.sort()
   newModel["conts"][col] = [[u,misc.get_effect_of_this_cont_on_single_input(u, model["conts"][col])] for u in uList]
 
 if "cats" in model:
  for col in model["cats"]:
   uList = df[col].unique()
   uList = [u for u in uList]
   uList.sort(key=str)
   newModel["cats"][col]={"uniques":{}, "OTHER":model["cats"][col]["OTHER"]}
   for u in uList:
    newModel["cats"][col]["uniques"][u] = misc.get_effect_of_this_cat_on_single_input(u, model["cats"][col])
 
 if "catcats" in model:
  for cols in model["catcats"]:
   cat1, cat2 = cols.split(" X ")
   
   uList1 = df[cat1].unique()
   uList1 = [u for u in uList1]
   uList1.sort(key=str)
   
   uList2 = df[cat2].unique()
   uList2 = [u for u in uList2]
   uList2.sort(key=str)
   
   newModel["catcats"][cols]={"uniques":{}, "OTHER":{"uniques":{}, "OTHER":model["catcats"][cols]["OTHER"]["OTHER"]}}
   
   for u2 in uList2:
    newModel["catcats"][cols]["OTHER"]["uniques"][u2] = misc.get_effect_of_this_catcat_on_single_input("SolidGoldMagikarp", u2, model["catcats"][cols])
   
   for u1 in uList1:
    newModel["catcats"][cols]["uniques"][u1] = {"uniques":{},"OTHER":misc.get_effect_of_this_catcat_on_single_input(u1,"SolidGoldMagikarp", model["catcats"][cols])}
    for u2 in uList2:
     newModel["catcats"][cols]["uniques"][u1]["uniques"][u2] = misc.get_effect_of_this_catcat_on_single_input(u1,u2, model["catcats"][cols])
   
  if "catconts" in model:
   for cols in model["catconts"]:
    cat, cont = cols.split(" X ")
    
    uList1 = df[cat].unique()
    uList1 = [u for u in uList1]
    uList1.sort(key=str)
    
    uList2 = df[cont].unique()
    uList2 = [u for u in uList2]
    uList2.sort()
    
    newModel["catconts"][cols] = {"uniques":{}, "OTHER":[[u2, misc.get_effect_of_this_catcont_on_single_input("SolidGoldMagikarp", u2, model["catconts"][cols])] for u2 in uList2]}
    for u1 in uList1:
     newModel["catconts"][cols]["uniques"][u1] = [[u2, misc.get_effect_of_this_catcont_on_single_input(u1, u2, model["catconts"][cols])] for u2 in uList2]
  
  if "contconts" in model:
   for cols in model["contconts"]:
    cont1, cont2 = cols.split(" X ")
    
    uList1 = df[cont1].unique()
    uList1 = [u for u in uList1]
    uList1.sort()
    
    uList2 = df[cont2].unique()
    uList2 = [u for u in uList2]
    uList2.sort()
    
    newModel["contconts"][cols]=[]
    for u1 in uList1:
     newModel["contconts"][cols].append([u1, [[u2, misc.get_effect_of_this_contcont_on_single_input(u1,u2,model["contconts"][cols])] for u2 in uList2]])
 
 return newModel

def find_max_len(model, multip=1):
 ml=0
 for col in model['cats']:
  if (len(model['cats'][col]['uniques'])+1)>ml:
   ml = len(model['cats'][col]['uniques'])+1
 return ml

def linesify_catcat(catcat):
 
 skeys1 = util.get_sorted_keys(catcat["uniques"])
 skeys2 = util.get_sorted_keys(catcat["OTHER"]["uniques"])
 
 op=[',']*(len(skeys2)+2)
 
 #Headline
 op[0]=op[0]+','
 for k1 in skeys1:
  op[0] = op[0]+","+k1
 op[0] = op[0]+",OTHER"
 
 #Work downwards
 for i in range(len(skeys2)):
  op[i+1]=op[i+1]+','+skeys2[i]
  for j in range(len(skeys1)):
   op[i+1] = op[i+1]+','+str(misc.get_effect_of_this_catcat_on_single_input(skeys1[j],skeys2[i],catcat))#str(catcat['uniques'][skeys1[j]]['uniques'][skeys2[i]])
  op[i+1] = op[i+1]+','+str(misc.get_effect_of_this_catcat_on_single_input("OTHER",skeys2[i],catcat))
 
 op[-1]=op[-1]+",OTHER"
 for j in range(len(skeys1)):
  op[-1]=op[-1]+','+str(misc.get_effect_of_this_catcat_on_single_input(skeys1[j],"OTHER",catcat))
 op[-1]=op[-1]+','+str(misc.get_effect_of_this_catcat_on_single_input("OTHER","OTHER",catcat))
 
 return op

def export_model(model, detail=1, filename="op.csv"):
 lines = ['']*(find_max_len(model, detail)+4)
 
 #Add base value
 
 for l in range(len(lines)):
  if l==1:
   lines[l] = lines[l]+",BASE_VALUE"
  elif l==2:
   lines[l] = lines[l]+","+str(model['BASE_VALUE'])
  else:
   lines[l] = lines[l]+','
 
 #Add cats
 if 'cats' in model:
  for col in model['cats']:
   lines[0] = lines[0]+',,,'
   lines[1] = lines[1]+',,'+col+','
   keys = util.get_sorted_keys(model["cats"][col]["uniques"])
   for i in range(len(keys)):
    lines[i+2] = lines[i+2]+',,'+str(keys[i])+','+str(model['cats'][col]['uniques'][keys[i]])
   lines[len(keys)+2] = lines[len(keys)+2] +',,OTHER,'+str(model['cats'][col]['OTHER'])
   for l in range(len(keys)+3, len(lines)):
    lines[l] = lines[l]+',,,'
 
 #Interactions
 
 if 'contconts' in model:
  for cols in model['contconts']:
   extra = linesify_contcont(model['contconts'][cols], detail)
   lines = lines + [',,'+cols] + extra + ['']
 
 #Write to file
 
 f = open(filename, "w")
 for line in lines:
  #print(line)
  f.write(line+'\n')
 f.close()

if __name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]], "cont3":[[12,1.2],[13, 0.9],[14, 1.1]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.01},"cat2":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, 'catcats':{'cat1 X cat2':{"uniques":{"wstfgl":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}, "florpalorp":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, "OTHER":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}}, 'catconts':{"cat1 X cont1":{"uniques":{"wstfgl":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "florpalorp":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}, "OTHER":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}}, 'contconts':{'cont1 X cont2':[[0.01, [[37,1.2],[98, 0.9]]],[0.02,[[37,1.2],[98, 0.9]]], [0.03, [[37,1.2],[98, 9000]]]]}}
 export_model(exampleModel, 4, 'example.csv')
 
 exampleDf = pd.DataFrame({"cont1":[0.01, 0.012, 0.017, 0.04], "cont2":[40,50,60,70], "cont3":[10,11,12,15], "cat1":["wstfgl","florpalorp","wstfgl","turlingdrome"], "cat2":["ruska","roma","rita","ryla"]})
 uModel = uniquify_model(exampleModel, exampleDf)
 
 print(uModel)
 #print()
