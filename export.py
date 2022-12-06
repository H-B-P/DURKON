import misc

exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}

def find_max_len(model, multip=1):
 ml=0
 for col in model['conts']:
  if (len(model['conts'][col])*multip)>ml:
   ml = (len(model['conts'][col])*multip)
 for col in model['cats']:
  if (len(model['cats'][col]['uniques'])+1)>ml:
   ml = len(model['cats'][col]['uniques'])+1
 return ml

def get_cont_inputs(cont, detail=1):
 op=[]
 for i in range(len(cont)-1):
  for j in range(detail):
   op.append(float(cont[i][0]*(detail-j) + cont[i+1][0]*j)/detail)
 op.append(cont[-1][0])
 return op

def linesify_catcat(catcat):
 
 skeys1 = misc.get_sorted_keys(catcat)
 skeys2 = misc.get_sorted_keys(catcat["OTHER"])
 
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

def linesify_catcont(catcont, detail=1):
 
 skeys = misc.get_sorted_keys(catcont)
 contPuts=get_cont_inputs(catcont['OTHER'], detail)
 
 op=[',']*(len(contPuts)+2)
 
 #Headline
 op[0]=op[0]+','
 for k in skeys:
  op[0] = op[0]+","+k
 op[0] = op[0]+",OTHER"
 
 #Work downwards
 for i in range(len(contPuts)):
  op[i+1]=op[i+1]+','+str(contPuts[i])
  for j in range(len(skeys)):
   op[i+1] = op[i+1]+','+str(misc.get_effect_of_this_catcont_on_single_input(skeys[j],contPuts[i], catcont))
  op[i+1] = op[i+1]+','+str(misc.get_effect_of_this_catcont_on_single_input("OTHER",contPuts[i], catcont))
 
 return op

def linesify_contcont(contcont, detail=1):
 
 contPuts1=get_cont_inputs(contcont, detail)
 contPuts2=get_cont_inputs(contcont[0][1], detail)
 
 op=[',']*(len(contPuts2)+2)
 
 #Headline
 op[0]=op[0]+','
 for contPut in contPuts1:
  op[0] = op[0]+","+str(contPut)
 
 #Work downwards
 for i in range(len(contPuts2)):
  op[i+1]=op[i+1]+','+str(contPuts2[i])
  for j in range(len(contPuts1)):
   op[i+1] = op[i+1]+','+str(misc.get_effect_of_this_contcont_on_single_input(contPuts1[j],contPuts2[i], contcont))
 
 return op

def model_to_lines(model, detail=1, filename="op.csv"):
 lines = ['']*(find_max_len(model, detail)+4)
 
 #Add base value
 
 for l in range(len(lines)):
  if l==1:
   lines[l] = lines[l]+",BASE_VALUE"
  elif l==2:
   lines[l] = lines[l]+","+str(model['BASE_VALUE'])
  else:
   lines[l] = lines[l]+','
 
 #Add conts
 
 if 'conts' in model:
  for col in model['conts']:
   lines[0] = lines[0]+',,,'
   lines[1] = lines[1]+',,'+col+','
   contPuts=get_cont_inputs(model['conts'][col], detail)
   for i in range(len(contPuts)):
     lines[i+2] = lines[i+2] + ',,' + str(contPuts[i]) + ',' + str(misc.get_effect_of_this_cont_on_single_input(contPuts[i], model['conts'][col]))
   for l in range(len(contPuts)+2, len(lines)):
    lines[l] = lines[l]+',,,'
 
 #Add cats
 if 'cats' in model:
  for col in model['cats']:
   lines[0] = lines[0]+',,,'
   lines[1] = lines[1]+',,'+col+','
   keys = misc.get_sorted_keys(model["cats"][col])
   for i in range(len(keys)):
    lines[i+2] = lines[i+2]+',,'+str(keys[i])+','+str(model['cats'][col]['uniques'][keys[i]])
   lines[len(keys)+2] = lines[len(keys)+2] +',,OTHER,'+str(model['cats'][col]['OTHER'])
   for l in range(len(keys)+3, len(lines)):
    lines[l] = lines[l]+',,,'
 
 #Interactions
 if 'catcats' in model:
  for cols in model['catcats']:
   extra = linesify_catcat(model['catcats'][cols])
   lines = lines + [',,'+cols] + extra + ['']
 
 if 'catconts' in model:
  for cols in model['catconts']:
   extra = linesify_catcont(model['catconts'][cols], detail)
   lines = lines + [',,'+cols] + extra + ['']
 
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
 model_to_lines(exampleModel, 4, 'example.csv')
 #print()