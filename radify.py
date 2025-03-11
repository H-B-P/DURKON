import util

def radify_cat(cat, col):
 op = "IF ISEMPTY(" + str(col) + ") THEN " + str(cat["OTHER"]) + " ELSE "
 for u in cat["uniques"]:
  if type(u)==str:
   op += ("IF " + col + '="' + u + '" THEN ' +  str(cat["uniques"][u]) + " ELSE ")
  else:
   op += ("IF " + col + '=' + str(util.denump(u)) + ' THEN ' +  str(cat["uniques"][u]) + " ELSE ")
 op += (str(cat["OTHER"]))
 return op

def radify_cont(cont, col, defaultValue=1):
 op = "IF ISEMPTY(" + str(col) + ") THEN " + str(defaultValue) + " ELSE IF ISNAN(" + str(col) + ") THEN " + str(defaultValue) + " ELSE "
 op += ("IF " + col + "<=" + str(cont[0][0]) + " THEN " + str(cont[0][1]) + " ELSE ")
 for i in range(len(cont)-1):
  p1 = str(cont[i][0])
  p2 = str(cont[i+1][0])
  y1 = str(cont[i][1])
  y2 = str(cont[i+1][1])
  op += ("IF " + col +  "<=" + p2 + " THEN " + "(("+col + "-" + p1+")*" + y2 + "+(" + p2 + "-" + col + ")*" + y1 + ")/(" + p2 + "-" + p1 + ") ELSE ") #((x-p1)y2 + (p2-x)y1) / (p2 - p1)
 op += ("IF " + col + ">=" + str(cont[-1][0]) + " THEN " + str(cont[-1][1]) + " ELSE " +str(defaultValue))
 return op

def radify_catcat(model, cols):
 col1, col2 = cols.split(" X ")
 catcat=model["catcats"][cols]
 
 op = "IF ISEMPTY(" + str(col1) + ") THEN (" + radify_cat(catcat["OTHER"], col2) + ") ELSE IF ISNAN(" + str(col1) + ") THEN " + radify_cat(catcat["OTHER"], col2) + " ELSE "
 for u in catcat["uniques"]:
  op += (" IF " + str(col1) + "= " + str(u) + " THEN (")
  op += radify_cat(catcat["uniques"][u], col2)
  op += (") ELSE ")
 op+=("("+radify_cat(catcat["OTHER"], col2)+")")
 return op

def radify_catcont(model, cols, defaultValue=1):
 col1, col2 = cols.split(" X ")
 catcont=model["catconts"][cols]
 
 op = "IF ISEMPTY(" + str(col1) + ") THEN (" + radify_cont(catcont["OTHER"], col2, defaultValue) + ") ELSE IF ISNAN(" + str(col1) + ") THEN " + radify_cont(catcont["OTHER"], col2, defaultValue) + " ELSE "
 for u in catcont["uniques"]:
  op += (" IF " + str(col1) + "= " + str(u) + " THEN (")
  op += radify_cont(catcont["uniques"][u], col2, defaultValue)
  op += (") ELSE ")
 op+= ("("+radify_cont(catcont["OTHER"], col2, defaultValue)+")")
 return op

def radify_contcont(model, cols, defaultValue=1):
 
 #we are using x and y to predict z
 
 col1, col2 = cols.split(" X ")
 contcont=model["contconts"][cols]
 
 op = ("IF ISEMPTY(" + str(col1) + ") THEN " + str(defaultValue) + " ELSE IF ISNAN(" + str(col1) + ") THEN " + str(defaultValue) + " ELSE ")
 op += ("IF ISEMPTY(" + str(col2) + ") THEN " + str(defaultValue) + " ELSE IF ISNAN(" + str(col2) + ") THEN " + str(defaultValue) + " ELSE ")
 
 #CORNERS
 
 op += ("IF " + str(col1) + "<=" + str(contcont[0][0]) + " AND " + str(col2) + "<=" + str(contcont[0][1][0][0]) + " THEN " + str(contcont[0][1][0][1]) + " ELSE ")
 op += ("IF " + str(col1) + ">=" + str(contcont[-1][0]) + " AND " + str(col2) + "<=" + str(contcont[-1][1][0][0]) + " THEN " + str(contcont[-1][1][0][1]) + " ELSE ")
 op += ("IF " + str(col1) + "<=" + str(contcont[0][0]) + " AND " + str(col2) + "<=" + str(contcont[0][1][-1][0]) + " THEN " + str(contcont[0][1][-1][1]) + " ELSE ")
 op += ("IF " + str(col1) + "<=" + str(contcont[-1][0]) + " AND " + str(col2) + "<=" + str(contcont[-1][1][-1][0]) + " THEN " + str(contcont[-1][1][-1][1]) + " ELSE ")
 
 #EDGES
 
 for i in range(len(contcont)-1):
  x1 = str(contcont[i][0])
  x2 = str(contcont[i+1][0])
  z1 = str(contcont[i][1][0][1])
  z2 = str(contcont[i+1][1][0][1])
  op+=("IF " + str(col2) + "<=" + str(contcont[i][1][0][0]) + " AND " + str(col1) + ">=" + str(x1) + " AND " + str(col1) + "<=" + str(x2) + " THEN ")
  op+=("(("+col1 + "-" + x1+")*" + z2 + "+(" + x2 + "-" + col1 + ")*" + z1 + ")/(" + x2 + "-" + x1 + ") ELSE ")
 
 for i in range(len(contcont)-1):
  x1 = str(contcont[i][0])
  x2 = str(contcont[i+1][0])
  z1 = str(contcont[i][1][-1][1])
  z2 = str(contcont[i+1][1][-1][1])
  op+=("IF " + str(col2) + ">=" + str(contcont[i][1][-1][0]) + " AND " + str(col1) + ">=" + str(x1) + " AND " + str(col1) + "<=" + str(x2) + " THEN ")
  op+=("(("+col1 + "-" + x1+")*" + z2 + "+(" + x2 + "-" + col1 + ")*" + z1 + ")/(" + x2 + "-" + x1 + ") ELSE ")
 
 for i in range(len(contcont[0][1])-1):
  y1 = str(contcont[0][1][i][0])
  y2 = str(contcont[0][1][i+1][0])
  z1 = str(contcont[0][1][i][1])
  z2 = str(contcont[0][1][i+1][1])
  op+=("IF " + str(col1) + "<=" + str(contcont[0][0]) + " AND " + str(col2) + ">=" + str(y1) + " AND " + str(col2) + "<=" + str(y2) + " THEN ")
  op+=("(("+col2 + "-" + y1+")*" + z2 + "+(" + y2 + "-" + col2 + ")*" + z1 + ")/(" + y2 + "-" + y1 + ") ELSE ")
 
 for i in range(len(contcont[0][1])-1):
  y1 = str(contcont[-1][1][i][0])
  y2 = str(contcont[-1][1][i+1][0])
  z1 = str(contcont[-1][1][i][1])
  z2 = str(contcont[-1][1][i+1][1])
  op+=("IF " + str(col1) + ">=" + str(contcont[-1][0]) + " AND " + str(col2) + ">=" + str(y1) + " AND " + str(col2) + "<=" + str(y2) + " THEN ")
  op+=("(("+col2 + "-" + y1+")*" + z2 + "+(" + y2 + "-" + col2 + ")*" + z1 + ")/(" + y2 + "-" + y1 + ") ELSE ")
 
 #INTERIOR
 
 for i in range(len(contcont)-1):
  x1 = str(contcont[i][0])
  x2 = str(contcont[i+1][0])
  for j in range(len(contcont[i][1])-1):
   y1 = str(contcont[0][1][j][0])
   y2 = str(contcont[0][1][j+1][0])
   z11 = str(contcont[i][1][j][1])
   z12 = str(contcont[i][1][j+1][1])
   z21 = str(contcont[i+1][1][j][1])
   z22 = str(contcont[i+1][1][j+1][1])
   op+=("IF " + col1 + ">=" + x1 + " AND " + col1 + "<=" + x2 + " AND " + col2 + ">=" + y1 + " AND " + col2 + "<=" + y2 + " THEN ")
   op+=("(("+col1+"-"+x1+")*("+col2+"-"+y1+")*"+z22+" + ("+col1+"-"+x1+")*("+y2+"-"+col2+")*"+z21+" + ("+x2+"-"+col1+")*("+col2+"-"+y1+")*"+z12+" + ("+x2+"-"+col1+")*("+y2+"-"+col2+")*"+z11+")/(("+x2+"-"+x1+")*("+y2+"-"+y1+")) ELSE ")
 op+=str(defaultValue)
 return op

def radify_model(model, filename="rad_model.txt", logistic=False):
 lines = ["BASE_VALUE = " + str(model["BASE_VALUE"]) + ";"]
 combLine = "COMB = BASE_VALUE"
 
 fc="*"
 defaultValue=1
 if "featcomb" in model:
  if model["featcomb"]=="addl":
   fc="+"
   defaultValue=0
 
 if "cats" in model:
  for col in model["cats"]:
   cat = model["cats"][col]
   lines.append(col+"___RELA = "+radify_cat(cat, col)+";")
   combLine+=(fc+col+"___RELA")
 
 if "conts" in model:
  for col in model["conts"]:
   cont = model["conts"][col]
   lines.append(col+"___RELA = "+radify_cont(cont, col, defaultValue)+";")
   combLine+=(fc+col+"___RELA")
 
 if "catcats" in model:
  for cols in model["catcats"]:
   col1, col2 = cols.split(" X ")
   lines.append(col1+"__X__"+col2+"___RELA = "+radify_catcat(model, cols)+";")
   combLine+=(fc+col1+"__X__"+col2)
 
 if "catconts" in model:
  for cols in model["catconts"]:
   col1, col2 = cols.split(" X ")
   lines.append(col1+"__X__"+col2+"___RELA = "+radify_catcont(model, cols, defaultValue)+";")
   combLine+=(fc+col1+"__X__"+col2)
 
 if "contconts" in model:
  print("WARNING: radification of cont-cont interactions is untested at time of writing; caveat!")
  for cols in model["contconts"]:
   col1, col2 = cols.split(" X ")
   lines.append(col1+"__X__"+col2+"___RELA = "+radify_contcont(model, cols, defaultValue)+";")
   combLine+=(fc+col1+"__X__"+col2)
 
 
 combLine+=";"
 lines.append(combLine)
 
 
 if logistic:
  lines.append("PREDICTION = 1/(1+EXP(-PREDICTION));")
 else:
  lines.append("PREDICTION = COMB;")
 
 #Write to file
 
 f = open(filename, "w")
 for line in lines:
  print(line)
  f.write(line+'\n')
 f.close()

if __name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]], "cont3":[[12,1.2],[13, 0.9],[14, 1.1]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.01},"cat2":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, 'catcats':{'cat1 X cat2':{"uniques":{"wstfgl":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}, "florpalorp":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, "OTHER":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}}, 'catconts':{"cat1 X cont1":{"uniques":{"wstfgl":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "florpalorp":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}, "OTHER":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}}, 'contconts':{'cont1 X cont2':[[0.01, [[37,1.2],[98, 0.9]]],[0.02,[[37,1.2],[98, 0.9]]], [0.03, [[37,1.2],[98, 9000]]]]}}
 
 radify_model(exampleModel, "rm.txt")
