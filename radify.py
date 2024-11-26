def radify_cat(model, col):
 op = col+"___MULTIPLIER = "
 for u in model["cats"][col]["uniques"]:
  op += ("IF " + col + "='" + u + "' THEN " +  str(model["cats"][col]["uniques"][u]) + " ELSE ")
 op += (str(model["cats"][col]["OTHER"]) + ";")
 return op

def radify_cont(model, col):
 op = col+"___MULTIPLIER = IF " + col + "<=" + str(model["conts"][col][0][0]) + " THEN " + str(model["conts"][col][0][1]) + " ELSE "
 for i in range(len(model["conts"][col])-1):
  p1 = str(model["conts"][col][i][0])
  p2 = str(model["conts"][col][i+1][0])
  y1 = str(model["conts"][col][i][1])
  y2 = str(model["conts"][col][i+1][1])
  op += ("IF " + col +  "<=" + p2 + " THEN " + "(("+col + "-" + p1+")*" + y1 + "+(" + p2 + "-" + col + ")*" + y2 + ")/(" + p2 + "-" + p1 + ") ELSE ") #((x-p1)y1 + (p2-x)y2) / (p2 - p1)
 op += (str(model["conts"][col][-1][1]) + ";")
 return op

def radify_model(model, filename="rad_model.txt"):
 lines = ["BASE_VALUE = " + str(model["BASE_VALUE"]) + ";"]
 finalLine = "PREDICTION = BASE_VALUE"
 
 if "cats" in model:
  for col in model["cats"]:
   lines.append(radify_cat(model, col))
   finalLine+=("*"+col+"___MULTIPLIER")
 
 if "conts" in model:
  for col in model["conts"]:
   lines.append(radify_cont(model, col))
   finalLine+=("*"+col+"___MULTIPLIER")
 
 lines.append(finalLine)
 
 #Write to file
 
 f = open(filename, "w")
 for line in lines:
  #print(line)
  f.write(line+'\n')
 f.close()


if __name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]], "cont3":[[12,1.2],[13, 0.9],[14, 1.1]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.01},"cat2":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, 'catcats':{'cat1 X cat2':{"uniques":{"wstfgl":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}, "florpalorp":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}, "OTHER":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}}, 'catconts':{"cat1 X cont1":{"uniques":{"wstfgl":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "florpalorp":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}, "OTHER":[[0.01, 1],[0.02,1.1], [0.03, 1.06]]}}, 'contconts':{'cont1 X cont2':[[0.01, [[37,1.2],[98, 0.9]]],[0.02,[[37,1.2],[98, 0.9]]], [0.03, [[37,1.2],[98, 9000]]]]}}
 
 radify_model(exampleModel, "rm.txt")
