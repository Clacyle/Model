def distribution(lst):
  dic={}
  n=len(lst)
  
  for element in lst:
    if element in dic:
      dic[element]+=1
    else:
      dic[element]=1
    
  return { v: o/n for v, o in dic.items() }
  