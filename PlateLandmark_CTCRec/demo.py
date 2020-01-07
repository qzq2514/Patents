import json

with open("name_dict.json","r") as fr:
    dict=eval(fr.read())
    print(dict)
    dict={int(ind):str(val) for ind,val in dict.items()}

for ind,val in dict.items():
    print("{" +str(ind)+",'"+val+"'},")