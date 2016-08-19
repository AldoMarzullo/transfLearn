import os
import json
from collections import OrderedDict

f_in = open("labels.txt","r")
content = f_in.readlines()
content.sort()
f_in.close()

data = []
it = 0

sort_order = ['x1', 'y1', 'x2', 'y2']


for i in sorted(os.listdir(os.getcwd())):
    if i.endswith(".jpg"): 
    	element = {}
    	target = [{
		'y1' : 	int(content[it].split(" ")[1]) - 16,
		'x1' : 	int(content[it].split(" ")[2].rstrip()) - 16,
		'y2' : 	int(content[it].split(" ")[1]) + 16,
		'x2' : 	int(content[it].split(" ")[2].rstrip()) + 16
	}]
	target = [OrderedDict(sorted(item.iteritems(), key=lambda (k, v): sort_order.index(k))) for item in target]
        element = {"rects" : target, "image_path" : 'stare_test/' + i}
        data.append(element)
        it = it + 1

out = open("../stare_test.json","w")
out.write(json.dumps(data))
out.close()
