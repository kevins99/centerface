import json 
output = []
with open('wider_face_train_bbx_gt.txt') as f:
	name = f.readline().strip()
	while name:
		print(name)
		name = name.split("/")[1]
		count = int(f.readline().strip())
		i = 0
		annots = []
		while i<count:
			annot = f.readline().strip()
			annot = annot.split()[:4]
			i+=1
			annots.append(annot)
		output.append((name,count,annots))
		print(output[-1])
		name = f.readline().strip()
	print(len(output))