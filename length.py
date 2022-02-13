file = open('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/train.txt','r').readlines()
#print(file)
max = 0
for i in file:
    i = i.replace(' ','')
    print(len(i))
    if len(i) > max:
        max= len(i)
print(max)
