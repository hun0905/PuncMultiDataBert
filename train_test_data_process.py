seqs = open('data/train.txt', encoding='utf8',errors='ignore').read()
file = open("data_after_process/train.txt", "w", encoding='utf8',errors='ignore')
End = True
count = 0
for c in seqs.split():
    if End == True:
        file.write(c)
        if (count+1) % 60 == 0:
            if c != '.PERIOD' and c!='?QUESTIONMARK' :
                End = False
            file.write('\n')
        else:
            file.write(' ')   
        count+=1   
    if c == '.PERIOD' or c=='?QUESTIONMARK':
        End = True