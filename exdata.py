import os

a=os.listdir('./train_data')

with open('ex_data.txt','a') as ex:
    for i in range(len(a)):
        if a[i][-4:]=='.txt':
            with open('./train_data/'+a[i],'r') as d:
                data=d.readlines()
                ex.writelines(data)
                ex.writelines('\n')
