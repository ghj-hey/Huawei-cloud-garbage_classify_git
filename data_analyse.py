import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
from matplotlib.pyplot import MultipleLocator

with open('./dataset/et_71.txt','r',encoding='utf-8') as l:
    data=l.readlines()

y_data=[0]*43
x_data=[i for i in range(43)]
for d in data:
    y_data[int(d.split(',')[1].split('\n')[0])]+=1
# 构建数据
# x_data=sorted([int(key) for key in dataset.keys()])
# y_data =[dataset[str(i)] for i in x_data]
num = np.sum(np.array(y_data))
#
#
print(x_data,y_data)




plt.figure(figsize = (30,24))
#
plt.bar(x=x_data, height=y_data, width=0.9,label='total nums %s'%(str(num)),color='steelblue', alpha=0.8)
#
for x, y in enumerate(y_data):
    plt.text(x, y, '%s' % y, ha='center', va='bottom',fontsize=20)
    # plt.text(x, y, '%s' % x, ha='center', va='top',fontsize=20)

plt.xticks([index  for index in x_data], x_data)


plt.title('The distribution of all dataset',fontsize=20)

plt.xlabel('labels',fontsize=20)
plt.ylabel("numbers",fontsize=20)

plt.legend()
plt.savefig('All-data.png')
plt.show()