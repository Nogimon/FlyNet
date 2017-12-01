import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_csv('/media/zlab-1/Data/Lian/keras/peak_data.csv',index_col=0)


n=pd.DataFrame(columns=['mean','std'])

t=['larva','pupa','adult']

for i in t:
    for j in ['highgt','highpd','lowgt','lowpd','rategt','ratepd','ratiogt','ratiopd']:
        a=np.asarray([df[i][j].mean(),df[i][j].std()]).reshape((1,-1))
        d=pd.DataFrame(a,index=[i+'_'+j], columns=['mean','std'])  
        n=pd.concat([n,d])
n.to_csv('/media/zlab-1/Data/Lian/keras/n.csv',index=True)
        
def plotbar(datagt, datapd, stdgt, stdpd, ylabel):
    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index + bar_width, datagt, bar_width, color = '#3f51b5', yerr = np.multiply(stdgt,0.6), error_kw=error_config, label = 'GroundTruth')
    rects2 = plt.bar(index + 2 * bar_width, datapd, bar_width, color = '#ff7043', yerr = np.multiply(stdpd,0.6), error_kw=error_config, label = 'Prediction')
    plt.xlabel('Developmental Stages')
    plt.ylabel(ylabel)
    plt.xticks(index + 1.5 * bar_width, ('Larva', 'Pupa', 'Adult'))
    plt.legend()
    plt.tight_layout()
    plt.savefig('./resultimage/fig4_'+ylabel+'.png')
#    plt.show()


plotbar([n['mean'][i+'_highgt'] for i in t],[n['mean'][i+'_highpd'] for i in t] , [n['std'][i+'_highgt'] for i in t], [n['std'][i+'_highpd'] for i in t], "End Diastolic Diameter")
plotbar([n['mean'][i+'_lowgt'] for i in t],[n['mean'][i+'_lowpd'] for i in t] , [n['std'][i+'_lowgt'] for i in t], [n['std'][i+'_lowpd'] for i in t], "End Systolic Diameter")
plotbar([n['mean'][i+'_rategt'] for i in t],[n['mean'][i+'_ratepd'] for i in t] , [n['std'][i+'_rategt'] for i in t], [n['std'][i+'_ratepd'] for i in t], "Heart Rate")
plotbar([n['mean'][i+'_ratiogt'] for i in t],[n['mean'][i+'_ratiopd'] for i in t] , [n['std'][i+'_ratiogt'] for i in t], [n['std'][i+'_ratiopd'] for i in t], "Fraction Shortening")


