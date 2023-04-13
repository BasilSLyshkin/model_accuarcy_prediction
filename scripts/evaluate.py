from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show(data, N= 100):
    final = pd.DataFrame(index = list(range(N)), columns = ['acc'])
    data['Accuracy'] = (data['prediction'] == data['target']).astype('int')
    models = ['LogGDA','LogGMA_1','LogGMA_2','LogGMA_3','LogGMA_4','LogGMA_5']
    data['LogEnsemble'] = data[models].var(1)
    models.append('LogEnsemble')
    fig, ax = plt.subplots(4,2, figsize = (20, 15))
    fig.tight_layout() 
    for k, model in enumerate(models):
        draw_one(data, model, ax[k//2][k%2], N)
    ax[3,1].axis('off')
    
def draw_one(data, name, axis, N):
    final = pd.DataFrame(index = list(range(N)), columns = [name])
    for i in range(N):
        period = [data[name].quantile(i/N),data[name].quantile((i+1)/N)]
        cut = data.loc[(period[0]<=data[name]) & (data[name] <period[1]), 'Accuracy'].mean()
        final.loc[i:, name] = cut
        
    lr = LinearRegression()
    X = np.arange(N).reshape(-1,1)
    lr.fit(X,final[name])
    final['linear'] = lr.predict(X)
    axis.plot(final.index, final['linear'],color = 'grey' , linestyle= '--')
    axis.plot(final.index, final[name],color = 'dodgerblue')
    coef = lr.coef_[0]
    R2 = r2_score(final['linear'], final[name])
    axis.set_title(f'{name}, R2 = {round(R2, 2)}, coef = {round(coef,4)}')

def show_agg(data):
    data['loss'] = data['prediction'] == data['target']
    data['month'] = data['date'].apply(lambda x: x[:-3])
    pivot = data.pivot_table(index=['month'],
                    values=['loss', 'LogEnsemble'],
                    aggfunc=[np.mean])
    x = pivot[('mean','loss')]
    y = pivot[('mean', 'LogEnsemble')]
    lr = LinearRegression()
    X = np.array(x).reshape(-1,1).reshape(-1,1)
    print(X)
    lr.fit(X,y)
    linear = lr.predict(x)
    plt.figure(figsize = (7,5))
    plt.scatter(x,y, color = 'dodgerblue')
    plt.plot(x, linear, color = 'grey' , linestyle= '--')
    plt.ylabel('LogEnsemle Uncertainty')
    plt.xlabel('Accuracy')
    plt.title('Аггрегация по месяцам', y = 1.08)