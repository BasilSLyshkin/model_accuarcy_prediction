import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
plt.style.use('bmh')

def process(dataset):
    tmp1 = dataset.map(remove_columns=['package_name'])
    tmp2 = tmp1.map(date)
    tmp3 = tmp2.map(target)
    tmp4 = tmp3.map(split)
    return tmp4

def date(example):
    s = example['date']
    split = s.split()
    month_dict = {
         'January': '01',
         'February': '02',
         'March': '03',
         'April': '04',
         'May': '05',
         'June': '06',
         'July': '07',
         'August': '08',
         'September': '09',
         'October': '10',
         'November': '11',
         'December': '12'
    }
    example['date'] = f'{split[2]}-{month_dict[split[0]]}-{split[1]}'
    return example

def target(example):
    example['star'] = int(example['star']>4)
    return example

def split(example):
    s = example['date']
    if s<'2016-12-04':
        example['sample'] = 'test'
    elif s<'2016-12-31':
        example['sample'] = 'validation'
    else:
        example['sample'] = 'train'
    return example



    
def plot_data(dataset):

    df = pd.DataFrame({
        'date':dataset['date'],
        'star':dataset['star']
    })
    fig, ax = plt.subplots(1,2,figsize = (13,5))
    old_target = df.groupby('star').size()
    old_target.plot(kind = 'bar', ax = ax[0], title = 'Old target')
    df['new_star'] = (df['star']>4).astype('int')
    new_target = df.groupby('new_star').size()
    new_target.plot(kind = 'bar', ax = ax[1], title = 'New target')
    
    df['date'] = df['date'].apply(func)
    df['sample'] = df['date'].apply(sample)
    df['color'] = df['sample'].apply(color)
    df['new_star'] = (df['star']>4).astype('int')
    df['month'] = df['date'].apply(lambda s:s[:7])
    group = df.groupby('month')
    fig, ax1 = plt.subplots(figsize = (13,5))
    colors = group['color'].max().values
    group.size().plot(kind = 'bar', ax = ax1, alpha = 0.7, color = colors)

    ax1.set_ylabel('Number of texts')
    ax2 = ax1.twinx()
    group['new_star'].apply(lambda a: a.sum()/a.count()).plot(kind = 'line', ax = ax2, color = 'black', alpha= 0.5)
    ax2.set_ylabel('Proportion of a target')

    legend_elements = [Patch(facecolor='green', label='Test'),
                       Patch(facecolor='orange', label='Validation'),
                       Patch(facecolor='blue', label='Train')]

    # Create the figure
    ax1.legend(handles=legend_elements)




def func(s):
    split = s.split()
    month_dict = {
         'January': '01',
         'February': '02',
         'March': '03',
         'April': '04',
         'May': '05',
         'June': '06',
         'July': '07',
         'August': '08',
         'September': '09',
         'October': '10',
         'November': '11',
         'December': '12'
    }
    return f'{split[2]}-{month_dict[split[0]]}-{split[1]}'

def sample(s):
    if s<'2016-12-04':
        return 'test'
    elif s<'2016-12-31':
        return 'validation'
    else:
        return 'train'

def color(s):
    if s == 'train':
        return 'blue'
    elif s=='validation':
        return 'orange'
    else:
        return 'green'
