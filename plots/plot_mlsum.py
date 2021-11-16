import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

sns.set_style("white")
font = {'family' : 'Sans Serif',
        'size'   : 12}
plt.rc('font', **font)
labels=['ca', 'de', 'en', 'es', 'tu']

palette = sns.set_palette('muted', n_colors=5)

linestyles_dict = {'dotted':(0, (1, 1)),
        'densely dotted': (0, (1, 1)),
        'dashed':(0, (5, 5)),
        'dashdotted':(0, (3, 5, 1, 5)),
        'densely dashdotted':(0, (3, 1, 1, 1)),
        'dashdotdotted':(0, (3, 5, 1, 5, 1, 5)),
        'loosely dashdotdotted':(0, (3, 10, 1, 10, 1, 10)),
        'densely dashdotdotted':(0, (3, 1, 1, 1, 1, 1)),
        'dashdot' : '-.',
        'solid' :'-',
        'dashed':'--'
        }


baselines = [-10,0.42,0.354,0.204,0.329]
colors = sns.color_palette(palette)
styles = ['solid','dashed','dotted', 'dashdot','densely dashdotted']
linestyles = zip(baselines,colors,styles)
#Add baselines
for baseline, color, style in linestyles:
        plt.axhline(baseline, c = color,linestyle = linestyles_dict[style])
plt.text(5.1, 0.45, "Baseline", ha="left", va="top")

# x axis
x_axis_dict = {'ada': math.log(3500000, 10), 'babbage' : math.log(130000000, 10), 'curie' : math.log(670000000, 10), 'davinci' : math.log(17500000000, 10)}
#PLOT MSLUM

df = pd.read_csv('data/mlsum.tsv', sep='\t')
df['x_axis'] = df['Model'].map(x_axis_dict) 
sns.lineplot(data=df[df['Metric']=='F1'], x='x_axis', y='Score', hue='Language', palette='muted', style='Language', markers=True, hue_order=labels)

ax = plt.gca()
ax2 = ax.twiny()

ax.set_ylim([0.00, 0.5])
ax.set_xlim([5, 11])
ax.set_xlabel('Parameters',weight = 'bold')
x = [6,11]
ax.set_xticks(np.arange(min(x), max(x)+1, 1))
ax.set_xticklabels(['$10^6$', '$10^7$', '$10^8$', '$10^9$', '$10^{10}$',''])

ax2.set_xlim([5, 11])
ax2.set_xlabel('Model', weight='bold')
ax2.set_xticks(list(x_axis_dict.values()))
ax2.set_xticklabels(['Ada','Babbage','Curie','Davinci'])
ax2.xaxis.tick_top()

ax.set_ylabel('ROUGE1', weight='bold')
ax.legend(loc='best')
plt.savefig('figures/mlsum_scores_f1.png')
plt.close()

#T'ho deixo escrit per dema: pels grafics fes servir aixo: 
#Ada, Babbage, Curie and Davinci line up closely with 350M, 1.3B, 6.7B, and 175B,
#agafant el logartime (import math; math.log() de 350M, 1.3B, etc)