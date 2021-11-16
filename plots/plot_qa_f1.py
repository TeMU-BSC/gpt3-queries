import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
sns.set_style("white")
font = {'family' : 'Sans Serif',
        'size'   : 12}

palette = sns.set_palette('muted', n_colors=6)

labels=['ca', 'de', 'en', 'es', 'tu', 'ru']

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


baselines = [0.6715,0.706,0.835,0.755,0.713,0.554]
colors = sns.color_palette(palette)
styles = ['dotted','densely dashdotted','densely dashdotdotted','dashdot','solid','dashed']
linestyles = zip(baselines,colors,styles)
#Add baselines
for baseline, color, style in linestyles:
        plt.axhline(baseline, c = color,linestyle = linestyles_dict[style])
plt.text(5.1, 0.88, "Baseline", ha="left", va="top")


plt.rc('font', **font)
# x axis
x_axis_dict = {'ada': math.log(3500000, 10), 'babbage' : math.log(130000000, 10), 'curie' : math.log(670000000, 10), 'davinci' : math.log(17500000000, 10)}

#PLOT QA F1SCORE
df = pd.read_csv('data/qa_F1.tsv', sep='\t')
df['x_axis'] = df['Model'].map(x_axis_dict) 
sns.lineplot(data=df, x='x_axis', y='Score', hue='Language', style='Language',markers=True, palette=palette, hue_order=labels)

#PLOT QA EM
#exact_match = pd.read_csv('data/qa_EM.tsv', sep='\t')
#sns.pointplot(data=exact_match, x='Model', y='Score', hue='Language', markers='^', dodge=0.7, color='black', linestyles='',  facet_kws={'legend_out': True})#, axx2)
ax = plt.gca()
ax2 = ax.twiny()

ax.set_ylim([0.00, 0.9])
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

ax.set_ylabel('QA F1', weight='bold')

ax.legend(loc='lower left')
plt.savefig('figures/qa_f1_baseline.pdf')
plt.close()