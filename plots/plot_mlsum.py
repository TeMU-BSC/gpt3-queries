import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
plt.rc('font', **font)

# x axis
x_axis_dict = {'ada': math.log(350), 'babbage' : math.log(1300), 'curie' : math.log(6700), 'davinci' : math.log(175000)}
#PLOT MSLUM

df = pd.read_csv('data/mlsum.tsv', sep='\t')
df['x_axis'] = df['Model'].map(x_axis_dict) 
df['x_axis'] = df['x_axis'].apply(lambda x: round(x,0))
sns.lineplot(data=df[df['Metric']=='F1'], x='x_axis', y='Score', hue='Language', palette='muted', style='Language', markers=True)

ax = plt.gca()
ax.set_ylim([0.00, 0.5])
ax.set_xlim([5.5, 12.5])
ax.set(xlabel = 'Model', ylabel='ROUGE F1', xticklabels=['', 'ada','babbage','','curie','','','davinci'])
plt.legend(loc='best')
plt.savefig('figures/mlsum_scores_f1.pdf')
plt.close()

#T'ho deixo escrit per dema: pels grafics fes servir aixo: 
#Ada, Babbage, Curie and Davinci line up closely with 350M, 1.3B, 6.7B, and 175B,
#agafant el logartime (import math; math.log() de 350M, 1.3B, etc)