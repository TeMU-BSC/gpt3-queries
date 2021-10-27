import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
font = {'family' : 'Sans Serif',
        'size'   : 15}
plt.rc('font', **font)

#PLOT MSLUM
df = pd.read_csv('mlsum.tsv', sep='\t')
sns.barplot(data=df, x='Model', y='Score', hue='Language', palette='rocket')
plt.legend(loc='upper right')
plt.savefig('plot_mlsum_scores.png')
plt.close()