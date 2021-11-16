import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
# https://stackoverflow.com/questions/11623056/matplotlib-using-a-colormap-to-color-table-cell-background
cm = sns.light_palette("green", as_cmap=True)

df = pd.read_csv('data/sum_human_en.tsv', sep='\t', header=None, index_col=0)
vals = np.around(df.values,2)
norm = plt.Normalize(vals.min()-5, vals.max()+5)
colours = plt.cm.Greens(norm(vals))
#8,5
fig = plt.figure(figsize=(8,1))
ax = fig.add_subplot(311, frameon=False, xticks=[], yticks=[])

the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=['1st', '2nd', '3rd','4th','5th'], 
                    colWidths = [0.03]*vals.shape[1], loc='top', 
                    cellColours=colours, cellLoc='center')
the_table.scale(2,2)
plt.text(0.29, 9.15, 'English', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='bold')

df = pd.read_csv('data/sum_human_ca.tsv', sep='\t', header=None, index_col=0)
vals = np.around(df.values,2)
norm = plt.Normalize(vals.min()-5, vals.max()+5)
colours = plt.cm.Greens(norm(vals))

ax = fig.add_subplot(311, frameon=False, xticks=[], yticks=[])

the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=['1st', '2nd', '3rd','4th','5th'], 
                    colWidths = [0.03]*vals.shape[1], loc='bottom', 
                    cellColours=colours, cellLoc='center')
the_table.scale(2,2)
plt.text(0.29, -0.65, 'Catalan', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='bold')

plt.savefig('figures/sum_human_eval.pdf',dpi=200, bbox_inches='tight')
plt.close()