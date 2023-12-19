import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
mgsm_en = {
    'Models': ['RTP', 'CS', 'BF', 'Loop'],
    'ex F&L': [0.62,  0.61, 0.78, 0.67],
    'ex two': [0.54,  0.59, 0.73, 0.67],
    'ex adj': [0.53, 0.56, 0.64, 0.32],
    'fix F&L': [0.50, 0.53, 0.57, 0.02],
    # 'match 1': [0.02,  0.62, 0.61, 0.06],

}

# Set the seaborn color palette
sns.set_palette("Set2")

# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 1, figsize=(8, 3))

# Plot for the first table
x = np.arange(len(mgsm_en['Models']))
width = 0.15

# Set the bar styles for each strategy
# bar_styles = ['', '', '', '', 'xxxxx', ]
bar_styles = ['//', '\\', '', '', 'xxxxx', ]

# Set the bar colors for each strategy
# bar_colors = ['mistyrose', 'linen', 'white', 'lightsteelblue', 'white', ]
bar_colors = ['mistyrose', 'linen', 'white', 'lightsteelblue']

# Set the bar line color for each strategy
# bar_line_colors = ['black', 'black', 'black', 'black', 'black', ]
bar_line_colors = ['black', 'black', 'black', 'black', ]

# for i, strategy in enumerate(['ex F&L', 'ex two', 'ex adj', 'fix F&L', 'match 1',]):
for i, strategy in enumerate(['ex F&L', 'ex two', 'ex adj', 'fix F&L', ]):
    axes.bar(x + (i * width), mgsm_en[strategy], width, label=strategy,
             hatch=bar_styles[i],
             color=bar_colors[i],
             edgecolor=bar_line_colors[i])

axes.set_xticks(x + width * 2)
axes.set_xticklabels(mgsm_en['Models'], fontsize=12)
# Add a horizontally arranged legend
axes.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)


# Adjust the layout
# fig.suptitle("MGSM", fontsize=20)
# Remove the top and right edges

plt.tight_layout()
plt.savefig('reorder.pdf', dpi=500)

# Show the plots
plt.show()