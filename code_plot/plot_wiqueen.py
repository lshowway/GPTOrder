import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
exp_models = ['ChatGPT', 'Claude', r'LLaMA$_{7B}$', r'LLaMA$_{13B}$', r'LLaMA$_{30B}$']

wiqueen_en = {
    'Models': exp_models,
    'original': [0.83, 0.82, 0.70, 0.76, 0.80],
    'ex_random_two': [0.77, 0.73, 0.65, 0.70, 0.74],
    'rotate_two_part': [0.77, 0.72, 0.66, 0.67, 0.75],
    'ex_adjacent': [0.78, 0.68, 0.65, 0.67, 0.70]
}

wiqueen_fr = {
    'Models': exp_models,
    'original': [0.63, 0.67, 0.43, 0.50, 0.59],
    'ex_random_two': [0.55, 0.61, 0.43, 0.42, 0.51],
    'rotate_two_part': [0.50, 0.54, 0.43, 0.44, 0.49],
    'ex_adjacent': [0.51, 0.51, 0.44, 0.45, 0.49]
}

# Set the seaborn color palette
sns.set_palette("pastel")

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot for the first table
x = np.arange(len(wiqueen_en['Models']))
width = 0.18

# Set the bar styles for each strategy
bar_styles = ['//////', '', '', '', ]

# Set the bar colors for each strategy
bar_colors = ['linen', 'lightsteelblue', 'white', 'mistyrose', ]

# Set the bar line color for each strategy
bar_line_colors = ['black', 'black',  'black', 'black', ]


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[0].bar(x + (i * width), wiqueen_en[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[0].set_ylabel('Precision@1', fontsize=14)
axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(wiqueen_en['Models'], fontsize=16)
axes[0].spines['top'].set_visible(False)



for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[1].bar(x + (i * width), wiqueen_fr[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[1].set_ylabel('Precision@1', fontsize=14)
axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(wiqueen_fr['Models'], fontsize=16)

# Adjust the layout and add legend

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.0), fontsize=11)
fig.tight_layout()

# Show the plots
plt.savefig('wiqueen.pdf', dpi=800)
plt.show()
