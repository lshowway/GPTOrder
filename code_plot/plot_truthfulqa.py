import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
exp_models = ['ChatGPT', 'Claude', r'LLaMA$_{7B}$', r'LLaMA$_{13B}$', r'LLaMA$_{30B}$']

info = {
    'Models': exp_models,
    'original': [0.7073, 0.5416, 0.6747, 0.6482, 0.6104],
    'ex_random_two': [0.6411, 0.3877, 0.5967, 0.5756, 0.5539],
    'rotate_two_part': [0.5191, 0.4986, 0.4971, 0.4769, 0.4674],
    'ex_adjacent': [0.5504, 0.2587, 0.4970, 0.5007, 0.4487]
}

judge = {
    'Models': exp_models,
    'original': [0.4718, 0.4955, 0.5344, 0.4639, 0.4725],
    'ex_random_two': [0.5146, 0.5981, 0.5357, 0.5050, 0.5132],
    'rotate_two_part': [0.5574, 0.5389, 0.5438, 0.5539, 0.5582],
    'ex_adjacent': [0.5726, 0.6706, 0.5695, 0.5692, 0.5860]
}

# Set the seaborn color palette
sns.set_palette("pastel")

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot for the first table
x = np.arange(len(info['Models']))
width = 0.18

# Set the bar styles for each strategy
bar_styles = ['//////', '', '', '', ]

# Set the bar colors for each strategy
bar_colors = ['linen', 'lightsteelblue', 'white', 'mistyrose', ]

# Set the bar line color for each strategy
bar_line_colors = ['black', 'black',  'black', 'black', ]


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[0].bar(x + (i * width), judge[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[0].set_ylabel('GPT-judge', fontsize=14)
axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(judge['Models'], fontsize=16)
axes[0].spines['top'].set_visible(False)

for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[1].bar(x + (i * width), info[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[1].set_ylabel('GPT-info', fontsize=14)
axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(info['Models'], fontsize=16)

# Adjust the layout and add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.0), fontsize=11)
fig.tight_layout()

# Show the plots
plt.savefig('truthfulqa.pdf', dpi=800)
plt.show()
