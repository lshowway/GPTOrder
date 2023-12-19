import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
exp_models = ['ChatGPT', 'Claude', r'LLaMA$_{7B}$', r'LLaMA$_{13B}$', r'LLaMA$_{30B}$']

wino_en = {
    'Models': exp_models,
    'original': [0.88, 0.84, 0.50, 0.51, 0.54],
    'ex_random_two': [0.72, 0.70, 0.43, 0.57, 0.51],
    'rotate_two_part': [0.69, 0.62, 0.47, 0.52, 0.54],
    'ex_adjacent': [0.56, 0.47, 0.44, 0.56, 0.40]
}

wino_zh = {
    'Models': exp_models,
    'original': [0.82, 0.79, 0.57, 0.51, 0.50],
    'ex_random_two': [0.74, 0.75, 0.47, 0.50, 0.54],
    'rotate_two_part': [0.69, 0.46, 0.56, 0.54, 0.48],
    'ex_adjacent': [0.57, 0.18, 0.53, 0.47, 0.52]
}

wino_fr = {
    'Models': exp_models,
    'original': [0.84, 0.74, 0.53, 0.51, 0.58],
    'ex_random_two': [0.64, 0.48, 0.52, 0.43, 0.62],
    'rotate_two_part': [0.62, 0.13, 0.50, 0.40, 0.55],
    'ex_adjacent': [0.56, 0.32, 0.42, 0.44, 0.48]
}


# Set the seaborn color palette
sns.set_palette("pastel")

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot for the first table
x = np.arange(len(wino_en['Models']))
width = 0.18

# Set the bar styles for each strategy
bar_styles = ['//////', '', '', '', ]

# Set the bar colors for each strategy
bar_colors = ['linen', 'lightsteelblue', 'white', 'mistyrose', ]

# Set the bar line color for each strategy
bar_line_colors = ['black', 'black',  'black', 'black', ]


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[0].bar(x + (i * width), wino_en[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[0].set_ylabel('Accuracy', fontsize=14)
axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(wino_en['Models'], fontsize=16)
axes[0].set_title('English', fontsize=16, loc='right', y=0.65)
axes[0].spines['top'].set_visible(False)

for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[1].bar(x + (i * width), wino_zh[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(wino_zh['Models'], fontsize=16)
axes[1].set_title('Chinese', fontsize=16, loc='right', y=0.75)
# axes[1].spines['top'].set_visible(False)


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[2].bar(x + (i * width), wino_fr[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[2].set_ylabel('Accuracy', fontsize=14)
axes[2].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[2].set_xticks(x + width * 1.5)
axes[2].set_xticklabels(wino_fr['Models'], fontsize=16)
axes[2].set_title('French', fontsize=16, loc='right', y=0.75)
# axes[2].spines['top'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.01), fontsize=11)
fig.tight_layout()

# Show the plots
plt.savefig('xwinogrande.pdf', dpi=800)
plt.show()
