import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
exp_models = ['ChatGPT', 'Claude', r'LLaMA$_{7B}$', r'LLaMA$_{13B}$', r'LLaMA$_{30B}$']

mgsm_en = {
    'Models': exp_models,
    'original': [0.54, 0.32, 0.15, 0.18, 0.48],
    'ex_random_two': [0.50, 0.17, 0.08, 0.25, 0.45],
    'rotate_two_part': [0.37, 0.48, 0.13, 0.18, 0.39],
    'ex_adjacent': [0.17, 0.15, 0.07, 0.15, 0.26]
}

# Data for the second table
mgsm_zh = {
    'Models': exp_models,
    'original': [0.25, 0.29, 0.06, 0.1, 0.04],
    'ex_random_two': [0.14, 0.11, 0.06, 0.04, 0.06],
    'rotate_two_part': [0.09, 0.05, 0.02, 0.02, 0.03],
    'ex_adjacent': [0.07, 0.07, 0.05, 0.3, 0.06]
}

mgsm_fr = {
    'Models': exp_models,
    'original': [0.46, 0.31, 0.16, 0.21, 0.25],
    'ex_random_two': [0.37, 0.24, 0.13, 0.20, 0.28],
    'rotate_two_part': [0.37, 0.38, 0.05, 0.13, 0.14],
    'ex_adjacent': [0.17, 0.03, 0.07, 0.09, 0.15]
}


# Set the seaborn color palette
sns.set_palette("pastel")

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot for the first table
x = np.arange(len(mgsm_en['Models']))
width = 0.18

# Set the bar styles for each strategy
bar_styles = ['//////', '', '', '', ]

# Set the bar colors for each strategy
bar_colors = ['linen', 'lightsteelblue', 'white', 'mistyrose', ]

# Set the bar line color for each strategy
bar_line_colors = ['black', 'black',  'black', 'black', ]


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[0].bar(x + (i * width), mgsm_en[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[0].set_ylabel('Major@k', fontsize=14)
axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(mgsm_en['Models'], fontsize=16)
axes[0].set_title('English', fontsize=16, loc='right', y=0.65)
axes[0].spines['top'].set_visible(False)

for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[1].bar(x + (i * width), mgsm_zh[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[1].set_ylabel('Major@k', fontsize=14)
axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(mgsm_zh['Models'], fontsize=16)
axes[1].set_title('Chinese', fontsize=16, loc='right', y=0.75)


for i, strategy in enumerate(['original', 'ex_random_two', 'rotate_two_part', 'ex_adjacent']):
    axes[2].bar(x + (i * width), mgsm_fr[strategy], width, label=strategy,
                hatch=bar_styles[i],
                color=bar_colors[i],
                edgecolor=bar_line_colors[i]
                )
axes[2].set_ylabel('Major@k', fontsize=14)
axes[2].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axes[2].set_xticks(x + width * 1.5)
axes[2].set_xticklabels(mgsm_fr['Models'], fontsize=16)
axes[2].set_title('French', fontsize=16, loc='right', y=0.75)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.01), fontsize=11)
fig.tight_layout()

# Show the plots
plt.savefig('mgsm.pdf', dpi=800)
plt.show()
