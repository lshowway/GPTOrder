import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the first table
mgsm_en = {
    'Models': ['RTP', 'CS', 'BF', 'Loop'],
    'original': [0.30, 0.45, 0.63, 0.56],
    'ex F&L': [0.28, 0.42, 0.42, 0.50],
    'ex two': [0.27, 0.45, 0.46, 0.44],
    'ex adj': [0.26, 0.45, 0.39, 0.38],
    'fix F&L': [0.23, 0.47, 0.36, 0.33]
}
avg_1 = [0.28, 0.27, 0.26, 0.23]
avg_2 = [0.42, 0.45, 0.45, 0.47]
avg_3 = [0.42, 0.46, 0.39, 0.36]
avg_4 = [0.50, 0.44, 0.38, 0.33]

print(sum(avg_1) / 4 / 0.30 - 1.0)
print(sum(avg_2) / 4 / 0.45 - 1.0)
print(sum(avg_3) / 4 / 0.63 - 1.0)
print(sum(avg_4) / 4 / 0.56 - 1.0)

# Set the seaborn color palette
sns.set_palette("Set2")

# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 1, figsize=(8, 3))

# Plot for the first table
x = np.arange(len(mgsm_en['Models']))
width = 0.15

# Set the bar styles for each strategy
bar_styles = ['', '/////', 'xxxxx', '+++++', '']

# Set the bar colors for each strategy
bar_colors = ['peru', 'white', 'white', 'white', 'sienna']

# Set the bar line color for each strategy
bar_line_colors = ['white', 'chocolate', 'sandybrown', 'peachpuff', 'white']

for i, strategy in enumerate(['original', 'ex F&L', 'ex two', 'ex adj', 'fix F&L']):
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




plt.tight_layout()
plt.savefig('generation.pdf', dpi=500)

# Show the plots
plt.show()