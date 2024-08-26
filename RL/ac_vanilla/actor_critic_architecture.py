import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, xy, boxstyle, color):
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            bbox=dict(boxstyle=boxstyle, facecolor=color, alpha=0.5),
            fontsize=12)

fig, ax = plt.subplots(figsize=(10, 6))

# Encoder
draw_box(ax, 'Encoder\n$f_1$', (0.1, 0.8), "round,pad=0.3", 'lightblue')
draw_box(ax, 'Latent Space\n$f_2$', (0.3, 0.8), "round,pad=0.3", 'lightgreen')

# Transition and Decoder
draw_box(ax, 'Transition Model\n$f_3$', (0.5, 0.8), "round,pad=0.3", 'lightblue')
draw_box(ax, 'Latent Space\n$f_4$', (0.7, 0.8), "round,pad=0.3", 'lightgreen')
draw_box(ax, 'Decoder\n$f_5$', (0.9, 0.8), "round,pad=0.3", 'lightblue')

# Priors
draw_box(ax, 'Prior Network\n$g_1$', (0.3, 0.5), "round,pad=0.3", 'lightcoral')
draw_box(ax, 'Prior Network\n$g_2$', (0.7, 0.5), "round,pad=0.3", 'lightcoral')

# Connecting arrows
ax.annotate('', xy=(0.17, 0.8), xytext=(0.1, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.37, 0.8), xytext=(0.3, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.57, 0.8), xytext=(0.5, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.77, 0.8), xytext=(0.7, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.83, 0.8), xytext=(0.9, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('', xy=(0.3, 0.75), xytext=(0.3, 0.55), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.7, 0.75), xytext=(0.7, 0.55), arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis('off')
plt.show()

