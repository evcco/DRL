import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, text, xy, boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'):
    box = patches.FancyBboxPatch(xy, width=1.5, height=0.5, boxstyle=boxstyle,
                                 facecolor=facecolor, edgecolor=edgecolor)
    ax.add_patch(box)
    ax.text(xy[0] + 0.75, xy[1] + 0.25, text, ha="center", va="center", fontsize=12)

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='black', arrowstyle='->'))

def visualize_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw Actor Network boxes
    draw_box(ax, 'State Input', (0, 4))
    draw_box(ax, 'Hidden Layer 1', (2, 4))
    draw_box(ax, 'Hidden Layer 2', (4, 4))
    draw_box(ax, 'Action Output', (6, 4))

    # Draw arrows for Actor Network
    draw_arrow(ax, (1.5, 4.25), (2, 4.25))
    draw_arrow(ax, (3.5, 4.25), (4, 4.25))
    draw_arrow(ax, (5.5, 4.25), (6, 4.25))

    # Draw Critic Network boxes
    draw_box(ax, 'State Input', (0, 2))
    draw_box(ax, 'Action Input', (2, 2))
    draw_box(ax, 'Concat (State, Action)', (4, 2))
    draw_box(ax, 'Hidden Layer', (6, 2))
    draw_box(ax, 'Q-Value Output', (8, 2))

    # Draw arrows for Critic Network
    draw_arrow(ax, (1.5, 2.25), (2, 2.25))
    draw_arrow(ax, (3.5, 2.25), (4, 2.25))
    draw_arrow(ax, (5.5, 2.25), (6, 2.25))
    draw_arrow(ax, (7.5, 2.25), (8, 2.25))
    
    # Draw connection between Actor and Critic
    draw_arrow(ax, (6.75, 4), (6.75, 2.25))

    # Set limits and hide axes
    ax.set_xlim(-1, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Show the plot
    plt.show()

visualize_architecture()
