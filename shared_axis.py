#%%
import matplotlib.pyplot as plt
import numpy as np

# Create data for demonstration
x = np.linspace(0, 10, 100)
y_data = [np.sin(x), np.cos(x), np.tan(x)]

# Create figure and subplots with shared axes
fig, axes = plt.subplots(3, 3, figsize=(12, 10), 
                        sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

# Plot data in each subplot
for i in range(3):
    for j in range(3):
        axes[i, j].plot(x, y_data[j] * (i+1))
        
        # Only show x labels on bottom row
        if i < 2:
            axes[i, j].tick_params(labelbottom=False)
        else:
            axes[i, j].set_xlabel('X Axis')
        
        # Only show y labels on left column
        if j > 0:
            axes[i, j].tick_params(labelleft=False)
        else:
            axes[i, j].set_ylabel('Y Axis')
        
        # Add secondary y-axis on right column
        if j == 2:
            sec_ax = axes[i, j].twinx()
            sec_ax.set_ylim(axes[i, j].get_ylim())  # Match primary y-axis limits
            if i == 1:  # Example: only label middle one
                sec_ax.set_ylabel('Secondary Y Axis')
            else:
                sec_ax.tick_params(labelright=False)

# Adjust layout to prevent label overlap
plt.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 3, hspace=0.1, wspace=0.1)

# Create first subplot
ax0 = fig.add_subplot(gs[0, 0])
# Subsequent subplots share axes with first one
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
# etc.
# %%
