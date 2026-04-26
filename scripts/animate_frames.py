import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch 

def animate_frames(batch_frames):
    batch_size=batch_frames.shape[0]
    fig = plt.figure(clear=True, figsize=(6, 6))
    axes = []
    for i in range(batch_size):
        axes.append(fig.add_subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i+1))
        axes[i].set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        axes[i].imshow(batch_frames[i][0])
    fig.tight_layout()

    ims = []
    for i in range(batch_frames.shape[1]):    # Loop over each timestep 
        sub_ims = []
        for j in range(batch_size):
            sub_im = axes[j].imshow(batch_frames[j][i], animated=True)
            sub_ims.append(sub_im)
        ims.append(sub_ims)

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    plt.close()
    return ani