import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def show_images(imgs, figsize=(20, 10), rows=1, cmap="gray"):
    """ Show a bunch of images in a grid """
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows + 1, int(len(imgs.items()) / rows))

    for (ii, (name, img)) in enumerate(imgs.items()):
        ax = plt.subplot(gs[ii])
        ax.set_title(name)
        ax.axis('off')
        ax.imshow(img, cmap=cmap)
