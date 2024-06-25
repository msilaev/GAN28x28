import matplotlib.pyplot as plt
import itertools

def plot_results(fname, real_imgs, real_labels, fake, fake_labels):
    #########################################
    nrows = 6
    ncols = 6

    #fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

    #for nrow, ncol in itertools.product(range(ncols // 2), range(nrows)):

    #    axes[ncol][nrow].imshow(real_batch.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
    #    axes[ncol][nrow].axis('off')

    #    axes[ncol][nrow + 2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
    #    axes[ncol][nrow + 1].axis('off')

    ncols = 6
    fig, axes = plt.subplots(ncols=ncols, figsize=(8, 8))

    for ncol in range(ncols // 2):

        axes[ncol].imshow(real_imgs.permute(0, 2, 3, 1)[ncol], cmap='gray')
        axes[ncol].axis('off')
        axes[ncol].set_title('real, ' + str(real_labels[ncol].item()))

        axes[ncol + ncols // 2].imshow(fake.permute(0, 2, 3, 1)[ncol + ncols // 2], cmap='gray')
        axes[ncol + ncols // 2].axis('off')
        axes[ncol + ncols // 2].set_title('fake, ' + str(fake_labels[ncol + ncols // 2].item()))

    plt.savefig(fname)

    plt.show()