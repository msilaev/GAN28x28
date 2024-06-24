import matplotlib.pyplot as plt
import itertools

def plot_results(fname, real_batch, fake):
    #########################################
    nrows = 5
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

    for nrow, ncol in itertools.product(range(ncols // 2), range(nrows)):
        axes[ncol][nrow].imshow(real_batch.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow].axis('off')

        axes[ncol][nrow + 2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow + 1].axis('off')

    plt.savefig(fname)

    plt.show()