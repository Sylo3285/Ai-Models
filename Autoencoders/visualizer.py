import matplotlib.pyplot as plt

plt.ion()  # interactive mode

def show_reconstruction(original, reconstructed, epoch, step):
    original = original.view(28, 28).cpu().detach()
    reconstructed = reconstructed.view(28, 28).cpu().detach()

    plt.clf()
    plt.suptitle(f"Epoch {epoch} | Step {step}")

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")

    plt.pause(0.001)
