import matplotlib.pyplot as plt
import os


def plot_adv_example(output_dir, orig_images, orig_labels, adv_images, adv_labels,
                     trackers, num_steps, epsilon, title="example"):
    for i in range(orig_images.shape[0]):
        orig_image, orig_label = orig_images[i].cpu().numpy(), orig_labels[i]
        adv_image, adv_label = adv_images[i].cpu().numpy(), adv_labels[i]
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(orig_image.transpose(1, 2, 0))
        axs[0, 0].set_title("Original image (cropped)")
        axs[0, 0].axis("off")
        delta = adv_image - orig_image
        delta -= delta.min()
        delta /= delta.max()
        axs[0, 1].imshow(delta.transpose(1, 2, 0))
        axs[0, 1].set_title(f"Perturbation (magnified) "
                            f"$\ell_\infty$ - $\epsilon={epsilon:.3f}$")
        axs[0, 1].axis("off")
        axs[0, 2].imshow(adv_image.transpose(1, 2, 0))
        axs[0, 2].set_title("Adversarial image")
        axs[0, 2].axis("off")
        axs[1, 0].plot(trackers[0].get().cpu()[i][:num_steps])
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].set_xlabel("Iterations")
        axs[1, 1].plot(trackers[1].get().cpu()[i][:num_steps])
        axs[1, 1].set_ylabel("Prediction")
        axs[1, 1].set_xlabel("Iterations")
        axs[1, 2].plot(trackers[-1].get().cpu()[i, 1][:num_steps], label="_")
        axs[1, 2].set_ylabel("Score")
        axs[1, 2].set_xlabel("Iterations")
        axs[1, 2].axhline(0, color="k", linestyle="--", label="Decision Threshold")
        axs[1, 2].legend()
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title}_{i + 1}.pdf"))
    plt.show()
