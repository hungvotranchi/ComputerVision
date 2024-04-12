import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
def show_images(dataloader, num_images=5):
    # Get a single batch from the dataloader
    images, targets = next(iter(dataloader))
    
    # This will plot the specified number of images
    fig, axs = plt.subplots(1, num_images, figsize=(20, 5))
    
    for i, (image, target) in enumerate(zip(images, targets)):
        if i >= num_images:
            break
        
        # Convert the tensor image to PIL for easy handling
        image = F.to_pil_image(image)

        # Create the subplot for each image
        ax = axs[i]
        ax.imshow(image)
        ax.axis('off')
        
        # Optional: Add bounding boxes and labels to the image
        for box, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, str(label.item()), color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        
    plt.show()
