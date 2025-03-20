def test():
    version=1.3
    print(f'utils.py version:{version}')

def plot_kMeans_RGB(X, centroids, idx, sample_size=5000, title="Original colors"):
    # Ensure input is a NumPy array
    X = np.array(X)
    # Sample pixels to reduce compute time
    sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sampled = X[sample_indices]
    colors = X_sampled # Original colors

    fig, axes = plt.subplots(1, 2, figsize=(22, 44), subplot_kw={'projection': '3d'})
    
     
    
    for i,ax in enumerate(axes): 
      ax.set_title(title)
      if i == 1:
        idx = idx[sample_indices]  # Sample idx to match X_sampled
        colors = centroids[idx] # Color points by their centroid color
        ax.set_title("Pixel Colors Based on Their Centroid")  
        ax.scatter(*centroids.T*255, depthshade=False, s=500, c='red', marker='x', lw=3)    
      ax.scatter(*X_sampled.T*255, zdir='z', depthshade=False, s=3, c=colors)
      
      ax.set_xlabel('R value - Redness')
      ax.set_ylabel('G value - Greenness')
      ax.set_zlabel('B value - Blueness')

      # Set the color of the Y-axis pane
      ax.yaxis.pane.fill = True
      ax.yaxis.pane.set_edgecolor('w')
      ax.yaxis.pane.set_facecolor((0., 0., 0., .2))  # Set the color
       
    plt.tight_layout()
    plt.show()
 

def plot_rgb_3d(image, title= '3D RGB Color Distribution', sample_size=10000):
    if isinstance(image, str):  # If input is a file path, load the image
        img = Image.open(image).convert("RGB")
        pixels = np.array(img, dtype=np.float32).reshape(-1, 3)  # Convert to float
    elif isinstance(image, np.ndarray):  # If input is already a NumPy array
        pixels = image.astype(np.float32).reshape(-1, 3)*255  # Ensure float type
    else:
        raise ValueError("Input should be a file path or a NumPy array.")

    if len(pixels) > sample_size:
        pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(r, g, b, c=pixels / 255.0, marker="o", alpha=0.6, edgecolors="w", linewidth=0.2)

    ax.set_xlabel("Red Channel", fontsize=12)
    ax.set_ylabel("Green Channel", fontsize=12)
    ax.set_zlabel("Blue Channel", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Adjust margins manually to ensure labels are not cut off
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)  # Adjust these values if necessary

    plt.show()


def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0,len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)

def display_images(images, titles):
    # Check that the number of images and titles match
    if len(images) != len(titles):
        raise ValueError("The number of images and titles must be the same.")
    
    # Calculate the grid size (rows and columns)
    num_images = len(images)
    cols = 3  # Number of columns in the grid
    rows = (num_images // cols) + (num_images % cols > 0)  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))  # Increase figure size for better scaling
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(images[i])  # Display the image
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')  # Set title with nicer font
        axes[i].axis('off')  # Hide axes
    
    # Turn off any remaining empty axes
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout to prevent overlapping titles and images
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Decrease space between images
    plt.show()

class Utils:
    pass
utils = Utils()
utils.test = test
utils.display_images = display_images 
utils.plot_kMeans_RGB=plot_kMeans_RGB 
utils.plot_rgb_3d = plot_rgb_3d
utils.show_centroid_colors=show_centroid_colors
