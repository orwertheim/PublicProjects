import numpy as np
import matplotlib.pyplot as plt    
from PIL import Image 
import requests
from io import BytesIO

def test():
    version=1.3
    print(f'utils.py version:{version}')
    
def plot_costs(x, title='', x_label='', y_label='', bins=30, y_lim=None):
    """
    Plots the costs of multiple K-Means initializations.

    Args:
        costs (ndarray or list): Array or list of cost values from different K-Means initializations.
    """
    plt.figure(figsize=(8, 6))
  
    plt.plot(x, color='skyblue', marker='o', linestyle='', markersize=5, alpha=0.7)
    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    if y_lim is not None:
      plt.ylim(y_lim[0], y_lim[1])

    # Show plot
    plt.show()


def download_from_gdrive(file_id, destination):
    try:
        # First create a session
        session = requests.Session()
        
        # Get the initial page to get cookies
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True)
        
        # Check if there's a download warning (for larger files)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Add confirmation parameter
                url = f"{url}&confirm={value}"
                break
        
        # Now download with the confirmation if needed
        response = session.get(url, stream=True)
        img = Image.open(BytesIO(response.content))
        img.save(destination)
        # Save the response content 
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False
        
def plot_kMeans_RGB(X, centroids_list, idx_list, titles, sample_size=5000):
    """
    Plots the RGB pixel distribution in 3D space for the original image and multiple sets of K-means cluster centroids.
    
    Parameters:
        X (array-like): Original image data in RGB format, shape (N, 3).
        centroids_list (list of array-like): List of cluster centroid arrays, each of shape (K, 3).
        idx_list (list of array-like): List of cluster assignments, each of shape (N,).
        titles (list of str): Titles for each plot.
        sample_size (int, optional): Number of pixels to sample for visualization (default: 5000).
    """
    X = np.array(X)
    
    # Sample pixels to reduce computational load
    sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sampled = X[sample_indices]
    
    num_plots = 1 + len(centroids_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(22, 8), subplot_kw={'projection': '3d'})
    
    if num_plots == 1:
        axes = [axes]  # Ensure iterable consistency
    
    for i, ax in enumerate(axes):
        if i == 0:
            colors = X_sampled  # Original colors
        else:
            idx_sampled = idx_list[i - 1][sample_indices]
            colors = centroids_list[i - 1][idx_sampled]  # Assign colors based on clustering
            ax.scatter(*centroids_list[i - 1].T * 255, depthshade=False, s=200, c='black', marker='x', lw=2)
        
        ax.set_title(titles[i])
        ax.scatter(*X_sampled.T * 255, zdir='z', depthshade=False, s=3, c=colors)
        ax.set_xlabel('Red Channel')
        ax.set_ylabel('Green Channel')
        ax.set_zlabel('Blue Channel')
        
        # Improve 3D plot appearance
        ax.yaxis.pane.fill = True
        ax.yaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_facecolor((0., 0., 0., .2))
    print('------------------')
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
 
