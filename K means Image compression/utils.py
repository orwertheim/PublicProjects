def test():
    version=1.1
    print(f'utils.py version:{version}')

def plot_rgb_3d(image, sample_size=10000):
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
    ax.set_title("3D RGB Color Distribution", fontsize=14, fontweight="bold")

    plt.show()


def plot_kMeans_RGB(X, centroids, idx, K):
    # Plot the colors and centroids in a 3D space
    fig = plt.figure(figsize=(22, 22))
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(*X.T*255, zdir='z', depthshade=False, s=.3, c=X)
    ax.scatter(*centroids.T*255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3)
    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')

    # Set the color of the Y-axis pane
    ax.yaxis.pane.fill = True
    ax.yaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_facecolor((0., 0., 0., .2))  # Set the color
    ax.set_title("Original colors and their color clusters' centroids")
    plt.show()


def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0,len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
