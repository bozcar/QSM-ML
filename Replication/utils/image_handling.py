import matplotlib.pyplot as plt

def show_image(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.show()