import matplotlib.pyplot as plt

def show_classes_images(train_images, train_labels, class_names):
    # create a plot with example of each class
    plt.figure(figsize=(10, 10))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        idx = (train_labels == i).nonzero()[0][0] # get the index of the first image with label i
        plt.imshow(train_images[idx])
        plt.title(f'{class_names[i]} ({i})')
        plt.axis('off')
    plt.show()