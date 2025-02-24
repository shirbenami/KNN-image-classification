import matplotlib.pyplot as plt

def show_predict(num_images, test_images, test_labels, predictions):
    # display the first num_image_to_show images and their predicted labels
    for i in range(num_images):
        plt.subplot(1,num_images,i+1) # create a subplot for each image
        plt.imshow(test_images[i].reshape(32, 32, 3)) # reshape the image to 32x32x3 
        plt.title(f'Label: {test_labels[i][0]}\nPrediction: {predictions[i]}') # set the title of the image
        plt.axis('off') # turn off the axis
    plt.show() # display the images