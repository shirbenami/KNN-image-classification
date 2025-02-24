import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from show_images import show_images
import json

with open('config.json','r') as f:
    config = json.load(f)
    
n_neighbors = config['n_neighbors']
metric = config['metric']
weights = config['weights']
num_images = config['num_images']

# download the dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#print the classes images
show_images(train_images, train_labels, class_names)

# flatten the images- each image is an array in size 32x32x3 = 3072 , because the input for KNN is 1D array
train_images = train_images.reshape((train_images.shape[0], 32 * 32 * 3))
test_images = test_images.reshape((test_images.shape[0], 32 * 32 * 3))


# normalize the images to be in the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0


# create PCA object
pca = PCA(n_components=25)
train_images_pca = pca.fit_transform(train_images)
test_images_pca = pca.transform(test_images)

# create the model
knn = KNeighborsClassifier(n_neighbors=n_neighbors,metric=metric,weights=weights)

# train the model
knn.fit(train_images_pca, train_labels.ravel())

# check the explained variance ratio for decide how many components to use in PCA
print(pca.explained_variance_ratio_)
# print the total explained variance - we want it to be close to 1 (100%)
print(f"Total explained variance: {sum(pca.explained_variance_ratio_)}")


# predict the test set
predictions = knn.predict(test_images_pca)

# calculate the accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# display the first num_image_to_show images and their predicted labels
for i in range(num_images):
    plt.subplot(1,num_images,i+1) # create a subplot for each image
    plt.imshow(test_images[i].reshape(32, 32, 3)) # reshape the image to 32x32x3 
    plt.title(f'Label: {test_labels[i][0]}\nPrediction: {predictions[i]}') # set the title of the image
    plt.axis('off') # turn off the axis
plt.show() # display the images