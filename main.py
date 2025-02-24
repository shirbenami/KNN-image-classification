import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# download the dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# display the first image
#plt.imshow(train_images[0])
#plt.title(f'Label: {train_labels[0]}')
#plt.show()



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
knn = KNeighborsClassifier(n_neighbors=18,metric='cosine',weights="distance")

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

# display the first test image
plt.imshow(test_images[0].reshape(32, 32, 3))  
plt.title(f'Predicted Label: {predictions[0]}')
plt.show()
