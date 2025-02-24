# Image Classification Project Using KNN and Performance Improvement
![knn-concept](https://github.com/user-attachments/assets/6a75e820-ebfd-4311-a41a-d39105c6eeed)

## Initial Training Explanation
Initially, I used the **K-Nearest Neighbors (KNN)** algorithm for image classification on the **CIFAR-10** dataset. The basic model was trained with the **K=18** value, and the accuracy achieved was **34%**. The model was trained without any preprocessing or hyperparameter optimization.

## CIFAR-10 Dataset Explanation
**CIFAR-10** is a dataset consisting of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 images in the training set and 10,000 images in the test set. 
![cifar_classes](https://github.com/user-attachments/assets/0ae9c2ea-55c6-4961-9bbc-5e1d3d42bb71)

## Performance Improvement
After achieving **34%** accuracy, I tried several methods to improve the model's performance. 

### 1. **Using PCA = 25**
I used **PCA** to reduce the dimensions and compress the data. I selected **25 components** of **PCA**, so the top 25 components explained 77% of the variance in the data. This process helped to reduce the dimensionality and remove unnecessary noise from the data.

After applying PCA with **n_components=25**, the accuracy improved significantly to **42.2%**.

### 2. **Using Cosine Distance Metric**
I also experimented with changing the distance metric from the default **Euclidean** to **Cosine Distance**. This adjustment improved the accuracy further to **44.68%**.

### 3. **Using Standardization**
I attempted to standardize the images using **StandardScaler** to adjust the data to have a mean of 0 and a standard deviation of 1:
```python
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)
```
However, this approach led to a significant drop in performance, with accuracy decreasing by 44%, resulting in a lower overall accuracy compared to previous experiments.

### 4. Weighting of Neighbors
Next, I experimented with Weighting of Neighbors, where closer neighbors received higher weights in the decision-making process. This improvement resulted in a higher accuracy of **46%**.

### Example of the results:
![Figure_1](https://github.com/user-attachments/assets/6915a428-f48b-49b1-95d9-e4e61948fb24)

### 5. Other Future Improvements
While PCA, the cosine distance metric, and weighting of neighbors have proven to be important steps in the process, there are several other directions to explore for further performance enhancement, such as:

* Automated selection of K using GridSearchCV to find the optimal K value.
* Increasing the data size through Data Augmentation.

## Conclusion
By applying all the methods, **the model's accuracy improved from 34% accuracy to 46% - by 35.29%!.**
The next step will be to explore additional techniques to further improve performance and achieve even higher results.

