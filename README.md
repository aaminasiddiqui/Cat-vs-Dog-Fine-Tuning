## Cat-vs-Dog-Classification

A Convolutional Neural Network (CNN) model for classifying images of cats and dogs. The model is built using TensorFlow Keras and utilizes transfer learning with the VGG16 architecture.

### 1. Data Preprocessing

- **Normalization**: Scale pixel values to the range [0, 1] by dividing by 255.
- **Resizing**: Resize all images to 150x150 pixels.
- **Data Augmentation**: Apply random transformations such as rotation, width/height shift, shear, zoom, and horizontal flip to increase dataset variability and improve generalization.

### 2. Model Architecture 
The model uses transfer learning with the VGG16 architecture:
- **Input Layer**: Resized images (150x150x3)
- **Base Model**: Pre-trained VGG16 without the top classification layer
- **Custom Layers**:
  - Flatten layer to convert 2D feature maps to 1D feature vectors.
  - Dense layer with 256 units and ReLU activation.
  - Output layer with 1 unit and Sigmoid activation for binary classification.


### 3. Training
- **Loss Function**:Used binary cross-entropy loss for binary classification.
- **Optimizer**:Used Adam to minimize the loss function.
- **Training Process**: Fed the model batches of images and labels, iteratively adjusting the weights to minimize the loss.

### 4. Evaluation
After training, the model is evaluated on a separate validation dataset. Key performance metrics include accuracy, precision, recall, and F1-score.

### 5. Testing
After training, the model is evaluated on a separate test dataset to assess its generalization performance.

### 6. Deployment
Saved the trained model in a format suitable for deployment and used it to classify new, unseen images of cats and dogs in a production environment.
