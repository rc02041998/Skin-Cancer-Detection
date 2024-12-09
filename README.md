Here's the updated README file including the provided code:

---

# Advanced Image Classification with Dual Attention Model

This project implements an advanced image classification pipeline using a dual-attention mechanism with DenseNet121 and a custom-defined parallel convolutional network. The model is trained to classify images into two categories.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
5. [Code Walkthrough](#code-walkthrough)
6. [Results](#results)
7. [License](#license)

---

## Introduction
This project explores deep learning techniques for image classification. A combination of DenseNet121 and a custom-defined convolutional network, enhanced with attention mechanisms, is employed to achieve high accuracy. 

## Setup
To run this project:
1. Clone the repository.
2. Prepare the data as per the instructions. Two datasets (`0.csv` and `1.csv`) are required, each containing image paths and labels.
3. Install the necessary dependencies.

## Dependencies
The following Python libraries are required:
- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`
- `tensorflow`
- `keras`
- `sklearn`
- `PIL`
- `plot-metric`

Install all dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare datasets in the specified format.
2. Run the training script to train the model:
   ```bash
   python train_model.py
   ```
3. Evaluate the model on the test set.

## Code Walkthrough
### Data Preprocessing
```python
# Load and preprocess datasets
train_set_0 = pd.read_csv('0.csv')
train_set_1 = pd.read_csv('1.csv')

# Load and normalize images
for i in tqdm(range(train_set_0.shape[0])):
    img = tf.keras.utils.load_img('0/' + train_set_0['image_name'][i], target_size=(224,224,3))
```

### Model Definition
- **Global Attention Block:** Enhances feature extraction with attention mechanisms.
- **Custom Model:** A parallel convolutional model is defined and combined with DenseNet121 for feature extraction.

```python
# Define the custom convolutional network
def define_model():
    # Custom architecture with parallel convolutional blocks
    ...
    return model

# Combine DenseNet121 with custom architecture
model1 = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))
model2 = define_model()

# Combine features using attention
conc = tf.keras.layers.Concatenate()([model1_out, model2_out])
attend_feature_1 = Global_attention_block(conc)
```

### Training
The model is compiled and trained using a learning rate scheduler and checkpointing.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr2),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit([X_train_x, X_train_x], y_train_x,
                    validation_data=([X_valid, X_valid], y_valid),
                    epochs=70, batch_size=32)
```

### Evaluation
The model's performance is evaluated using metrics like accuracy, loss, confusion matrix, and ROC curve.

```python
score = model.evaluate([X_test, X_test], y_test, batch_size=32)
print('Test Accuracy:', score[1])

# Generate confusion matrix
matrix = confusion_matrix(y_test, y_pred)
```

### Visualization
Plots for accuracy, loss, and ROC curve provide insights into the training process.

```python
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
```

## Results
The model achieves significant accuracy on the test dataset. The following metrics are calculated:
- Test Loss: `...`
- Test Accuracy: `...`

## License
This project is licensed under the MIT License.

--- 

Let me know if you'd like further adjustments or additions!