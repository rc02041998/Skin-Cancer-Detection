# README: Dual-Model Image Classification using ResNet50 and Custom CNN with Attention Mechanism

## Overview
This project is a dual-model image classification pipeline integrating a pre-trained ResNet50 and a custom convolutional neural network (CNN) with a global attention mechanism. The model is designed to classify images into two categories with robust feature extraction and attention-based feature refinement.

## Features
- **Pre-trained ResNet50**: Extracts rich features from images.
- **Custom CNN**: Provides additional feature extraction with a parallel architecture.
- **Global Attention Block**: Refines features to focus on significant regions of the image.
- **Data Augmentation**: Improves model generalization.
- **Flexible Training**: Fully trainable layers with learning rate decay and checkpointing.

## Requirements
- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pandas
- Seaborn
- tqdm
- PIL (Pillow)
- scikit-learn

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
- **Input**: Images organized into two categories (`0` and `1`) with corresponding CSV files (`0.csv`, `1.csv`) containing image names and labels.
- **Preprocessing**: Images are resized to `224x224` and normalized to `[0, 1]`.

## Model Architecture
### 1. **ResNet50**:
- Pre-trained on ImageNet.
- Truncated at the final convolutional block.

### 2. **Custom CNN**:
- Multiple convolutional blocks with concatenation and feature refinement.
- Designed to capture diverse spatial features.

### 3. **Global Attention Block**:
- Refines features by emphasizing important regions.
- Combines channel-wise and spatial attention.

### 4. **Final Model**:
- Combines features from ResNet50 and the custom CNN using concatenation.
- Applies global average pooling and dense layers for binary classification.

## Training and Validation
- **Hyperparameters**:
  - Learning Rate: `0.0001`
  - Loss Function: Binary Cross-Entropy
  - Optimizer: Adam
  - Epochs: `70`
  - Batch Size: `32`
- **Callbacks**:
  - ReduceLROnPlateau: Adjusts learning rate based on validation loss.
  - ModelCheckpoint: Saves the best model based on validation accuracy.

## Usage
1. **Prepare Dataset**:
   - Place images in directories `0/` and `1/`.
   - Create `0.csv` and `1.csv` containing `image_name` and `target` columns.

2. **Run Training**:
   Execute the script to preprocess data, train the model, and save checkpoints:
   ```bash
   python train.py
   ```

3. **Evaluate**:
   Evaluate the model on the test set:
   ```bash
   python evaluate.py
   ```

4. **Visualize**:
   Plot accuracy vs. epochs:
   ```bash
   python plot_results.py
   ```

## Results
- **Test Loss**: [Output from the script]
- **Test Accuracy**: [Output from the script]

## Visualization
Accuracy vs. Epoch graph is generated to monitor training and validation performance.

## Contributions
Feel free to submit issues or pull requests for improvements.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
