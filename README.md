# Corn-Disease-Detection
Corn Disease Classification with CNN and Genetic Algorithm Optimization
Project Overview
This project implements a Convolutional Neural Network (CNN) to classify corn leaf diseases using a dataset containing images of corn leaves affected by three diseases and healthy leaves. The CNN architecture is optimized using a Genetic Algorithm (GA) to enhance model performance. The pipeline includes data balancing, model training, evaluation, and visualization techniques such as Grad-CAM, ROC curves, and confusion matrices.
Dataset
The dataset consists of images from four classes:

Cercospora Leaf Spot (Gray Leaf Spot)
Common Rust
Northern Leaf Blight
Healthy

The original dataset is balanced to ensure each class contains 1000 images using data augmentation techniques. The balanced dataset is stored in balanced_corn_dataset/.
Requirements
To run this project, install the required Python packages:
pip install numpy matplotlib seaborn scikit-learn tensorflow opencv-python deap

Additionally, ensure you have:

Python 3.7+
CUDA-compatible GPU (optional, for faster training)
TensorFlow with GPU support (if using GPU)

Project Structure

balance_dataset: Balances the dataset by augmenting or sampling images to achieve 1000 images per class.
prepare_data: Creates data generators for training and validation with augmentation.
build_model: Constructs the CNN with configurable hyperparameters.
train_model: Trains the CNN with early stopping.
plot_training_metrics: Visualizes training and validation accuracy/loss.
grad_cam: Generates Grad-CAM heatmaps to visualize important regions in images.
optimize_model: Uses a Genetic Algorithm to optimize CNN hyperparameters.
test_model: Evaluates the final model and generates a confusion matrix.
evaluate_model: Produces classification reports and confusion matrices.
plot_roc_curve: Plots ROC curves for each class.
per_class_accuracy: Compares per-class accuracy between initial and optimized models.
evaluate_and_plot_predictions: Visualizes actual vs. predicted labels for 10 random validation images.

Usage

Set Dataset Path:Update the dataset_path in the main execution block to point to your dataset directory:
dataset_path = r'path_to_your_corn_dataset'


Run the Script:Execute the Python script to perform the entire pipeline:
python corn_disease_classification.py


Outputs:

Balanced dataset in balanced_corn_dataset/
Initial model saved as model_cnn2.h5
Final optimized model saved as final_model_cnn2.h5
Visualizations (confusion matrix, ROC curves, Grad-CAM, etc.) saved as PNG files
Printed classification reports and final model parameters



Key Features

Data Balancing: Ensures equal representation of classes using augmentation.
CNN Architecture: Three convolutional layers with batch normalization, followed by dense layers.
Genetic Algorithm: Optimizes hyperparameters (filters, dense units, dropout rate, learning rate).
Evaluation Metrics: Includes accuracy, loss, confusion matrix, classification report, ROC curves, and per-class accuracy.
Visualizations: Grad-CAM heatmaps, training/validation plots, and actual vs. predicted image comparisons.

Results

The initial model is trained and evaluated, followed by optimization using GA.
The final optimized model typically achieves higher validation accuracy (see confusion_matrix.png and classification report).
ROC curves and per-class accuracy plots provide insights into model performance across classes.
Grad-CAM visualizations highlight regions of interest in sample images.

Notes

Ensure the dataset directory structure matches the expected format (subfolders for each class).
The Genetic Algorithm may require significant computational resources; adjust generations and population size as needed.
CUDA GPU is recommended for faster training, but the script will run on CPU if no GPU is detected.
Saved models (model_cnn2.h5, final_model_cnn2.h5) can be loaded for further evaluation or inference.

License
This project is licensed under the MIT License.
Acknowledgments

Dataset sourced from [specify source if known, e.g., Kaggle, PlantVillage, etc.].
Built using TensorFlow, DEAP, and OpenCV.

