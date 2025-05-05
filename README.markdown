# Corn Disease Classification with CNN and Genetic Algorithm Optimization

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify corn leaf diseases using a dataset of corn leaf images. The CNN architecture is optimized using a **Genetic Algorithm (GA)** to improve performance. The pipeline includes:

- Data balancing
- Model training and evaluation
- Visualization techniques (ROC curves, confusion matrices)

---

## Dataset
The dataset comprises images from four classes:
- **Cercospora Leaf Spot (Gray Leaf Spot)**
- **Common Rust**
- **Northern Leaf Blight**
- **Healthy**

The dataset is balanced to ensure **1000 images per class** using data augmentation. The balanced dataset is saved in `balanced_corn_dataset/`.

---

## Requirements
To run this project, install the required Python packages:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow opencv-python deap
```

### Additional Requirements:
- **Python**: 3.7 or higher
- **CUDA-compatible GPU**: Optional, for faster training
- **TensorFlow**: GPU support recommended for CUDA users

---

## Project Structure
The project is organized into modular functions:

- **balance_dataset**: Balances the dataset to 1000 images per class using augmentation or sampling.
- **prepare_data**: Creates training and validation data generators with augmentation.
- **build_model**: Constructs a CNN with configurable hyperparameters.
- **train_model**: Trains the CNN with early stopping to prevent overfitting.
- **plot_training_metrics**: Plots training and validation accuracy/loss.
- **optimize_model**: Optimizes CNN hyperparameters using a Genetic Algorithm.
- **test_model**: Evaluates the model and generates a confusion matrix.
- **evaluate_model**: Produces classification reports and confusion matrices.
- **plot_roc_curve**: Plots **ROC curves** for each class.
- **per_class_accuracy**: Compares per-class accuracy for initial and optimized models.
- **evaluate_and_plot_predictions**: Visualizes actual vs. predicted labels for 10 random validation images.

---

## Usage
Follow these steps to run the project:

1. **Set Dataset Path**:
   Update the `dataset_path` in the main execution block:
   ```python
   dataset_path = r'path_to_your_corn_dataset'
   ```

2. **Run the Script**:
   Execute the script to run the entire pipeline:
   ```bash
   python corn_disease_classification.py
   ```

3. **Outputs**:
   - **Balanced dataset**: Saved in `balanced_corn_dataset/`
   - **Initial model**: Saved as `model_cnn2.h5`
   - **Final optimized model**: Saved as `final_model_cnn2.h5`
   - **Visualizations**: Confusion matrix, ROC curves, Grad-CAM heatmaps, etc., saved as PNG files
   - **Logs**: Classification reports and final model parameters printed to console

---

## Key Features
- **Data Balancing**: Ensures equal class representation using augmentation.
- **CNN Architecture**: Three convolutional layers with batch normalization, followed by dense layers.
- **Genetic Algorithm**: Optimizes hyperparameters:
  - Number of filters
  - Dense units
  - Dropout rate
  - Learning rate
- **Evaluation Metrics**:
  - Accuracy and loss
  - Confusion matrix
  - Classification report
  - ROC curves
  - Per-class accuracy
- **Visualizations**:
  - **Training/Validation Plots**: Accuracy and loss over epochs
  - **Actual vs. Predicted**: Compares predictions for sample images

---

## Results
- The **initial model** is trained and evaluated, followed by GA optimization.
- The **final optimized model** typically achieves higher validation accuracy (refer to `confusion_matrix.png` and classification report).
- **ROC curves** and **per-class accuracy** plots provide detailed performance insights.

---

## Notes
- Ensure the dataset directory has subfolders for each class.
- The **Genetic Algorithm** may be computationally intensive; adjust `generations` and `population` size as needed.
- A **CUDA GPU** is recommended for faster training, but the script supports CPU execution.
- Saved models (`model_cnn2.h5`, `final_model_cnn2.h5`) can be loaded for further evaluation or inference.

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgments
- **Dataset**: [Kaggle's PlantVillage Dataset]
- **Libraries**: Built using **TensorFlow**, **DEAP**, and **OpenCV**.
- **Community**: Thanks to open-source contributors for tools and documentation.
