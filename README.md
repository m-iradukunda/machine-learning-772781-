Image Classification Project
### Team Members
* Marie Dominique IRADUKUNDA
* Souleymane Compaore
* Matteo Piccirilli

## Table of content
Introduction
Methods
Design Choices
Detailed Description
Results
Conclusions

### Introduction
This project involves developing a machine learning system to classify images into predefined categories such as resumes, advertisements, emails, and handwritten documents. The goal is to automate the sorting of images based on their content, which can streamline organizational tasks and improve efficiency.

### Methods
Proposed Ideas
Our approach utilizes a machine learning model trained on a labeled dataset of images. The key features of the project include:
* Feature Extraction with VGG16
Load pre-trained VGG16 model without the classifier layers, Path to your images Load and preprocess images.
* Model Training: We train a Support Vector Machine (SVM)/ kernel used for classification and regression tasks.
* Environment: The project is implemented we used Python, the following libraries and tools were used:

NumPy: Utilized for numerical operations and handling arrays.
Pillow (PIL): Used for image processing tasks.
shutil: Utilized for moving files into categorized directories.
Keras: Used for building and training the deep learning model (VGG16).
scikit-learn (sklearn): Used for various machine learning tasks such as clustering (KMeans), model evaluation, and classification (SVM).

### Design Choices
We made several design choices to ensure the effectiveness and efficiency of our image classification project:

* Support Vector Machine (SVM):

We chose SVM for classification due to its effectiveness in high-dimensional spaces and its ability to handle a large number of features. SVM is known for its robustness in classification tasks, particularly when the margin of separation between classes is clear.
* Keras (for Feature Extraction with VGG16):

We used the VGG16 model from Keras for feature extraction. VGG16 is a pre-trained convolutional neural network that provides a rich set of features for each image, which are then used by the SVM for classification. This combination leverages the power of deep learning for feature extraction and the efficiency of SVM for classification.
* Pillow (PIL):

We used Pillow for image processing tasks. Pillow is a powerful library that provides extensive image manipulation capabilities, making it ideal for preprocessing images before they are fed into the VGG16 model for feature extraction.
* scikit-learn (sklearn):

For classification tasks, we utilized scikit-learn’s SVM implementation. Scikit-learn also provided tools for clustering (KMeans), model evaluation, and data splitting (train_test_split), which were essential for the development and evaluation of our classification model.
* Shutil:

The shutil library was used to handle file operations, such as moving files into categorized directories after classification. This ensured efficient organization and management of image files based on their predicted categories.

* Explanation
SVM: Chosen for its effectiveness in classification tasks, especially in high-dimensional feature spaces.
VGG16 with Keras: Used for extracting rich feature representations from images, which are then classified by the SVM.
Pillow: Utilized for preprocessing and handling image data before feature extraction.
scikit-learn: Leveraged for its efficient SVM implementation and additional machine learning tools like clustering and evaluation metrics.
shutil: Utilized for efficient file handling and organization post-classification.

Experiment 1: Baseline Model
Purpose:
To establish a baseline accuracy for image categorization.

Baseline:
A simple SVM model with a linear kernel and minimal tuning.

* Evaluation Metrics:
Accuracy, Precision, Recall. These metrics are chosen to evaluate not only the overall performance but also the model's ability to correctly classify each category.

Experiment 2: Optimized Model
Purpose:
To improve the model's performance through hyperparameter tuning and data augmentation.

Baseline:
Results from the baseline SVM model.

Evaluation Metrics:
Same as Experiment 1, with additional focus on F1-score to balance precision and recall.


### Detailed Description
* Experiment 1: Baseline Model

Model: Support Vector Machine (SVM) with a linear kernel.
Purpose: Establish a baseline performance for the image categorization task.
Approach: Train an SVM with default parameters and a linear kernel on the extracted features from the images.
Evaluation Metrics:
Accuracy: Measures the overall correctness of the model.
Precision: Evaluates the accuracy of the positive predictions.
Recall: Measures the model's ability to identify all relevant instances.

* Experiment 2: Optimized Model

Approach: pre-trained VGG16 model without the classifier layers
Model: Support Vector Machine (SVM) with kernel.
Purpose: Improve the categorization performance by optimizing the SVM's hyperparameters and using a more complex kernel.
Approach:Use cross-validation to ensure the robustness of the model.
Potentially augment the dataset to improve generalization.
Evaluation Metrics:
Accuracy: Measures the overall correctness of the model.
Precision: Evaluates the accuracy of the positive predictions.
Recall: Measures the model's ability to identify all relevant instances.
F1-score: Balances precision and recall, providing a single metric for model performance.

### Results
* Main Findings
Our optimized Support Vector Machine (SVM) model with an RBF kernel achieved excellent performance in categorizing images into the specified categories: advertisement, emails, handwritten, and resume. The validation accuracy of the model was 0.9765, indicating a high level of accuracy in the predictions.

* Classification Report:

	              Precision      recall         F1-score       support
Advertisement	    0.98          0.98	          0.98           296
emails	            0.97          1.0	          0.98	         247
handwritten	        0.96	      0.97	          0.97	         223
resume	            0.99	      0.96	          0.97	         341

### Conclusions
Summary:
The primary takeaway from our project is that using a Support Vector Machine (SVM) model with an RBF kernel, combined with features extracted from the VGG16 neural network, yields highly accurate image categorization. The model achieved a validation accuracy of 97.65%, with precision, recall, and f1-scores consistently high across all categories: advertisement, emails, handwritten, and resume. These results underscore the effectiveness of SVMs in handling high-dimensional feature spaces and their robustness in classification tasks when paired with deep learning-based feature extraction.

Future Work
While our model demonstrates strong performance, there are several areas that warrant further investigation. Firstly, the model's generalizability to unseen, real-world data needs further validation. Additionally, the impact of using different feature extraction methods or combining multiple feature sets could be explored to potentially enhance performance. Moreover, the model's ability to handle more diverse and complex image categories beyond the four used in this project remains an open question. Future work could involve expanding the dataset, experimenting with different kernels and hyperparameters, and exploring the integration of more advanced neural network architectures for feature extraction. Finally, deploying the model in a real-time system and evaluating its performance in practical applications would be a valuable next step.
