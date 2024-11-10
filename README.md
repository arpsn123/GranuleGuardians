# Detectron2-based Segmentation of Cells, Mitochondria, Canalicular Vessels, and Alpha Granules

![Detectron2](https://img.shields.io/badge/Framework-Detectron2-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow) ![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-red) ![Deep Learning](https://img.shields.io/badge/Approach-Deep%20Learning-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview
This project utilizes **Detectron2**, a high-performance library for object detection and segmentation, to perform **instance segmentation** on cellular images. The goal of the project is to segment four biologically significant classes — cells, mitochondria, canalicular vessels, and alpha granules — in high-resolution microscopic images. Through precise segmentation, this project enables detailed observation of cellular structures and their spatial relationships, which can aid in medical diagnostics, cellular biology research, and pathology studies.

The project includes a comprehensive data preprocessing and annotation phase, followed by model training and evaluation. Detectron2’s powerful Mask R-CNN model, fine-tuned for this task, is capable of generating high-quality segmentations, accurately delineating the complex shapes and sizes of each structure. Outputs include both fully segmented images and binary masks for each class, making this project a versatile tool for research applications requiring cellular analysis.

### Key Features
- **High-Resolution Cellular Segmentation**: Detectron2's capabilities are leveraged to deliver high-quality segmentations for each biological structure, aiding in detailed structural and spatial analysis.
- **Four-Class Segmentation**: Aims at four distinct cellular classes that are essential for understanding cell functionality and morphology.
- **Configurable Training Pipeline**: Detailed configurations to support reproducibility and adjustments to the segmentation model as needed.

## Dataset Description and Annotation Process

### Dataset Description
The dataset used for this project comprises high-resolution images capturing cellular components at a microscopic scale, which allows the model to focus on the fine details necessary for distinguishing complex biological structures. The images include the following classes:

1. **Cells**: Fundamental structural units. Cells are annotated as individual objects to allow the model to distinguish distinct cell boundaries, supporting insights into cellular interactions and morphology.
   
2. **Mitochondria**: Organelles essential for cellular energy production. Accurate segmentation of mitochondria provides critical data on cell health, aiding in the study of diseases linked to mitochondrial function.
   
3. **Canalicular Vessels**: Tube-like structures that facilitate nutrient transport. Segmentation of these vessels helps researchers observe their integrity and distribution, which is crucial in understanding tissue health.

4. **Alpha Granules**: Granules in platelet cells involved in clotting. Segmenting these structures enables research into clotting disorders and platelet function, as their distribution and abundance are medically relevant.

Each class’s inclusion was selected to aid in understanding cellular structure in both healthy and potentially pathological samples, providing valuable insights for researchers.

### Annotation Process
An extensive manual annotation process was used to ensure the highest level of accuracy and consistency:

1. **Tool Selection**: Advanced annotation platforms such as **CVAT** and **Labelbox** were employed, chosen for their ability to manage complex multi-class labels with precision.

2. **Annotation Guidelines**:
   - **Cells**: Annotated with clear contours to separate individual cells without overlap, providing distinct instances for segmentation.
   - **Mitochondria**: Marked based on specific organelle size and shape, critical for the model to recognize irregular shapes and diverse distributions within the cellular matrix.
   - **Canalicular Vessels**: These were annotated with narrow, elongated marks, capturing their linear structure and preventing blending with nearby cellular features.
   - **Alpha Granules**: Each granule was annotated as a distinct small circular region to capture their dispersed distribution within cells accurately.

3. **Quality Control**: To ensure annotation reliability, each labeled image underwent expert review. Any inconsistencies were corrected, and unclear boundaries were refined, creating a high-quality dataset for training.



## Project Structure
The project is organized into folders, each with a dedicated purpose:
1. **`output` Folder**: Contains trained model artifacts and metadata:
   - `model_final.pth`: Final model weights after training.
   - `last_checkpoint`: Points to the most recent checkpoint.
   - `events.out.tfevents...`: Contains training logs for performance monitoring in TensorBoard.
   - `config.yaml`: Records all configuration settings, including hyperparameters, model layers, and other training details.

2. **`predicted_seg_test_images` Folder**: Stores fully segmented test images where all four classes are visualized together in a single image. Each class is color-coded to easily differentiate between cells, mitochondria, canalicular vessels, and alpha granules, showcasing the model's multi-class segmentation ability.

3. **`instances_test_seg_images` Folder**: Includes binary segmentation outputs for each class. Here, each test image has four separate copies, with each copy showing only one class in a binary mask (values of 0 and 255). These binary masks enable further analysis, like per-class evaluation and isolated visualization.



## Technical Approach
Detectron2's Mask R-CNN architecture was employed to achieve precise instance segmentation, optimized specifically for biomedical imaging. Key components of the approach include:

1. **Data Preprocessing**:
   - **Augmentation**: Resizing, normalization, and rotation augmentations were applied to make the model invariant to image scaling and orientation changes.
   - **Binary Label Masks**: Separate binary masks were generated for each class, allowing the model to learn class-specific features and contours effectively.

2. **Model Architecture**:
   - **Backbone**: We utilized a ResNet-50 with a Feature Pyramid Network (FPN) backbone. FPNs enable multi-scale feature extraction, essential for detecting varying-sized cellular structures.
   - **Region Proposal Network (RPN)**: The RPN generates region proposals that are refined through successive stages of Mask R-CNN, ensuring robust instance-level segmentation.
   - **ROI Heads**: ROI Align is applied to the proposals to maintain spatial alignment, which is essential for detecting small, detailed cellular structures.

3. **Loss Functions**:
   - **Classification Loss**: Cross-entropy loss was used for class predictions.
   - **Bounding Box Regression Loss**: Smooth L1 loss ensured accurate bounding box placement around each segmented instance.
   - **Mask Loss**: Binary cross-entropy was used for segmentation masks, maximizing accuracy by focusing on pixel-level predictions.

4. **Evaluation Metrics**:
   - **Mean Intersection over Union (mIoU)**: Calculated for each class to gauge overlap accuracy between predicted masks and ground truth masks.
   - **Pixel Accuracy**: Assessed the ratio of correctly predicted pixels across all test images, providing an indicator of model precision.



## Model Training and Optimization
Training was conducted using Detectron2's flexible configuration settings, allowing the model to be fine-tuned for our biomedical segmentation tasks. Key training details include:

1. **Learning Rate Scheduling**:
   - A **Cosine Annealing** schedule dynamically adjusted the learning rate to avoid overshooting and ensure stable convergence, a technique particularly useful when training deep networks.

2. **Batch Normalization**:
   - Batch normalization was applied to stabilize learning by normalizing feature distributions. This technique reduces covariate shifts and improves generalization on unseen biomedical data.

3. **Augmentation Techniques**:
   - Augmentations, such as color jitter, Gaussian blur, and horizontal flips, were implemented to improve model robustness, enhancing performance on variable test samples. These augmentations address challenges like class imbalance and intra-class variations in cell structure and size.

4. **Regularization**:
   - **Weight Decay** was applied to prevent overfitting, particularly critical for biomedical datasets where samples are often limited and overfitting can be a concern.


## Generated Outputs

1. **Model and Configuration Files**:
   - The **`output`** folder contains all files necessary for reproducing and evaluating the model:
     - `model_final.pth`: The trained model weights.
     - `config.yaml`: Configuration file detailing the model settings, hyperparameters, and training setup.
     - Training logs are stored in `events.out.tfevents...`, allowing visualization in TensorBoard for performance tracking.

2. **Predicted Segmentation Images**:
   - **`predicted_seg_test_images`**: Test images with all classes segmented in a single color-coded output, demonstrating the model’s capability to distinguish among cellular structures.

3. **Binary Instance Segmentations**:
   - **`instances_test_seg_images`**: Binary masks for each class, enabling detailed analysis and flexible use in downstream research or visualization tasks.

## Technology Stack

The project leverages a diverse set of tools, frameworks, and libraries that contribute to its robustness, precision, and ease of deployment. Each component was chosen based on its performance, compatibility, and relevance to segmentation tasks in computer vision and biomedical imaging.

#### 1. **Python 3.8+**
   - **Purpose**: Python was chosen as the primary programming language due to its extensive ecosystem, ease of integration with machine learning libraries, and popularity in the computer vision community.
   - **Benefits**: Python provides access to a broad array of scientific computing libraries, simplifying data manipulation, model development, and deployment.

#### 2. **Detectron2**
   - **Purpose**: Detectron2 is the core framework used for model training and instance segmentation. Developed by Facebook AI Research (FAIR), Detectron2 is optimized for high-performance image analysis, making it ideal for handling complex biological segmentation tasks.
   - **Features**: 
      - **Mask R-CNN**: Detectron2’s Mask R-CNN architecture excels in instance segmentation, enabling the model to generate pixel-level masks for multiple classes.
      - **Customizable Configurations**: Detectron2’s modular design and configuration options allow fine-tuning of model hyperparameters, enhancing its adaptability to biomedical images.
      - **Hardware Acceleration**: Built with support for GPU acceleration, Detectron2 enables faster model training and inference.

#### 3. **OpenCV (Open Source Computer Vision Library)**
   - **Purpose**: OpenCV is used for various image processing tasks such as resizing, color transformations, and data augmentation, preparing images for segmentation and enhancing model performance.
   - **Features**:
      - **Image Manipulation**: Efficient tools for transformations, filtering, and adjustments to improve data quality.
      - **Augmentation**: Essential for expanding the dataset by adding variations, which helps the model generalize across diverse cellular images.

#### 4. **NumPy**
   - **Purpose**: NumPy provides efficient handling of large datasets and supports multi-dimensional arrays, facilitating data processing and manipulation, especially in high-dimensional biomedical data.
   - **Benefits**:
      - **Fast Computation**: NumPy's optimized array operations allow rapid manipulation of large datasets, a crucial aspect of high-resolution image processing.
      - **Integration**: Acts as a backbone for other libraries such as OpenCV and Detectron2, supporting seamless data transformations.

#### 5. **Annotation Tools: CVAT (Computer Vision Annotation Tool) and Labelbox**
   - **Purpose**: CVAT and Labelbox were employed for precise manual annotations of cellular structures, allowing the creation of labeled datasets required for training the segmentation model.
   - **Features**:
      - **Multi-Class Labeling**: These tools support multi-class and instance labeling, essential for delineating the four classes in this project.
      - **User-Friendly Interfaces**: Both tools provide easy-to-use interfaces for drawing boundaries around cells, mitochondria, vessels, and granules.
      - **Collaboration and Quality Control**: Annotation review and refinement are facilitated, improving dataset quality.

#### 6. **PyTorch**
   - **Purpose**: As the underlying deep learning framework for Detectron2, PyTorch provides the building blocks for model training, data handling, and GPU acceleration.
   - **Features**:
      - **Dynamic Computational Graphs**: PyTorch’s flexible graphing system enables on-the-fly model adjustments and debugging, improving experimentation speed.
      - **Extensive Library Support**: PyTorch integrates with numerous machine learning libraries and supports a wide array of pre-trained models.
      - **GPU Acceleration**: Compatibility with CUDA allows leveraging NVIDIA GPUs for faster computations, significantly reducing training time on high-resolution datasets.

#### 7. **CUDA (Compute Unified Device Architecture)**
   - **Purpose**: CUDA enables GPU acceleration for PyTorch and Detectron2, facilitating faster training and inference on large image datasets.
   - **Benefits**:
      - **Parallel Computing**: CUDA allows parallel execution of operations, essential for managing the intensive computations in deep learning.
      - **Efficiency**: Significantly reduces model training times, especially for computationally heavy models like Mask R-CNN.

#### 8. **Matplotlib and Seaborn**
   - **Purpose**: These visualization libraries were used to analyze model performance and generate graphs, such as loss curves and accuracy metrics, throughout the training process.
   - **Features**:
      - **Plotting Metrics**: Provides easy-to-interpret visualizations of metrics such as training/validation loss, helping to monitor model progress and diagnose potential overfitting or underfitting.
      - **Customization**: Extensive customization options allow fine-tuning of plots to create professional and clear graphical outputs.

#### 9. **Jupyter Notebook**
   - **Purpose**: Used as the development environment for experimentation and training, allowing for organized and interactive exploration of code and results.
   - **Features**:
      - **Interactive Coding**: Supports step-by-step code execution, enabling rapid testing of code snippets and model adjustments.
      - **Documentation**: Allows embedding of markdown notes alongside code, making it ideal for documenting experiments and findings.
      - **Visualization Integration**: Provides support for inline plots and visualizations, essential for observing segmentation results and model performance metrics.

#### 10. **Git and GitHub**
   - **Purpose**: Git and GitHub were used for version control and project management, facilitating collaborative development and code maintenance.
   - **Features**:
      - **Version Control**: Allows tracking of code changes over time, making it easier to manage iterations and improvements.
      - **Collaboration**: GitHub provides a platform for collaborative work, issue tracking, and review, enhancing project robustness.
      - **Documentation Hosting**: The README.md file and other project documents are hosted on GitHub, providing a centralized resource for project details and usage instructions.



