# Thesis Project


## Folder Structure
- **Preprocessing_Scripts**: 
  - Contains scripts for cleaning and preparing the raw audio data.
  - Tasks include loading audio files, trimming silence, segmenting audio into smaller chunks, and encoding labels.
  - Ensures the data is ready for feature extraction and modeling.

- **Feature Extraction Scripts**: 
  - Includes scripts for extracting meaningful features from audio data.
  - Features include MFCC, Mel Spectogram, Spectral Contrast, Zero Crossing Rate, and more.
  - These features are used as input for machine learning models to improve classification accuracy.

- **Modeling Scripts**: 
  - Contains scripts for training machine learning models such as SVM (Support Vector Machine), Random Forest, Decision Tree, and Naive Bayes.
  - Includes brute-force feature selection methods to identify the best combination of features for optimal model performance.
  - Provides flexibility to experiment with different models and configurations.

- **Performance Evaluation Scripts**: 
  - Scripts for evaluating model performance using various metrics such as accuracy, confusion matrix, and classification report.
  - Includes cross-validation to ensure robustness and reliability of the models.
  - Helps compare the performance of different models and configurations.

- **Visualization and Reports Scripts**: 
  - Includes scripts for generating plots and visualizations to analyze and present results.
  - Visualizations include:
    - Model accuracy comparison (bar charts).
    - Emotion distribution across configurations (subplots).
    - Emotion profiles over time (line plots).
  - Provides insights into the data and model performance.