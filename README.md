# Voice Command Recognition with Custom Dataset and Quantization

## Project Overview
This project focuses on adapting a pre-trained sequential Convolutional Neural Network (CNN) model for recognizing custom voice commands. The model is fine-tuned using a personalized dataset of voice recordings and optimized with quantization-aware training to balance performance and computational efficiency for deployment on resource-constrained devices.

## Datasets
1. **Speech Commands Dataset (Warden, 2018)**:
   - Contains 105,829 utterances recorded from 2,618 speakers.
   - Optimized for training and evaluating keyword spotting models.
   
2. **Custom Dataset**:
   - Built with personal voice recordings.
   - Includes 35 distinct command categories such as 'yes', 'no', 'up', 'down', etc.

## Model Architecture
- **Input Layer**: Accepts spectrogram representations of audio.
- **Preprocessing Layers**: Resizing and normalization for efficient processing.
- **Convolutional Layers**: Multiple layers with increasing filters to capture complex patterns.
- **Pooling and Regularization**: Reduces spatial dimensions and prevents overfitting.
- **Fully Connected Layers**: Processes extracted features for final classification.
- **Output Layer**: Highlights and classifies the spoken word.

## Fine-Tuning and Quantization
- **Fine-Tuning**: The model was fine-tuned on the custom dataset with 10 epochs. Key parameters included model checkpointing and early stopping to prevent overfitting.
- **Quantization**: Applied quantization-aware training to optimize the model for deployment on resource-constrained devices, balancing performance and efficiency.

## Evaluation Metrics
- **Accuracy**: Measures the proportion of correctly classified instances.
- **Loss**: Quantifies the difference between predicted and actual values to monitor learning progress.

## Real-World Applications
- **Healthcare**: Assists with transcribing patient notes and interacting with electronic health records.
- **Smart Homes**: Controls devices like lighting and security systems through voice commands.
- **Elderly Care**: Provides reminders for medication and emergency assistance.
- **Customer Service**: Enhances user experience with efficient voice-activated support systems.
- **Automotive**: Enables hands-free control of navigation and communication systems in vehicles.

## Future Work
Future enhancements could include integrating the system with natural language processing for more complex interactions and expanding its use in smart home automation and wearable technology.

## Getting Started
1. **Data Preparation**:
   - Extract datasets from `.tar` files and organize them as required.
2. **Model Training**:
   - Use TensorFlow to train and fine-tune the model.
   - Apply quantization-aware training for optimization.
3. **Evaluation**:
   - Assess model performance using accuracy and loss metrics.

## Installation
Ensure you have TensorFlow installed. You can install it using pip:

```bash
pip install tensorflow
