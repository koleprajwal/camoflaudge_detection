# Camouflage Detection

A deep learning project for detecting camouflaged objects using U-Net architecture. This project identifies objects that blend into their surroundings using semantic segmentation techniques.

## Dataset

This project uses the **COD10K-v3 Dataset** for training and evaluation.

**Dataset Link**: [COD10K-v3 - Camouflaged Object Detection Dataset](https://github.com/DengPingFan/COD10K)

### Dataset Structure
- **Train**: Contains training images and ground truth masks (edge, instance, object)
- **Test**: Contains test images and ground truth masks
- **Info**: Contains file lists for train/test splits

## Project Structure

```
camouflage_detection/
├── app.py                    # Flask/Streamlit web application
├── train.py                  # Training script
├── predict.py                # Prediction/inference script
├── prepare_data.py           # Data preprocessing script
├── model.pth                 # Pre-trained model weights
├── model/
│   └── unet.py              # U-Net model architecture
├── utils/                    # Utility functions
├── dataset/
│   ├── images/              # Processed dataset images
│   └── masks/               # Corresponding segmentation masks
├── COD10K-v3/               # Original COD10K dataset
└── venv/                    # Virtual environment
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/koleprajwal/camoflaudge_detection.git
   cd camouflage_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Dataset
```bash
python prepare_data.py
```
This script processes the COD10K-v3 dataset and organizes it into `dataset/images` and `dataset/masks`.

### 2. Train Model
```bash
python train.py
```
Trains the U-Net model on the prepared dataset. The trained model is saved as `model.pth`.

### 3. Make Predictions
```bash
python predict.py --image_path <path_to_image>
```
Runs inference on a single image to detect camouflaged objects.

### 4. Run Web Application
```bash
python app.py
```
Launches the web interface for interactive predictions.

## Model Architecture

The project uses a **U-Net** convolutional neural network for semantic segmentation:
- Encoder-decoder architecture
- Skip connections for preserving spatial information
- Suitable for object detection in complex backgrounds

## Requirements

Key dependencies:
- torch
- torchvision
- opencv-python
- numpy
- pillow
- (See `requirements.txt` for complete list)

## Results

The model achieves high accuracy in detecting camouflaged objects across various natural environments by learning features that distinguish hidden objects from their surroundings.

## License

This project uses the COD10K-v3 dataset. Please refer to the dataset's license for usage terms.

## References

- [COD10K: A Large-Scale Camouflaged Object Detection Benchmark](https://github.com/DengPingFan/COD10K)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

## Contact

For questions or suggestions, please open an issue on GitHub.
