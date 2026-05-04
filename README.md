# Wildlife Camouflage Detection

**Short Description:**
This Wildlife Camouflage Detection model uses **U-Net semantic segmentation** to detect hidden animals in images. It analyzes features like **image textures, color patterns, spatial information, and edge details** to predict precise **segmentation masks** showing camouflaged wildlife locations. The system uses **Streamlit** for demonstration, providing an intuitive interface for real-time predictions on uploaded images.

## Overview

Wildlife Camouflage Detection is a deep learning project designed to identify and segment animals that are camouflaged in their natural habitats. Using advanced computer vision techniques, this model reveals wildlife concealed through adaptive coloration and background matching. The project leverages the **U-Net architecture with ResNet18 backbone** for pixel-level semantic segmentation, enabling precise localization of hidden animals in complex natural scenes.

### Key Features:
- 🎯 **Pixel-Level Accuracy**: Generates precise segmentation masks highlighting exact animal locations
- 🔬 **Advanced Architecture**: U-Net with ResNet18 backbone for superior feature extraction
- 📊 **Large-Scale Dataset**: Trained on COD10K-v3 with thousands of camouflaged object examples
- ⚡ **Real-Time Inference**: Fast predictions suitable for interactive applications
- 🌐 **Web Interface**: Streamlit-based UI for easy image upload and visualization
- 🎨 **Visual Overlay**: Displays detected camouflage regions overlaid on original images

### Applications:
- Wildlife monitoring and conservation
- Ecological research and biodiversity assessment
- Surveillance and security systems
- Object detection in complex backgrounds

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

### U-Net with ResNet18 Backbone

The project employs a **U-Net convolutional neural network** enhanced with a **ResNet18 encoder** for semantic segmentation:

**Architecture Components:**
- **Encoder**: Pre-trained ResNet18 backbone extracts multi-scale image features
- **Bottleneck**: Compressed feature representation capturing high-level semantic information
- **Decoder**: Progressive upsampling layers reconstruct the image to original resolution
- **Skip Connections**: Direct links from encoder to decoder preserve spatial details and boundary information
- **Output**: Binary segmentation mask (1 = camouflaged animal, 0 = background)

**Why U-Net?**
- ✅ Excellent for precise boundary detection
- ✅ Skip connections prevent loss of fine spatial details
- ✅ Data-efficient architecture suitable for limited datasets
- ✅ Fast inference for real-time applications
- ✅ Industry standard for semantic segmentation tasks

## Results & Performance

The trained model achieves:
- High accuracy in detecting camouflaged animals across diverse environments
- Precise pixel-level segmentation with clear boundary delineation
- Fast inference times suitable for real-time applications
- Robust performance on unseen test images from the COD10K-v3 dataset

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite the COD10K dataset:

```bibtex
@article{fan2021camouflaged,
  title={Camouflaged Object Detection},
  author={Fan, Deng-Ping and others},
  journal={IEEE CVPR},
  year={2021}
}
```

## Contact & Support

For questions or issues, please open an issue on GitHub or contact the project maintainers.

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
