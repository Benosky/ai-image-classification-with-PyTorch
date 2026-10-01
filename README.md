# Flower Image Classification with PyTorch

An end-to-end deep learning application designed to train a deep neural network on a dataset of **102 flower categories** and use the trained model to perform real-time classification on new flower images. 

This project leverages **transfer learning** with a pre-trained **DenseNet-121** (or various VGG models) base backend architecture to map deep visual features to flower species names via a customized feed-forward network classifier.

---

## 🚀 Features

*   **Data Preprocessing Pipeline:** Automatic image resizing, center cropping, data augmentation (rotation, horizontal flips), and ImageNet dataset normalization.
*   **Transfer Learning Flexibility:** Supports training via multiple underlying architectures (`DenseNet121`, `VGG11`, `VGG13`, `VGG16`, `VGG19`, and `AlexNet`).
*   **Checkpoint Savings:** Fully serializes the deep network state, classification layer architectures, class-to-index mappings, and optimizer states for seamless recovery.
*   **Command Line Tooling:** Offers two straightforward scripts (`train.py` and `predict.py`) supporting customizable hyperparameter arguments.
*   **GPU Acceleration:** Full hardware acceleration support over CUDA cores if a compatible GPU device is passed.

---

## 📁 Directory Structure

```text
├── train.py              # Script to train a network on a dataset and save a checkpoint
├── predict.py            # Script to parse an image and predict flower class with a checkpoint
├── workspace_utils.py    # Keep-alive session handlers for training within GPU environments
├── cat_to_name.json      # Dictionary mapping class integer IDs to real flower name strings
├── README.md             # Project documentation (this file)
└── flowers_data/         # Labeled Dataset Directory 
    ├── train/            # Subfolders split per integer flower class IDs for Training
    ├── valid/            # Subfolders split per integer flower class IDs for Validation
    └── test/             # Subfolders split per integer flower class IDs for Testing
```

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python 3.6+ installed alongside the required machine learning and graphing modules:

```bash
pip install torch torchvision matplotlib numpy pandas pillow requests
```

### Dataset Structure
The program expects your target folder path (e.g., `flowers_data`) to have structured indices split inside `train`, `valid`, and `test` directories:

```text
flowers_data/train/1/image_06734.jpg
flowers_data/train/2/image_05102.jpg
```

---

## 🏋️ Training a New Model (`train.py`)

The training script lets you download a base architecture, swap the feed-forward network classifier with customized hidden layer dimensions, run validation paths, and save the final `.pth` checkpoint.

### Basic Usage
```bash
python train.py flowers_data checkpoint.pth
```

### Custom Architecture & Hyperparameters Example
```bash
python train.py flowers_data my_checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 10 --device gpu --loss NLL --batch_size 32
```

### Command Line Arguments for Training

| Argument | Shortcut | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `data_dir` | *(Positional)* | `str` | *Required* | Path directory where the datasets are located. |
| `checkpoint` | *(Positional)* | `str` | *Required* | Path filename where your checkpoint file will save. |
| `--arch` | `-a` | `str` | `densenet` | Model architecture choices: `densenet`, `alexnet`, `vgg11_bn`, `vgg13`, `vgg16`, `vgg19`. Type `help` to see list. |
| `--learning_rate`| `-l` | `float` | `0.003` | Hyperparameter learning rate (between `0` and `1`). |
| `--hidden_units` | `-u` | `int` | `900` | The number of nodes inside your model's hidden layers. |
| `--epochs` | `-e` | `int` | `9` | Number of iterations over the full training set. |
| `--device` | `-d` | `str` | `gpu` | Device setting option: `gpu` or `cpu`. |
| `--loss` | `-f` | `str` | `NLL` | Criterion loss option: `NLL`, `L1`, `Poisson`, `MSE`, `Cross`. Type `help` to see list. |
| `--batch_size` | `-b` | `int` | `32` | Number of input items per training batch iteration. |

---

## 🔮 Inference & Classification Predictions (`predict.py`)

The prediction script takes a single target image file and evaluates it against a stored `.pth` network checkpoint. It then prints the most likely integer classes and real flower names along with their relative probabilities.

### Basic Usage
```bash
python predict.py flowers_data/test/102/image_08012.jpg checkpoint.pth
```

### Custom Top-K Classes and Category JSON Mapping Example
```bash
python predict.py paths/to/flower_img.jpg my_checkpoint.pth --topk 3 --cat_names category_labels.json --gpu y
```

### Command Line Arguments for Inference

| Argument | Shortcut | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `path_to_image` | *(Positional)* | `str` | *Required* | Target location string pointing to the image file. |
| `checkpoint` | *(Positional)* | `str` | *Required* | Filename reference of the saved model weights checkpoint. |
| `--topk` | `-t` | `int` | `5` | Returns top K most likely matching flower classes. |
| `--cat_names` | `-c` | `str` | `cat_to_name.json` | JSON mapping path matching integer class IDs to real flower string names. |
| `--gpu` | `-g` | `str` | `n` | Set `y` or `n` to activate GPU hardware acceleration. |

### Terminal Output Preview
```text
Shape of ps is: torch.Size()
['102', '79', '59', '23', '69']
['blackberry lily', 'toad lily', 'orange dahlia', 'fritillary', 'windflower']
[9.99719620e-01, 2.57086009e-04, 1.69719151e-05, 1.45413389e-06, 1.02587001e-06]
```

---

## 📊 Pipeline Specifications

### Image Normalization Tensor Transforms
To remain consistent with the pre-trained weights compiled within the ImageNet dataset, images are parsed using the following standard tensors:
*   **Mean Channel Adjustments:** `[0.485, 0.456, 0.406]`
*   **Standard Deviation Adjustments:** `[0.229, 0.224, 0.225]`
*   **Dimension Architecture:** PIL images are transposed from width/height dimensions to channel first vectors matching PyTorch expectancies: `[Batch Size, Channels, Height, Width]`.