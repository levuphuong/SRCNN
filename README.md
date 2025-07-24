# SRCNN

Implementation of the Super-Resolution Convolutional Neural Network (SRCNN) for single-image super-resolution.

![SRCNN Example]

---

## Overview

SRCNN is a simple yet effective deep learning model for **image super-resolution**, aiming to recover high-resolution images from their low-resolution counterparts.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/levuphuong/SRCNN.git
   cd SRCNN
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Download the dataset
```bash
python -m dataset.download_dataset
```

### 2. Convert the dataset to HDF5 format
```bash
python -m dataset.build_dataset
```

### 3. Train the model
```bash
python -m model.train
```

### 4. Evaluate the model
```bash
python -m script.validate
```

---

## Project Structure

```
SRCNN/
├── dataset/          # Code for downloading and preprocessing the dataset
├── model/            # SRCNN model definitions and training scripts
├── script/           # Scripts for evaluation and validation
├── requirements.txt  # Required Python packages
└── README.md
```

---

## Results

| Input (LR) | Output (SRCNN) | Ground Truth (HR) |
|------------|----------------|-------------------|
| ![](assets/lr.png) | ![](assets/sr.png) | ![](assets/hr.png) |

---

## References

- [SRCNN Paper](https://arxiv.org/abs/1501.00092)
- [Original Implementation](https://github.com/yjn870/SRCNN-pytorch)

---

## License

MIT License.
