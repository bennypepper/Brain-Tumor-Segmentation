
# Brain Tumor Segmentation using Fuzzy C-Means (FCM) 🧠

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5-blue.svg) ![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.4.2-orange.svg) ![Graphviz](https://img.shields.io/badge/Graphviz-2.49-lightgrey.svg)

An automated medical image analysis tool that applies the **Fuzzy C-Means (FCM)** soft clustering algorithm to segment brain MRI scans and highlight potential tumor regions.

## 📌 Project Overview
Diagnosing brain tumors accurately from Magnetic Resonance Imaging (MRI) is critical but often subjective and time-consuming. This project provides an automated segmentation pipeline using Fuzzy Logic. By clustering pixel data into distinct tissue groups (background, healthy tissue, fluids, and potential tumors), this tool serves as a reliable aid for medical image analysis.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Computer Vision:** OpenCV
* **Machine Learning / Fuzzy Logic:** scikit-fuzzy, NumPy
* **Visualization:** Matplotlib, Graphviz (for automated process mapping)
* **GUI / Interactions:** Tkinter

## 🚀 Key Features
* **Fuzzy C-Means Implementation:** Configured to dynamically group MRI pixels into 4 distinct clusters with a fuzziness parameter of 2.0, ideal for handling the blurred boundaries of medical tissues.
* **Automated Workflow Mapping:** Dynamically generates and displays Graphviz flowcharts mapping both the overall system process and the internal FCM algorithmic loops.
* **Interactive File Selection:** Built-in Tkinter GUI for seamless, user-friendly local image selection and processing.
* **Quantitative Evaluation:** Automatically calculates and outputs the Fuzzy Partition Coefficient (FPC) and objective function convergence to validate clustering quality.

## 📂 Repository Structure
```text
fcm-brain-tumor-segmentation/
├── data/
│   └── sample_mri.jpg        # Example MRI scans for testing
├── src/
│   └── segment_fcm.py        # Main execution script
├── requirements.txt          # Python dependencies
└── README.md
```

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/fcm-brain-tumor-segmentation.git](https://github.com/yourusername/fcm-brain-tumor-segmentation.git)
   cd fcm-brain-tumor-segmentation
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note for Windows Users: Graphviz requires a system-level installation. If the executable is missing from your PATH, the script will open an interactive prompt to help you locate the `bin` folder automatically.*

3. **Execute the segmentation script:**
   ```bash
   python src/segment_fcm.py
   ```

## 📊 Results & Performance
The algorithm efficiently reached optimal convergence in **34 iterations** with a minimized objective function of **10,852,001.55**.
* **Fuzzy Partition Coefficient (FPC):** `0.9156` (A score highly close to 1.0, indicating exceptionally well-defined and compact clusters).
* **Visual Output:** The system outputs a side-by-side comparison of the original MRI, the isolated cluster labels, and a color-mapped segmentation where potential tumor mass is isolated.

## 🔮 Future Development
* **Ground Truth Validation:** Compare algorithmic outputs against expert-annotated masks to calculate exact Accuracy, Specificity, and Dice Coefficients.
* **Skull Stripping Pipeline:** Introduce a pre-processing step to isolate brain tissue from the skull for improved clustering focus.
* **Texture Analysis Integration:** Expand feature extraction beyond pixel intensity to include tissue texture matrices.
  
