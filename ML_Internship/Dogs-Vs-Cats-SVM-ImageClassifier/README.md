#  Dogs vs. Cats - SVM Image Classifier

This is a simple image classification project that uses Support Vector Machine (SVM) to distinguish between images of cats and dogs. The model is trained using the popular [Dogs vs. Cats dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data) provided by Kaggle.



##  Project Structure

Dogs-vs-Cats SVM/
├── dogs_vs_cats_svm.py # Python script containing the model code
├── requirements.txt # Python dependencies (optional)
├── .gitignore # Files/folders excluded from version control
├── README.md # Project documentation
└── Data/
└── train/
├── cat.0.jpg
├── cat.1.jpg
├── ...
├── dog.0.jpg
├── dog.1.jpg
└── ...



# Dataset

The dataset used for this project is **Dogs vs. Cats**, available on Kaggle:
 https://www.kaggle.com/competitions/dogs-vs-cats/data

Due to GitHub's file size limitations, the dataset is **not included** in this repository.

>  The full dataset is over 500 MB. GitHub does not support uploading large datasets directly, so you need to download it manually.

# Steps to Download and Set Up:

1. Go to the [Kaggle competition page](https://www.kaggle.com/competitions/dogs-vs-cats/data)
2. Download the `train.zip` file
3. Extract the zip file
4. Move the extracted `train/` folder inside the `Data/` directory of this project

After this, your directory should look like:

Dogs-vs-Cats SVM/
└── Data/
└── train/
├── cat.0.jpg
├── dog.0.jpg
└── ...


##  How It Works

- The script loads 500 images of dogs and 500 images of cats from the dataset.
- Each image is resized to 64x64 and flattened into a feature vector.
- The SVM model is trained on the processed features.
- A few sample predictions are shown using matplotlib.



##  How to Run

1. Make sure Python is installed (recommended: Python 3.9+)
2. Install the dependencies:


pip install -r requirements.txt
Run the script:


python dogs_vs_cats_svm.py
 Requirements
The project uses the following Python libraries:

opencv-python
scikit-learn
matplotlib
numpy
If not using the requirements.txt file, you can install them manually:


pip install opencv-python scikit-learn matplotlib numpy
 Output
The script will show sample prediction images like:


Predicted: dog, True: dog
Predicted: cat, True: cat
With the image preview.

 Notes
This is a simple classifier using only pixel values. For better accuracy, you can explore CNNs or transfer learning with deep models like VGG16, ResNet, etc.

Dataset is not pushed to GitHub to avoid issues with large files.
