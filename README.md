# Persian Digits Recognition
Identifying Farsi (Arabic) Scanned Handwritten Digits

## In this repository

- Scanned input images: [image1.jpg](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/image1.jpg), [image2.jpg](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/image2.jpg), [image3.jpg](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/image3.jpg), [image4.jpg](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/image4.jpg), [image5.jpg](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/image5.jpg)
- Convert images to CSV and JSON datasets (with OpenCV): [create_dataset_from_images.py](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/create_dataset_from_images.py)
- Some prepared CSV sets: [large_dataset_validation.csv](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/large_dataset_validation.csv), [large_dataset_train.csv](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/large_dataset_train.csv)
- An unlabeled dataset for testing: [large_dataset_test.csv](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/large_dataset_test.csv)
- A modified version of [PyTorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py): [main.py](https://github.com/Erfaniaa/persian-digits-recognition/blob/master/main.py)

## Install requirements

- Install [OpenCV](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html).

- Install Python packages:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

- Converting scanned images to CSV and JSON datasets:

  ```bash
  python create_dataset_from_images.py
  ```

- Using large datasets for training, validation and testing:

  ```bash
  python main.py
  ```

## Notes

Use Python version 3.
