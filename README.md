## About  

---

The Flask web application that solves the image classification problem for e-commerce (classifier of vegetables and fruits).

The classifier can define 44 classes, from an apple to a watermelon.

Structure of a project: 

```
.
├── Dockerfile
├── README.md
├── app.py
├── model.pth
├── requirements.txt
├── start.sh
├── static
│   ├── css
│   │   ├── background1.jpg
│   │   ├── logo1.jpeg
│   │   └── style.css
│   └── uploads
├── templates
│   ├── index.html
│   └── result.html
├── train
│   └── Train_model.ipynb
└── utils.py
```

## Getting started 

--- 

These instructions will help you get this project up and running on your local machine. 

(_Only for MacOS or Linux_)

1. Clone the repository 
2. In command line: 

```commandline
chmod  +x ./start.sh 
./start/sh
```

3. Start your internet browser and open the link: 
```commandline
http://127.0.0.1:5000
```

**(Optional)** Build the app's container image for deploying on the server:

```commandline
docker build -t flask-app .
```

start an app container 

```commandline
docker run -dp 127.0.0.1:5000:5000 flask-app
```

## Details 

--- 

Notebook with pre-processing data and training model is located [here](https://github.com/serebrennikov94/vegetables_and_fruits_classification/blob/main/train/Train_model.ipynb). 

In this project, transfer learning was used with the ResNet50 model, which was pre-trained on the ImageNet dataset. 
This model was trained on a dataset combining two datasets from Kaggle:

* [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) 
* [Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)


## License 

--- 

This project is licensed under the Apache License - see the [LICENSE](https://github.com/serebrennikov94/vegetables_and_fruits_classification/blob/main/LICENSE) file for details.



