# Gender Classification Model with PETA Dataset

이 프로젝트는 PETA 데이터셋을 사용하여 성별 분류 모델을 학습합니다. 코드에는 Ray Tune을 사용한 자동 하이퍼파라미터 튜닝과 학습 결과를 관리하기 위한 MLflow 통합이 포함되어 있습니다. 모델은 Hugging Face의 timm 라이브러리의 모델을 사용합니다.

This project trains a gender classification model using the PETA dataset. The code includes automatic hyperparameter tuning with Ray Tune and MLflow integration for managing learning results. The model uses a model from Hugging Face's timm library.

<br/>


## Execution Environment

To execute this project, you need the following software installed:
- Python 3.8 or higher
- CUDA (if you plan to use GPU acceleration)
- pip for package management

<br/>

## Required Python Packages

Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

<br/>

## Project Structure

The project structure should look like this:    

        .
        ├─── dataset/PETA/PETAdataset/(데이터셋명)/archive/
        │  └── (이미지파일).*
        │  └── Label.txt
        ├─── module_test/
        │  ├── test_lit_model.py
        │  ├── test_mlflow.py
        │  └── test_ray.py
        ├── config.yaml
        ├── dataloader.py
        ├── ImageClassifier.py
        ├── main.py
        ├── README.md
        ├── requirements.txt
        └── Hugging Face_timm_model_list.txt

### Detailed Descriptions
- config.yaml
    - 모델, 데이터셋, MLflow, 학습, 검증 및 하이퍼파라미터 등 설정과 관련된 모든 값을 포함하고 있습니다.
    - Contains configuration settings for the dataset, MLflow, training, validation, and hyperparameters. Ensure the paths and other parameters match your setup.
- dataloader.py
    - PETA 데이터셋에서 이미지를 로드하고 처리하는 PETADataset 클래스를 포함하고 있습니다. Data Augmentation과 관련한 내용이 포함되어 있습니다.
    - Contains the `PETADataset` class, which loads and processes images from the PETA dataset. It includes functions for data augmentation and transformation.
- ImageClassifier.py
    - Hugging Face의 timm 라이브러리의 모델을 사용한 이미지 분류를 위한 PyTorch Lightning 모듈인 ImageClassifier 모델 클래스를 정의합니다.
    - Defines the `ImageClassifier` class, which is a PyTorch Lightning module for image classification using models from the timm library.
- main.py
    - Ray Tune을 사용한 하이퍼파라미터 튜닝과 MLflow를 사용한 실험 추적을 통합합니다. 데이터 로더, 모델 및 학습 설정을 구성합니다.
    - Integrates Ray Tune for hyperparameter tuning and MLflow for experiment tracking. It sets up the data loaders, model, and training configuration.

<br/>

## Configurations
The `config.yaml` file contains all values related to settings, including model, dataset, MLflow, training, validation, and hyperparameters. Make sure the paths and other settings match your environment.


<br/>

## Dataset 
###  PETA (PEdesTrian Attribute dataset)
The PEdesTrian Attribute dataset (PETA) is a dataset for recognizing pedestrian attributes, such as gender and clothing style, at a far distance. It is of interest in video surveillance scenarios where face and body close-shots are hardly available. It consists of 19,000 pedestrian images with 65 attributes (61 binary and 4 multi-class). Those images contain 8705 persons.
- 보행자 속성 인식 데이터 세트
- 19,000개의 보행자 이미지
- 65개 속성(61개 바이너리 및 4개 멀티클래스)
- 8,705명이 포함
- 10개의 공개된 데이터셋을 통합

Homemape : https://mmlab.ie.cuhk.edu.hk/projects/PETA.html   
Download : https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?e=1&dl=0

<br/>

## How to Execute
1. Set up MLflow Tracking Server:
Ensure you have an MLflow tracking server running at the URI specified in config.yaml. You can start an MLflow server locally using:
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    ```

2. Run the Training Script:
    - This will start the training process with hyperparameter tuning using Ray Tune and log the results to the MLflow server.
    ```bash
    python main.py
    ```
<br/>

## Viewing Results
### TensorBoard for Scalability Metrics
- TensorBoard is used for viewing the scalability metrics of the model training process. To view these metrics, start TensorBoard and navigate to the following URL:
    ```bash
    tensorboard --logdir <your_log_directory> --port 6006
    ```
  Then, open your web browser and go to: http://127.0.0.1:6006/#scalars

<br/>

### Ray Tune Dashboard
- Ray Tune provides a dashboard to monitor the hyperparameter tuning process. Start the Ray dashboard using:

    ```bash
    ray dashboard --port 8265
    ```
    Then, open your web browser and go to: http://127.0.0.1:8265/#/jobs/01000000

### MLflow UI
- MLflow's user interface allows you to track experiments, view metrics, and compare runs. Start the MLflow UI using:
    ```bash
    mlflow ui --port 5000
    ```
    Then, open your web browser and go to: http://127.0.0.1:5000

<br/>