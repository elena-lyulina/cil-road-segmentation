# ETHZ Spring 2024 CIL Road Segmentation Project

### Deadline:
✨July 31st 23:59✨

### Environment Setup
To replicate our experiments, follow these steps to create and activate a virtual environment:

#### For Windows:
```bash
python -m venv ./cil-rs
cil-rs\Scripts\activate
```

#### For MacOS and Linux:
```bash
python -m venv ./cil-rs
source cil-rs/bin/activate
```

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

Note: We run our experiments on Python version 3.11.5. We recommend users use the correct Python executable to initiate the virtual environment.

### Directory Structure
```
CIL-ROAD-SEGMENTATION-2024
└───cil-rs
    |───data
    |   |───test
    |   |   └───images
    |   └───training
    |       |───groundtruth
    |       └───images
    |───docs
    |───notebook
    |───out
    └───src
        |───models
        └───submission
```

### Links: 
1. [Video recording](https://video.ethz.ch/lectures/d-infk/2024/spring/263-0008-00L/fe8cb982-d061-4350-8c3e-26b0cdb43119.html) of the preparatory session about this project
2. [Kaggle link](https://www.kaggle.com/t/0fe22c50cf504e64b2decda075f71c87)
3. [Massachusetts Road Dataset download](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset/data?select=tiff)
