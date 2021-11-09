
# Revisiting Model Fairness via Adversarial Examples

Authors: **Tao Zhang**, Tianqing Zhu, Jing Li, Wanlei Zhou

This is the code for paper 'Revisiting Model Fairness via Adversarial Examples'. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
- Download .csv files for 4 datasets used in the paper from `data` directory

## Preprocess
- Data preprocessing is implemented in the prepare_data.py.

## Training
- Two adversarial attack methods, LowProFool and DeepFool, are implemented in the Adverse.py.

- Training and evaluation are implemented in the train_model.py.

- Demographic parity is implemented in the Fairness_metrics.py.

## Evaluation
- All evaluation metrics are implemented in the Metrics.py.


## Results
- We provide an example to train models and obtain experimental results on the `German` dataset with sensitive attribute `age` in the Playground_German.ipynb.
