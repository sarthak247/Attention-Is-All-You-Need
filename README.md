# Attention Is All You Need

[Attention is All You Need](https://arxiv.org/abs/1706.03762) PyTorch implementation based on [Umar Jamil's Coding a Transformer from Scratch video](https://www.youtube.com/watch?v=ISNdQcPhsts&t=101s&ab_channel=UmarJamil).

![Attention Is All You Need Transformer Architecture](attention.png)

## Getting Started

To set up and run this project:

- Create a new environment with the provided `requirements.txt` file:
  ```bash
  virtualenv venv
  source venv/bin/activate
  pip3 install -r requirements.txt

- Log in to your wandb account to track the training process:
```
wandb login
```

- Start model training
```bash
python train.py
```
- Optionally, redirect Python output to a log file:
```bash
python train.py > log.txt
```
- The default hyperparameters are defined in `config.py` and can be modified as needed.

## Model card (for English-Italian)
- Vanilla Transformer Model
- Batch size: 8
- d_model: 512
- seq_len: 350
- lr = 10e-4
- Optimizer: Adam


## Model card (for English-Russian)
- Vanilla Transformer Model
- Batch size: 16
- d_model: 512
- seq_len: 250
- lr = 10e-4
- Optimizer: Adam

The model was initially trained for English-to-Italian and English-to-Russian translations but with limited resources. The author trained these models for 20 epochs each; you can continue training from the provided checkpoints if needed. The logs for the training can be found [here](https://wandb.ai/sarthak-thakur/Transformers)

## Resume Training
To resume training from a checkpoint:
- Create a `weights` directory in the root folder.
```bash
mkdir weights
```
- Download the model weights from this [link](https://universityofadelaide.box.com/s/jjmlphigf0lyb2icjmo5gsti4dlcm2qw) and place them in the `weights` directory.
- The directory structure should resemble:
```bash
tree weights/
weights/
├── tmodel_00.pt
├── tmodel_01.pt
├── tmodel_02.pt
...
├── tmodel_18.pt
└── tmodel_19.pt

0 directories, 20 files
```
- Continue training as usual.
```bash
python train.py
```

## Demo App
This implementation features a demo app built on Streamlit for model inference. Please ensure that you have correctly placed the appropriate model weights in the `weights` folder. For instance, if you intend to utilize the English-to-Russian model, download the corresponding weights and place them in the weights folder. Once the weights are correctly situated, you can initiate model inference by executing:
```bash
streamlit run app.py
```