# Attention Is All You Need

[Attention is All You Need](https://arxiv.org/abs/1706.03762) PyTorch implementation based on [Umar Jamil's Coding a Transformer from Scratch video](https://www.youtube.com/watch?v=ISNdQcPhsts&t=101s&ab_channel=UmarJamil).

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

## Model card
- Vanilla Transformer Model
- Batch size: 8
- d_model: 512
- seq_len: 350
- lr = 10e-4
- Optimizer: Adam

The model was initially trained for English-to-Italian and English-to-French translations but with limited resources. The author trained these models for 20 epochs each; you can continue training from the provided checkpoints if needed.

## Resume Training
To resume training from a checkpoint:
- Create a `weights` directory in the root folder.
```bash
mkdir weights
```
- Download the model weights from this [link](https://universityofadelaide.box.com/s/sf952gnruu8c752norn03st3t1ax3mq4) and place them in the `weights` directory.
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