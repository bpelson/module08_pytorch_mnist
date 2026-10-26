# Module 08: MNIST Classification with PyTorch

This repository adapts Module 7 Lecture B, Section 3 into Python scripts
using object oriented programming and command line arguments.

The ConvNet class contains two convolutional blocks followed by two fully
connected layers. The model uses cross-entropy loss and the Adam optimizer.
Training uses CUDA when available and CPU otherwise.

## Files

- `utilities.py`: Model, argument parsing, data loading, training, evaluation, and plotting.
- `run.py`: Runs the experiment using the command line arguments.
- `.gitignore`: Excludes Python cache files and the downloaded dataset.

## Installation

```bash
python -m pip install torch torchvision matplotlib tqdm
```

## Run

```bash
python run.py -e 2 -b 1000 -l 0.001
```

The default values are 2 epochs, a batch size of 1000, and a learning rate
of 0.001. Running `python run.py` uses these defaults.

The corresponding arguments are `--num_epochs`, `--batch_size`, and
`--learning_rate`.

## Results

The program prints the model and displays training progress in the terminal.
Training loss and training/testing accuracy are saved in `figure.png`.
