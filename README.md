# Module 08: MNIST Classification with PyTorch

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

The default arguments are 2 epochs, a batch size of 1000, and a learning rate
of 0.001.

The arguments calls are `--num_epochs`, `--batch_size`, and
`--learning_rate`.

## Results

The program prints the model and displays training progress in the terminal.
Training loss and training/testing accuracy are saved in `figure.png`.
