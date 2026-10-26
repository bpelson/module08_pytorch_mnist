#@title Main run
from utilities import get_arguments, main


if __name__ == '__main__':
    args = get_arguments()
    main(args.num_epochs, args.batch_size, args.learning_rate)