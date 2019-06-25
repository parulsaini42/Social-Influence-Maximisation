import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Run NN")

    parser.add_argument("--input",
                        nargs = "?",
                        default = "./data/grid.csv",
                        help = "Input graph path.")
    parser.add_argument("--json",
                        nargs = "?",
                        default = "./output/grid.txt",
                        help = "vector path.")

    parser.add_argument("--vector",
                        nargs = "?",
                        default = "./data/grid_vector.txt",
                        help = "vector path.")
    parser.add_argument("--epoch",
                        type = int,
                        default = 10,
                        help = "number of epochs")
    parser.add_argument("--batch",
                        type = int,
                        default = 100,
                        help = "batch size")
    parser.add_argument("--hidden",
                        type = int,
                        default = 100,
                        help = "hidden layer size")
    parser.add_argument("--threshold",
                        type = float,
                        default = 0.5,
                        help = "threshold for finding prediction values")
    parser.add_argument("--lr",
                        type = float,
                        default = 0.001,
                        help = "learning rate")

    return parser.parse_args()




