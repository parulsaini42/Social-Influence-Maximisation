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


    


    return parser.parse_args()




