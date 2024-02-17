import argparse

def get_flags():
    parser = argparse.ArgumentParser(description="Please, specify run arguments.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model-path", help="path to the model")
    args = parser.parse_args()
    config = vars(args)
    return config