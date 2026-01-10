from  argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", default="xhfg29")

arg = parser.parse_args()

arg.name

