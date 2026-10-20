from src.train import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
print(train(quick=args.quick))
