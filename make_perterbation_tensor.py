import argparse
import json

import regex as re
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Create Perterbation Tensor From Ground Strings")
parser.add_argument(
    "-i", default="data/groundStrings_autogen.txt", type=str, help="sets the input file name."
)
parser.add_argument(
    "-o", default="data/perterbationTensor.json", type=str, help="sets the output file name"
)

args = vars(parser.parse_args())

with open(args["i"], "r") as f:
    rawStrings = f.read()
strings = re.split("\n", rawStrings)

# -- Truncate Strings for Smol RAM Devices -- #
#  strings = strings[0:500]

vectorArray = []
for s in strings:
    vector = re.findall(r"\s|((?<=\s)\w+|\w+(?=[\s.,?!:;'\"\(\)\-])|[.,?!:;'\"\(\)\-])", s)
    for i, token in enumerate(vector):
        if token == "":  # matches on spaces get returned as empty strings for some reason
            vector[i] = " "
    vectorArray.append(vector)

with open("symbols.json", "r") as f:
    symbols = json.load(f)

# -- generate perterbationTensor -- #

progressBar = tqdm(
    total=sum([len(x) - x.count(" ") for x in vectorArray]) * len(symbols),
    desc="P.Tensor",
    position=0,
)

perterbationTensor = []
for sym in symbols:
    pCube = []
    for s in vectorArray:
        pPlane = []
        for i in range(len(s)):
            pVector = s[::]
            if pVector[i] != " ":
                pVector[i] = sym
                pVector = "".join(pVector)
                pPlane.append(pVector)
                progressBar.update()
        pCube.append(pPlane)
    perterbationTensor.append((sym, pCube))

with open(args["o"], "w") as f:
    json.dump(perterbationTensor, f)

# -- Show the First Little Bit of the Perterbation Tensor -- #
for iA in range(3):
    print("\n\n\nSymbol: ", perterbationTensor[iA][0])
    for iB in range(3):
        print("\nPlane: ", iB)
        print("\nVectors: ")
        for iC in range(3):
            print(perterbationTensor[iA][1][iB][iC])
