import argparse
import json
import logging
import os

import nltk
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Bulk Radius Production")
parser.add_argument("-n", default=1000, type=int, help="sets the number of strings to select")
parser.add_argument("-i", default="samples/", type=str, help="sets the input folder name")
parser.add_argument(
    "-o", default="data/groundStrings_selected.json", type=str, help="sets the output folder name"
)

args = vars(parser.parse_args())

logging.basicConfig(
    filename="logs/gs_select.log",
    level=logging.DEBUG,
    format="[%(asctime)s|%(name)s|ground_string_select.py|%(levelname)s] %(message)s",
)

try:  # get natural language processing prereqs
    nltk.download("punkt", quiet=True)  # this sticks around permanently on the system.
except Exception:  # so we dont have to worry about it not updating sometimes.
    pass


def select_ground_strings(
    count: int = 1000, inDir: str = "samples/", outDir: str = "data/groundStrings_selected.json"
):
    samples = os.listdir(inDir)
    fileSizes = [os.path.getsize(inDir + x) for x in samples]
    sampleSizeMap = dict(zip(samples, fileSizes))
    progressBar = tqdm(total=count, desc="Total", position=0)
    weightedSamples = np.random.choice(
        samples, size=50 * len(samples), p=[x / sum(fileSizes) for x in fileSizes]
    )
    gstrings = []
    chunksize = 18 * 6 * 10  # read roughly 10 sentences worth of characters at once
    chunk = ""
    for _ in range(count):
        while True:
            s = np.random.choice(weightedSamples, size=1)[0]
            with open(inDir + s, "r") as f:
                f.seek(np.random.randint(0, sampleSizeMap[s] - chunksize))
                chunk = f.read(chunksize)
            split_text = nltk.tokenize.sent_tokenize(chunk)
            try:
                protostring = split_text[1]
                if len(protostring) <= 1 or '"' in protostring:
                    raise IndexError
                print(protostring)
                gstrings.append(protostring)
                progressBar.update()
                break
            except IndexError:  # try again if split_text is too short
                continue
    with open(outDir, "w") as f:
        json.dump(gstrings, f)


if __name__ == "__main__":
    select_ground_strings(count=args["n"], inDir=args["i"], outDir=args["o"])
