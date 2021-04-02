import argparse
import json
import logging
import random

import numpy as np
import torch
from decouple import config
from tqdm import tqdm

from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight

#  import os
#  import torch.nn.functional as F
#  from array import array

parser = argparse.ArgumentParser(description="Validity Tensor Estimation")
parser.add_argument(
    "-gs", default="data/groundStrings.json", type=str, help="sets the input grond string file"
)
parser.add_argument(
    "-pt",
    default="data/perterbationTensor.json",
    type=str,
    help="sets the input perterbation tensor file.",
)
parser.add_argument(
    "-gvi",
    default="data/groundValidityTensor.json",
    type=str,
    help="sets the input ground validity tensor file.",
)
parser.add_argument(
    "-gvo",
    default="data/groundValidityTensor.json",
    type=str,
    help="sets the output ground validity tensor file.",
)
parser.add_argument(
    "-vo",
    default="data/validityTensor.json",
    type=str,
    help="sets the output validity tensor file.",
)
parser.add_argument(
    "-d",
    type=str,
    help="Sets the device to use.\n"
    "Choices: 'gpu' for GPU, 'cpu' for CPU\n"
    "(If left blank defaults to 'DEVICE' entry in .env file.)\n",
)
args = vars(parser.parse_args())

logging.basicConfig(
    filename="logs/validtyTensor.log",
    level=logging.DEBUG,
    format="[%(asctime)s|%(name)s|make_validity_tensor.py|%(levelname)s] %(message)s",
)

if args["d"]:
    device_choice = args["d"]
else:
    device_choice = config("DEVICE")

print("\nDEVICE:", device_choice, "\n")

if device_choice == "gpu" and not torch.cuda.is_available():
    print("CUDA unavailable, defaulting to CPU.")
    device_choice = "cpu"

if device_choice == "gpu":
    print("gpu accellerated")
else:
    print("cpu bound")

state_dict = torch.load(
    config("MODEL_LOCATION"),
    map_location="cpu" if (not torch.cuda.is_available() or device_choice == "cpu") else None,
)

print("\nValidity Tensor Estimation\n")

# -- Setting up PyTorch Information -- #
seed = random.randint(0, 2147483647)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
# device = torch.device("cpu")
device = torch.device("cuda" if (torch.cuda.is_available() and device_choice == "gpu") else "cpu")

known_configurations = {
    "s_ai": GPT2Config(),
    "xl_ai": GPT2Config(
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1600,
        n_layer=48,
        n_head=25,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    ),
}

# -- Load Model -- #
gpt2_config = known_configurations[config("MODEL_NAME")]
model = GPT2LMHeadModel(gpt2_config)
model = load_weight(model, state_dict)
model.share_memory()
model.to(device)
model.eval()

# -- serving BrainSqueeze resources. --#


def tokenize(text: str):
    enc = get_encoder()
    tokens = enc.encode(text)
    return tokens


def detokenize(tokens: iter):
    enc = get_encoder()
    text = enc.decode(tokens)
    return text


def firstMismatch(tokensA: iter, tokensB: iter):
    # assumes tokensA is shorter than, or as long as, tokensB.
    for i in range(len(tokensA)):
        if tokensA[i] != tokensB[i]:
            return i
    return None


def firstMismatchInclusive(tokensA: iter, tokensB: iter):
    # makes no assumptions about the lengths of tokensA and tokensB.
    for i in range(min(len(tokensA), len(tokensB))):
        if tokensA[i] != tokensB[i]:
            return i
    return min(len(tokensA), len(tokensB))


def predictedDistribution(
    model=model,
    start_token=50256,
    batch_size=1,
    tokens=None,
    temperature: float = None,
    top_k=1,
    device=device,
):
    """returns a probability distribution for the next byte-pair encoding"""
    if tokens is None:
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    elif type(tokens) is torch.Tensor:
        context = tokens.unsqueeze(0).repeat(batch_size, 1)
    else:
        context = (
            torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        )
    prev = context
    past = None
    with torch.no_grad():
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :]

    return logits[0]


def errorSeries(tokens: list, pbar: tqdm):
    radii = []
    # get first radius (special case)
    logits = predictedDistribution(start_token=50256)  # 50256 => <|endoftext|>
    prob = logits[tokens[0]]
    clamped = torch.clamp(logits, min=prob, max=None)
    clamped.add_(-prob)
    radius = torch.count_nonzero(clamped).item()
    radii.append(radius)
    if pbar is not None:
        pbar.update(1)

    # get all following radii
    for i in range(1, len(tokens)):
        logits = predictedDistribution(tokens=tokens[:i])
        prob = logits[tokens[i]]
        clamped = torch.clamp(logits, min=prob, max=None)
        clamped.add_(-prob)
        radius = torch.count_nonzero(clamped).item()
        radii.append(radius)
        if pbar is not None:
            pbar.update(1)
    return radii


def partialErrorSeries(tokens: list, start: int):
    def getRadius(logits, token):
        prob = logits[token]
        clamped = torch.clamp(logits, min=prob, max=None)
        clamped.add_(-prob)
        radius = torch.count_nonzero(clamped).item()
        return radius

    radii = []
    if start == 0:
        # get first radius (special case)
        logits = predictedDistribution(start_token=50256)  # 50256 => <|endoftext|>
        radius = getRadius(logits, tokens[0])
        radii.append(radius)

        # then get all following radii
        for i in range(1, len(tokens)):
            logits = predictedDistribution(tokens=tokens[:i])
            radius = getRadius(logits, tokens[i])
            radii.append(radius)
        return radii
    else:
        for i in range(start, len(tokens)):
            logits = predictedDistribution(tokens=tokens[:i])
            radius = getRadius(logits, tokens[i])
            radii.append(radius)
        return radii


def calculateGroundValidityTensor(groundStrings: iter):
    gvBar = tqdm(total=len(groundStrings), desc="GroundValidity", position=0)
    gvTen = []
    coder = get_encoder()
    for gs in groundStrings:
        tokens = coder.encode(gs)
        radii = errorSeries(tokens, None)
        gvTen.append(radii)
        gvBar.update()
    return gvTen


def calculateValidityTensor(
    groundTokens: iter, groundValidityTensor: iter, perterbationTensor: iter
):
    # iterate through each file in inDir
    totalBar = tqdm(total=len(perterbationTensor), desc="Total", position=0)
    symbolBar = tqdm(total=len(perterbationTensor[0][1]), desc="TBD", position=1)
    vectorBar = tqdm(total=len(perterbationTensor[0][1][0]), desc="Vector", position=2)

    coder = get_encoder()
    validityTensor = []
    for sym, plane in perterbationTensor:
        logging.info("Started Symbol: " + sym)
        symbolBar.reset()
        symbolBar.set_description(sym)
        vPlane = []
        for i, vector in enumerate(plane):
            vVector = []
            vectorBar.reset(total=len(vector))
            for pString in vector:
                # tokenize pString
                pTokens = coder.encode(pString)
                # locate departure form ground tokens
                departure = firstMismatch(pTokens, groundTokens[i])
                if departure is not None:
                    # sum error up to agreement with groundTokens
                    agreement = sum(groundValidityTensor[i][:departure])
                    # calculate validity of peterbed string from departure onward
                    departureValidity = partialErrorSeries(pTokens, departure)
                    # calculate total validity
                    validity = agreement + sum(departureValidity)
                    # compare to ground validity
                    validity_delta = (
                        sum(groundValidityTensor[i]) - validity
                    )  # lower validity is better
                else:
                    validity_delta = 0
                vVector.append(validity_delta)
                vectorBar.update()
            vPlane.append(vVector)
            symbolBar.update()
        validityTensor.append((sym, vPlane))
        totalBar.update()
        logging.info("Finished Symbol: " + sym)
        with open(args["vo"], "w") as f:  # save checkpoint
            json.dump(validityTensor, f)
    vectorBar.close()
    symbolBar.close()
    totalBar.close()
    return validityTensor


if __name__ == "__main__":

    #  with open(args["gs"], "r") as f:
    #      groundStrings = json.load(f)

    #  gvTen = calculateGroundValidityTensor(groundStrings)
    #  with open(args["gvo"], "w") as f:
    #      json.dump(gvTen, f)

    with open(args["gs"], "r") as f:
        groundStrings = json.load(f)

    groundTokens = []
    coder = get_encoder()
    for gs in groundStrings:
        groundTokens.append(coder.encode(gs))

    with open(args["gvi"], "r") as f:
        groundValidity = json.load(f)

    with open(args["pt"], "r") as f:
        perterbationTensor = json.load(f)

    vt = calculateValidityTensor(groundTokens, groundValidity, perterbationTensor)
    print("\n\n\n### --- SUCCESS! --- ###\n\n\n")
