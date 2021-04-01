import argparse
import logging
import random

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from decouple import config
from tqdm import tqdm

from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight

parser = argparse.ArgumentParser(description="Automatic Ground String Generation")
parser.add_argument("-o", default="groundStrings.txt", type=str, help="sets the output folder name")
parser.add_argument("-t", default=0.7, type=float, help="sets the temperature")
parser.add_argument("-top_k", default=50, type=int, help="sets the temperature")
parser.add_argument("-n", default=50, type=int, help="sets the number of ground strings to output")
parser.add_argument("-p", default="<|endoftext|>", type=str, help="sets the prompt to start from")
parser.add_argument(
    "-d",
    type=str,
    help="Sets the device to use.\n"
    "Choices: 'gpu' for GPU, 'cpu' for CPU\n"
    "(If left blank defaults to 'DEVICE' entry in .env file.)\n",
)

args = vars(parser.parse_args())

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

try:  # get natural language processing prereqs
    nltk.download("punkt", quiet=True)  # this sticks around permanently on the system.
except Exception:  # so we dont have to worry about it not updating sometimes.
    pass

logging.basicConfig(
    filename="gs_gen.log",
    level=logging.DEBUG,
    format="[%(asctime)s|%(name)s|ground_string_gen.py|%(levelname)s] %(message)s",
)

state_dict = torch.load(
    config("MODEL_LOCATION"),
    map_location="cpu" if (not torch.cuda.is_available() or device_choice == "cpu") else None,
)

print("\nAutomatic Ground String Generation\n")

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


# -- Tools -- #
def count_sentences(text: str):
    split_text = nltk.tokenize.sent_tokenize(text)
    return len(split_text)


def trim_to_sentence(text: str):
    rev_stops = (
        text[::-1].find("."),
        text[::-1].find("!"),
        text[::-1].find("?"),
        text[::-1].find("\n"),
    )  # finding index of last punctuation
    endstop = len(text) - min([x for x in rev_stops if x >= 0])
    return text[0:endstop]


def generate_ground_strings(
    model=model,
    length=-1,
    sentences=-1,
    start_token=None,
    batch_size=1,
    context=None,
    temperature=1,
    top_k=0,
    device=device,
    sample=True,
):
    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
        context = (
            torch.tensor(context, device=device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    enc = get_encoder()
    elaboration = ""
    with torch.no_grad():
        # -- Check Context Size -- #
        if length == -1:
            length = gpt2_config.n_ctx // 2
        elif length > gpt2_config.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % gpt2_config.n_ctx)
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            elab_fragment = enc.decode([output[0].tolist()[-1]])
            if (elab_fragment == "<|endoftext|>") or ((i + 1) == length) or ("\n" in elab_fragment):
                # try again
                prev = context
                i = 0
                continue
            elaboration += elab_fragment
            # break at endoftext tag or length limit reached
            # limit number of sentences if parameter set
            if sentences != -1:
                possible_sentence = False
                # see if this is possibly the end of a sentence
                for char in elab_fragment:
                    if char in ".?!":  # checking for end punctuation
                        possible_sentence = True
                        break
                if possible_sentence:
                    sentence_count = count_sentences(elaboration)
                    if sentence_count >= sentences:  # stop elaborating, trim, and return.
                        # trim message to last punctuation, and return it.
                        elaboration = trim_to_sentence(elaboration)
                        return elaboration


def top_k_logits(logits, k):
    """returns top_K logits for inference tasks"""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )


if __name__ == "__main__":
    filename = args["o"]
    with open(filename, "w") as f:  # this deletes the contents of the file
        pass
    progressBar = tqdm(total=args["n"], desc="Total", position=0)
    for i in range(args["n"]):
        gstring = generate_ground_strings(
            start_token=50256,  # <|endoftext|> token
            length=100,
            sentences=1,
            top_k=args["top_k"],
            temperature=args["t"],
        )
        if not gstring:
            i -= 1
            continue
        with open(filename, "a") as f:
            f.write(gstring + "\n")
        progressBar.update()
