import argparse
import json
import logging

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import regex as re
from sklearn import preprocessing
from sklearn.manifold import TSNE

#  import plotly.express as px
#  import seaborn as sns

TESTMODE = True

try:  # get nlp prereqs
    nltk.download("averaged_perceptron_tagger")  # this sticks around permanently on the system.
except Exception:  # so we dont have to worry about it not updating sometimes.
    pass

parser = argparse.ArgumentParser(description="Validity Tensor Estimation")

parser.add_argument(
    "-i",
    default="data/validityTensor.json",
    type=str,
    help="sets the input validity tensor file",
)

logging.basicConfig(
    filename="logs/t-sne.log",
    level=logging.DEBUG,
    format="[%(asctime)s|%(name)s|t-sne.py|%(levelname)s] %(message)s",
)

args = vars(parser.parse_args())

vtfile = args["i"]

# get validity tensor
with open(vtfile, "r") as f:
    validityTensor = json.load(f)

# join individual ground stings into one string for each symbol
for i, element in enumerate(validityTensor):
    joinedDimensions = []
    for gstring in element[1]:
        joinedDimensions.extend(gstring)
    validityTensor[i][1] = joinedDimensions

# convert tensor to numpy n-dimensional array
symbols = [x[0] for x in validityTensor]
validityArray = np.array([x[1] for x in validityTensor], np.dtype("int"))
print(validityArray[0][:50])

if not TESTMODE:
    # normalize the validity array
    def heaviside(x, threshold: float):
        return np.heaviside(x + threshold, 0)  # for jaccard & yule

    def attention(x):
        #  return np.exp((x+1000)/150)/500
        return 50 + 50 * np.tanh(x / 2.5)
        #  return 0.5 + 0.5*np.tanh((x+1000)/500)  # for jaccard & yule

    vHeaviside = np.vectorize(heaviside)
    vAttention = np.vectorize(attention)

    # rescale all values to be within -100 and 100.
    scaler = preprocessing.RobustScaler(with_centering=False, with_scaling=True).fit(validityArray)
    validityArray = scaler.transform(validityArray)  # scales to between -1 and 1
    validityArray = np.multiply(validityArray, 100)  # just for convenience
    print(validityArray[0][:50])

    #  cut to validity threshold
    validityArray = vHeaviside(validityArray, 5)
    print(validityArray[0][:50])

    # apply attention function
    #  validityArray = vAttention(validityArray)
    #  print(validityArray[0][:50])

    #  convert to boolean
    validityArray = np.array(validityArray, np.dtype("bool"))
    print(validityArray[0][:50])

partsOfSpeech = nltk.pos_tag(symbols)
partsOfSpeech = [x[1] for x in partsOfSpeech]
for i, sym in enumerate(symbols):
    if len(sym) == 1 and re.match(r"[a-zA-Z]", sym):
        partsOfSpeech[i] = "L"

with open("colorMapping/RGBcolors.json", "r") as f:
    COLORS = json.load(f)

fine_color_map = {
    ".": COLORS["black"],  # sentence terminator
    "--": COLORS["dim gray"],  # dash
    "(": COLORS["dim gray"],  # opening parenthesis
    ")": COLORS["dim gray"],  # closing parenthesis
    ",": COLORS["dim gray"],  # comma
    ":": COLORS["dim gray"],  # colon or ellipsis
    "''": COLORS["gray"],  # closing quotation mark
    "$": COLORS["light gray"],  # dollar
    "LS": COLORS["light gray"],  # list marker    1)
    "FW": COLORS["light gray"],  # foreign word
    "UH": COLORS["gray"],  # interjection    errrrrrrrm
    "CD": COLORS["purple"],  # cardinal digit    0, 1, 2, 3, 4, ...
    "L": COLORS["saddle brown"],  # single letter
    "DT": COLORS["dark green"],  # determiner    a, an, the, every
    "PDT": COLORS["forest green"],  # predeterminer    'all the kids'
    "CC": COLORS["lime"],  # coordinating conjuntion    and, but, or
    "IN": COLORS["lime green"],  # preposition/subordinating conjunction
    "TO": COLORS["lawn green"],  # to	go 'to' the store.
    "EX": COLORS["medium spring green"],  # existential there    ("there is" ... "there exists")
    "RP": COLORS["yellow"],  # particle    give up
    "MD": COLORS["orange red"],  # modal could, will
    "JJ": COLORS["slate blue"],  # adjective    'big'
    "JJR": COLORS["medium slate blue"],  # adjective, comparative    'bigger'
    "JJS": COLORS["medium purple"],  # adjective, superlative    'biggest'
    "PRP": COLORS["midnight blue"],  # personal pronoun	I, he, she
    "PRP$": COLORS["blue"],  # possessive pronoun	my, his, hers
    "POS": COLORS["blue"],  # possessive ending    parent\'s
    "NN": COLORS["cyan"],  # noun, singular    'desk'
    "NNS": COLORS["turquoise"],  # noun plural    'desks'
    "NNP": COLORS["navy"],  # proper noun, singular    'Harrison'
    "NNPS": COLORS["dark cyan"],  # proper noun, plural    'Americans'
    "WDT": COLORS["cadet blue"],  # wh-determiner    which
    "WP": COLORS["steel blue"],  # wh-pronoun     who, what
    "WP$": COLORS["corn flower blue"],  # possessive wh-pronoun    whose
    "WRB": COLORS["sky blue"],  # wh-abverb    where, when
    "RB": COLORS["maroon"],  # adverb	very, silently
    "RBR": COLORS["dark red"],  # adverb, comparative    better
    "RBS": COLORS["brown"],  # adverb, superlative    best
    "VB": COLORS["firebrick"],  # verb, base form    take
    "VBD": COLORS["crimson"],  # verb, past tense    took
    "VBG": COLORS["red"],  # verb, gerund/present participle    taking
    "VBN": COLORS["tomato"],  # verb, past participle    taken
    "VBP": COLORS["coral"],  # verb, non-3rd person singular present    take
    "VBZ": COLORS["indian red"],  # verb, 3rd person singular present    takes
}

coarse_color_map = {
    ".": COLORS["black"],  # sentence terminator
    "--": COLORS["black"],  # dash
    "(": COLORS["black"],  # opening parenthesis
    ")": COLORS["black"],  # closing parenthesis
    ",": COLORS["black"],  # comma
    ":": COLORS["black"],  # colon or ellipsis
    "''": COLORS["black"],  # closing quotation mark
    "$": COLORS["yellow"],  # dollar
    "LS": COLORS["yellow"],  # list marker    1)
    "FW": COLORS["yellow"],  # foreign word
    "UH": COLORS["yellow"],  # interjection    errrrrrrrm
    "RP": COLORS["yellow"],  # particle    give up
    "CD": COLORS["purple"],  # cardinal digit    0, 1, 2, 3, 4, ...
    "L": COLORS["saddle brown"],  # single letter
    "DT": COLORS["lime"],  # determiner    a, an, the, every
    "PDT": COLORS["lime"],  # predeterminer    'all the kids'
    "CC": COLORS["lime"],  # coordinating conjuntion    and, but, or
    "IN": COLORS["lime"],  # preposition/subordinating conjunction
    "TO": COLORS["lime"],  # to	go 'to' the store.
    "EX": COLORS["lime"],  # existential there    ("there is" ... "there exists")
    "MD": COLORS["orange"],  # modal could, will
    "JJ": COLORS["blue"],  # adjective    'big'
    "JJR": COLORS["blue"],  # adjective, comparative    'bigger'
    "JJS": COLORS["blue"],  # adjective, superlative    'biggest'
    "PRP": COLORS["cyan"],  # personal pronoun	I, he, she
    "PRP$": COLORS["cyan"],  # possessive pronoun	my, his, hers
    "POS": COLORS["cyan"],  # possessive ending    parent\'s
    "NN": COLORS["cyan"],  # noun, singular    'desk'
    "NNS": COLORS["cyan"],  # noun plural    'desks'
    "NNP": COLORS["cyan"],  # proper noun, singular    'Harrison'
    "NNPS": COLORS["cyan"],  # proper noun, plural    'Americans'
    "WDT": COLORS["cyan"],  # wh-determiner    which
    "WP": COLORS["cyan"],  # wh-pronoun     who, what
    "WP$": COLORS["cyan"],  # possessive wh-pronoun    whose
    "WRB": COLORS["cyan"],  # wh-abverb    where, when
    "RB": COLORS["red"],  # adverb	very, silently
    "RBR": COLORS["red"],  # adverb, comparative    better
    "RBS": COLORS["red"],  # adverb, superlative    best
    "VB": COLORS["pink"],  # verb, base form    take
    "VBD": COLORS["pink"],  # verb, past tense    took
    "VBG": COLORS["pink"],  # verb, gerund/present participle    taking
    "VBN": COLORS["pink"],  # verb, past participle    taken
    "VBP": COLORS["pink"],  # verb, non-3rd person singular present    take
    "VBZ": COLORS["pink"],  # verb, 3rd person singular present    takes
}


def tsneWordTaxonomyPyPlot(
    validityArray: list,
    tags: list,
    perplexity: int = 25,
    learning_rate: int = 5,
    dimensions: int = 3,
    testmode: bool = False,
):
    if not testmode:
        m = TSNE(
            n_components=dimensions,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=36,
            n_iter=50000,
            n_iter_without_progress=1000,
            #  n_iter=2000,
            #  n_iter_without_progress=250,
            min_grad_norm=1e-10,
            #  metric="euclidean",
            metric="jaccard",
            #  metric="yule",
            square_distances=True,
            method="barnes_hut",
            angle=0.0005,
            verbose=True,
        )
        #  tsne_features = m.fit_transform(animalLegs.reshape(-1, 1))  # useful for one dimensional data.
        tsne_features = m.fit_transform(validityArray)
        # save the representation
        with open("fittest_" + str(perplexity) + ".json", "w") as f:
            json.dump(tsne_features.tolist(), f)
    else:
        with open("projections/fittest_" + str(perplexity) + ".json", "r") as f:
            tsne_features = np.array(json.load(f), np.dtype("float"))

    if dimensions == 3:
        df = pd.DataFrame(
            {
                "x": tsne_features[:, 0],
                "y": tsne_features[:, 1],
                "z": tsne_features[:, 2],
                #  "colors": [coarse_color_map[tag] for tag in tags]
                "colors": [coarse_color_map[tag] for tag in tags],
                "labels": symbols,
            }
        )

        #  fig_3d = px.scatter_3d(
        #      df,
        #      x="x",
        #      y="y",
        #      z="z",
        #      #  color="colors",
        #      #  color=[(255, 0, 0) for x in range(500)],
        #      #  labels={'color': 'species'}
        #      mode="marker",
        #      marker=dict(size=15, color=df["colors"]),
        #  )

        fig_3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=df["x"],
                    y=df["y"],
                    z=df["z"],
                    #  color="colors",
                    #  color=[(255, 0, 0) for x in range(500)],
                    #  labels={'color': 'species'}
                    mode="markers",
                    marker=dict(size=12, color=df["colors"], line=None),
                    #  text=df["labels"],
                    hovertext=df["labels"],
                    hovertemplate=" %{hovertext}<extra></extra> ",
                    #  showlegend=True,
                )
            ],
        )

        axis_settings = dict(
            visible=False,
            showbackground=False,
            #  backgroundcolor="grey",
            showaxeslabels=False,
            showspikes=False,
            color="black",
            ticks="",
            #  title=dict(
            #      text="",
            #  ),
            tickfont=dict(color="rgba(0,0,0,0)"),
        )

        fig_3d.update_scenes(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            dragmode="orbit",
        )
        fig_3d.update_layout(
            hoverlabel=dict(
                font_size=32,
            )
        )

        config = dict(displaylogo=False)

        fig_3d.show(config=config)
        fig_3d.write_html("plottest_" + str(perplexity) + ".html", config=config)

        #  exit()

        #  ax = plt.axes(projection="3d")
        #  scatter = ax.scatter3D(
        #      df["x"],
        #      df["y"],
        #      df["z"],
        #      c=[fine_color_map[tag] for tag in tags],
        #      #  c=[[1,0,0] for tag in tags],
        #      cmap="rgb",
        #      #  edgecolors="black",
        #      s=50,
        #  )
        #  plt.show()

    elif dimensions == 2:
        dataDict = {
            "x": tsne_features[:, 0],
            "y": tsne_features[:, 1],
            #  "tags": tags,
        }
        df = pd.DataFrame(dataDict)
        ax = plt.axes()
        ax.scatter(
            df["x"],
            df["y"],
            c=[[i / 255.0 for i in coarse_color_map[tag]] for tag in tags],
            cmap="rgb",
            #  edgecolors="black",
            s=50,
        )
        plt.show()
    elif dimensions == 1:
        dataDict = {
            "x": tsne_features[:, 0],
            #  "tags": tags,
        }
        df = pd.DataFrame(dataDict)
        ax = plt.axes()
        ax.scatter(
            df["x"],
            [0 for x in range(len(df["x"]))],
            c=[[i / 255.0 for i in coarse_color_map[tag]] for tag in tags],
            cmap="rgb",
            #  edgecolors="black",
            s=50,
        )
        plt.show()
    #  plt.savefig("data/figures/taxonomy" + str(p) + ".png")
    plt.close()


for p in range(20, 100, 5):
    #  p = 50
    p = 0.5 * p
    print(p)
    tsneWordTaxonomyPyPlot(
        validityArray=validityArray,
        tags=partsOfSpeech,
        perplexity=p,
        learning_rate=10,
        dimensions=3,
        testmode=TESTMODE,
    )
