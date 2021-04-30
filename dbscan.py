import argparse
import json
import logging

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.cluster import DBSCAN

#  import plotly.express as px
#  import seaborn as sns

TESTMODE = False

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

with open("colorMapping/RGBcolors.json", "r") as f:
    COLORS = json.load(f)

lowContrastList = [
    "pale golden rod",
    "light cyan",
    "antique white",
    "beige",
    "corn silk",
    "light yellow",
    "misty rose",
    "lavender blush",
    "linen",
    "old lace",
    "papaya whip",
    "sea shell",
    "mint cream",
    "lavender",
    "floral white",
    "alice blue",
    "ghost white",
    "honeydew",
    "ivory",
    "azure",
    "snow",
    "gainsboro",
    "white smoke",
    "white",
]
for color in lowContrastList:
    del COLORS[color]
COLORS = list(COLORS.values())


if not TESTMODE:
    # normalize the validity array
    def heaviside(x, threshold: float):
        return np.heaviside(x + threshold, 0)  # for jaccard & yule

    vHeaviside = np.vectorize(heaviside)

    # rescale all values to be within -100 and 100.
    scaler = preprocessing.RobustScaler(with_centering=False, with_scaling=True).fit(validityArray)
    validityArray = scaler.transform(validityArray)  # scales to between -1 and 1
    validityArray = np.multiply(validityArray, 100)  # just for convenience
    print(validityArray[0][:50])

    #  cut to validity threshold
    validityArray = vHeaviside(validityArray, 5)
    print(validityArray[0][:50])

    #  convert to boolean
    validityArray = np.array(validityArray, np.dtype("bool"))
    print(validityArray[0][:50])


def dbscanWordTaxonomyPyPlot(
    validityArray: list, dimensions: int = 3, testmode: bool = False, eps=0.75
):
    with open("projections/fittest_15.0.json", "r") as f:
        projection = np.array(json.load(f), np.dtype("float"))
    if not testmode:
        m = DBSCAN(
            eps=eps,
            min_samples=3,
            metric="euclidean",
            algorithm="auto",
            leaf_size=30,
            n_jobs=1,
        )
        clustering = m.fit(projection)
        print(clustering.labels_)
        minLabel = min(clustering.labels_)
        maxLabel = max(clustering.labels_)
        colorCodes = []
        for label in clustering.labels_:
            code = round(((label - minLabel) / (maxLabel - minLabel)) * (len(COLORS) - 1))
            colorCodes.append(COLORS[code])
        # save the representation
        #  with open("clustering_" + str(eps) + ".json", "w") as f:
        #      json.dump(clustering.labels_.tolist(), f)
    else:
        with open("clusterings/clustering_" + str(eps) + ".json", "r") as f:
            clustering = np.array(json.load(f), np.dtype("float"))

    if dimensions == 3:
        df = pd.DataFrame(
            {
                "x": projection[:, 0],
                "y": projection[:, 1],
                "z": projection[:, 2],
                #  "colors": [coarse_color_map[tag] for tag in tags]
                "colors": colorCodes,
                "labels": symbols,
            }
        )

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
        fig_3d.write_html("plottest_" + str(eps) + ".html", config=config)

    elif dimensions == 2:
        dataDict = {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "colors": colorCodes,
            #  "tags": tags,
        }
        df = pd.DataFrame(dataDict)
        ax = plt.axes()
        ax.scatter(
            df["x"],
            df["y"],
            c=df["colors"],
            cmap="rgb",
            #  edgecolors="black",
            s=50,
        )
        plt.show()
    elif dimensions == 1:
        dataDict = {
            "x": projection[:, 0],
            "colors": colorCodes,
            #  "tags": tags,
        }
        df = pd.DataFrame(dataDict)
        ax = plt.axes()
        ax.scatter(
            df["x"],
            [0 for x in range(len(df["x"]))],
            c=df["colors"],
            cmap="rgb",
            #  edgecolors="black",
            s=50,
        )
        plt.show()
    #  plt.savefig("data/figures/taxonomy" + str(p) + ".png")
    plt.close()


for eps in range(225, 350, 5):
    #  p = 50
    eps = 0.01 * eps
    print(eps)
    try:
        dbscanWordTaxonomyPyPlot(
            validityArray=validityArray,
            dimensions=3,
            eps=eps,
            testmode=TESTMODE,
        )
    except Exception:
        print("Fail")
