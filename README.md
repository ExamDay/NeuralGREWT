<h1 align="center">NeuralGREWT:<br>Neural Grammar Rule Extraction and Word Taxonomy</h1>
Unsupervised learning of grammar rules and fuzzy symbol categories by stochasic neighbor embedding
of string probabilities derived from a natural language generation model.

## Theory
A <em>grammar</em> is a set of rules for how symbols may be arranged to form valid strings in a
language. Strings of symbols that follow all grammar rules are grammatically <em>valid</em>.
Grammatic validity is necessary for, but does not imply, sensicality.
For example: the string of symbols (in this case a sentence of words)
"The hat runs and cooks into a hairdresser." is a grammatically valid sentence that
is also nonsensical in isolation ("in isolation" meaning in the absence of explanatory context).

Given this definition of grammatic validity, valid sentences will be
more common than invalid ones in any string-set comprised mostly of sensical strings; making it
possible to infer a string's validity from its probability.

Therefore, it may be possible to learn grammar rules of any language, including computer,
fictional, and extra-terrestrial languages, without needing to make sense of
anything written in the given language (so long as we are in possession of an ample body of text that
would be sensical to a speaker of that language).

Let's test this hypothesis.

One good first step to learning grammar rules might be to identify the different categories of symbols present
in a given language.

Symbols will be said to share a syntactic <em>category</em> in proportion to their mutual interchangability.
In other words: 2 symbols share a category in proportion to the probability that one can be
replaced by the other in a randomly chosen valid string without rendering that string invalid.

Knowing this, we can use a natural language generator (predictor from context really) to determine the degree of
mutual interchangabilty of symbols in a language using total string likelihood before and after replacement
as an indicator of validity. Once we have these validity scores we can use T-SNE to infer discrete symbol
categories from the degree of mutual interchangability (or equivilently the clustering of "vadlity-under-replacement
scores").

## Technique Overview
- Compile a large <em>corpus</em> of valid and/or sensical writings in the chosen language.
- Train a natural language model, or <em>predictor</em>, on the corpus until it is very good at predicting
symbols in the language given context. Anything like BERT or GPT-2 will do fine, and in fact are probably overkill.
- Compile a large "ground set" of valid and/or meaningful strings in the chosen language.
(in the case of unknown languages, this ground set can just be a random subset of the corpus)
- Compile a "symbol set" of all symbols in the chosen language, or at least a large
set of the most common ones.
- Create a 4 dimensional <em>perturbation tensor</em> from the ground and symbol sets by:
    - Add to the perturbation tensor a <em>cube</em> for each <em>symbol</em> in the symbol set;
    construct each cube by stacking a <em>plane</em> for each <em>ground string</em> in the ground
    set; construct each plane by stacking a <em>vector</em> for all strings that can be
    created by replacing any one item in the ground string with the symbol. Of course, making
    sure to do this in the same order for all vectors.
    (In an actual tensor all such vectors need to be padded to uniform dimension with null strings
    ―  In practice we can use a 3 dimensional tensor of pointers to vectors of variable dimension)
- Create a 4 dimensional <em>validity tensor</em> from the perturbation tensor by:
    - For each vector in the perturbation tensor, judge the probability
    of that vector by summing the relative likelihoods of each symbol appearing at
    its location given all previous symbols in the vector (using the predictor) and dividing by the
    length of that vector. That division might be unnecessary, but we will find out.
- Perform T-SNE followed by PCA on the validity tensor to infer number and relative
importance of symbol categories.
- Name each symbol category. (can be totally arbitrary)
- Create a <em>sym-cat</em> mapping of each symbol to a list of its categories
sorted in descending order of the symbol's <em>belongingness</em> to each category.
- Create a <em>cat-sym</em> mapping of each category to a list of its symbols sorted in descending order of each symbol's belongingness to the category.
- Create a <em>token set</em> from the corpus by replacing each symbol in the corpus with
a name, or <em>token</em>, for its corresponding category. (in the proof of concept we will simply pick the highest ranked
category from each symbol's entry in the sym-cat mapping. Future work will be needed to make an
informed choice of category in the case of context dependent taxonomy.)
- Create a <em>garbage set</em> from the token set by randomizing a large enough proportion of the tokens in the token set to render each string likely invalid.
- Train a comprehensible, spiking, convolutional neural network, or <em>grammar net</em>, to distinguish between items of the token and garbage sets.
- The filters of the trained grammar net are first order grammar rules, the fully connected layers behind those are second order grammar rules, and so on. (or something to that effect depending on the structure of the grammar net.)

## How to Install
<!-- - Clone this repository. -->
[TBD]

## How to use
[TBD]

## Contributing
For contributors to the project; do this before making your first commit:

- Install pre-commit
```bash
cd /path/to/this/repository/
sudo apt install pre-commit
pre-commit install
```
(we do all of our development on linux)

- To test updates to the readme and other GitHub flavored markdown, simply install Grip
and feed it your desired file.
```bash
pip3 install grip
grip README.md
```
- Then follow the link provided by the Grip sever for a live preview of your work.

- When satisfied with your changes you can compile to an html file with:
```bash
grip README.md --export README.html
```

## Authors
* **Gabe M. LaFond** - *Initial work* - [ExamDay](https://github.com/ExamDay)

See also the list of [contributors](https://github.com/ExamDay/NeuralGREWT/contributors) who participated in this project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
