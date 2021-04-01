<!-- - Compile a large <em>corpus</em> of valid and sensical writings in the chosen language. -->
<!-- - Train a natural language model, or <em>predictor</em>, on the corpus until it is very good at predicting -->
<!-- symbols in the language given context. Anything like BERT or GPT-2 will do. -->
<!-- - Compile a large "ground set" of valid and/or meaningful strings in the chosen language. -->
<!-- (in the case of unknown languages, this ground set can just be a random subset of the corpus) -->
<!-- - Compile a "symbol set" of all symbols in the chosen language, or at least a large -->
<!-- set of the most common ones. -->
- Create a 3 dimensional <em>perturbation matrix</em> from the ground and symbol sets by:
    - For each <em>ground string</em> in the ground set; for each <em>symbol</em> in the symbol set;
    add to the perturbation matrix the vector of all strings that can be created by replacing any
    one item in the ground string with the symbol. Of course, making sure to do this in a
    consistent order for each vector. (in an actual matrix all such vectors need to be
    padded to uniform dimension with null strings ―  In practice we can use a 2 dimensional
    matrix of pointers to vectors of variable dimension)
- Create a 3 dimensional <em>validity matrix</em> from the perturbation matrix by:
    - For each item, or <em>p-string</em>, in the perturbation matrix, judge the probability
    of that string by summing the relative likelihoods of each symbol appearing at
    its location given all previous symbols in the p-string (using the predictor) and dividing by the
    length of that p-string.
- Perform principal component analysis on the validity matrix to infer number and relative
importance of symbol categories.
- Name each symbol category.
- Create a <em>sym-cat</em> mapping of each symbol to a list of its categories
sorted in descending order of the symbol's <em>belongingness</em> to each category.
- Create a <em>cat-sym</em> mapping of each category to a list of its symbols sorted in descending order of each symbol's belongingness to the category.
- Create a <em>token set</em> from the corpus by replacing each symbol in the corpus with
a name or <em>token</em> for its corresponding category. (in the proof of concept we will simply pick the highest ranked
category from each symbol's entry in the sym-cat mapping. Future work will be needed to make an
informed choice of category in the case of context dependent taxonomy.)
- Create a <em>garbage set</em> from the token set by randomizing a large enough proportion of the tokens in the token set to render each string likely invalid.
- Train a comprehensible, spiking, convolutional neural network, or <em>grammar net</em>, to distinguish between items of the token and garbage sets.
- The filters of the trained grammar net are first order grammar rules, the fully connected layers behind those are second order grammar rules, and so on. (or something to that effect depending on the structure of the grammar net.)
