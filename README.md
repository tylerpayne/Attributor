# Attributor

Attributor is a library for evaluating and exploring attention attribution in Transformer networks.

## Motivation

Assume you have a question you want answered. You want the answer to be based in fact, so you collect a set of documents from sources you trust and think are relevant to the question topic. You don't have time to read them yourself, though, and enlist a friend. You give your friend the documents and your question. You ask them to read the documents and then write down the answer to your question in at most one sentence and mail the sentence to you. Eventually you get the time to read your friend's letter. With a healthy level of skepticism, you wonder which documents your friend got their answer from. You decide to ask them but, unfortunately, your friend is a Transformer. 

Attribution is the problem of assigning a cause to an effect. In Transformers, attention attribution attempts to assign _which_ input (prompt) tokens caused each output (generated) token.It does this by mechanistically examining the flow of information through the Transformer from input tokens to output tokens. Interestingly, this examination is _content agnostic_: we are only concerned with _where_ information came from not _what_ that information represents. 

Let's get specific. There are only two operations in a (sufficiently vanilla, decoder-only) Transformer that mix information across token positions (a.k.a. residual streams): the attention blocks and residual connections. 

Consider a sequence of $n$ tokens $X=(x_1,\dots,x_n)$ where the first $k$ tokens are the input (prompt) tokens and the $o=n-k$ remaining tokens are the (already generated) output tokens. Assume you have the  $l$ attention matrices $A=(A_1\dots,A_l)$ computed while generating the $o$ output tokens. 

We are only interested in how the positional information of each input token flows to the output, so $X_0$ is the $n\times n$ identity matrix. For each layer $i\in(1\dots,l)$ we compute $X_{i}=A_iX_i + 2*X_{i-1}$. Again, each layer simply mixes the previous layer's features according to the attention weights and adds two residual connections (generally: one before MLP, one after). Therefore, at the output, we have $(X_l)_{ij}$ is the strength of input token $j$'s informational influence on output token $i$. Or, in other words, the _attention attribution_ of token $i+1$ to token $j$ (+1 because each residual stream produces an output that informs the generation of the _next_ token).

Above is a mechanistic interpretation of how information flows from input token positions to output token positions in Transformers. Obviously we are glossing over the content-based transformations that the Transformer makes. Is information flow sufficient to attribute output tokens to input tokens in a _human interpretable_ way? To do that, we need to evaluate this technique over datasets.

TODO: Talk about implementation over HotPotQA

## Installation

Clone this repo. And navigate into it. Then, in your preferred virtual environment:

```sh
pip install -r requirements.txt
```

## Usage

Given a document question-answering dataset, Attributor can be used to perform data-driven evaluation of attention attribution in _any_ sufficiently vanilla (GPT, Llama, etc) decoder-only Huggingface Transformer.

See [hotpot_qa](hotpot_qa.py) for an implementation over the HotPotQA dataset.

```sh
python hotpot_qa.py --model your/favorite-hf-model 
```

## Post-hoc Exploration

After running an evaluation script (e.g. hotpot_qa.py), you can explore the data via an interactive web UI with the following command:

```sh
python explorer.py
```

Then open a browser and navigate to localhost:3000. Open a file and hit `Load` to explore!

## Interactive Exploration

TODO: per-token interactive visualization of attribution scores