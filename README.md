# Attributor (🚧 Under Construction)

Attributor is a library for evaluating and exploring attention attribution in Transformer networks.


## Installation

Clone this repo. And navigate into it. Then, in your preferred virtual environment:

```sh
pip install -r requirements.txt
```

## Motivation

You have a burning question you want answered. You also want the answer to be based in data, so you collect a set of documents from sources you trust and think are relevant to the question topic. You don't have time to read them yourself, though, and enlist a friend. 

You ask your friend to read the documents and then write down a succinct answer to your question and mail the answer to you. 

Eventually you get the time to read your friend's letter. With a healthy level of skepticism, you wonder which documents your friend based their answer on. 

You decide to ask them but, unfortunately, your friend is a Transformer. 

## Attention Attribution

Attribution is the problem of assigning a cause to an output. In Transformers, attention attribution attempts to assign _which_ input (prompt) tokens caused each output (generated) token. It does this by mechanistically examining the flow of information through the Transformer from input tokens to output tokens. For now this repo only focuses on _content agnostic_ attribution: we are only concerned with _where_ information came from not _what_ that information represents. 

Let's get specific. There are only two operations in a (sufficiently vanilla, decoder-only) Transformer that mix information across token positions (a.k.a. residual streams): the attention blocks and residual connections. 

Consider a sequence of $n$ tokens $X=(x_1,\dots,x_n)$ where the first $k$ tokens are the input (prompt) tokens and the $o=n-k$ remaining tokens are the (already generated) output tokens. Assume you have the  $l$ attention matrices $A=(A_1\dots,A_l)$ computed while generating the $o$ output tokens. 

In the current implementation we are only interested in how the information from each input position flows to each output position. Each layer simply mixes the previous layer's features according to the attention weights and residual connections (note: the post-MLP residual will get normalized out). Specifically, let $X_0$ be the $n\times n$ identity matrix (i.e. a one hot encoding of each position). Then, for each layer $i\in(1\dots,L)$ we compute $X_{i}=A_iX_{i-1} + X_{i-1}$.  

Finally, at the output layer $L$, we have $(X_L)_{ij}$ is the amount input token $j$ was attended to when generating output token $i$.

Above is a mechanistic interpretation of how information flows from input token positions to output token positions in Transformers. Obviously we are glossing over most of the transformations in a Transformer. __Is information flow sufficient to  attribute output tokens to input tokens in a _human interpretable_ way?__ To try and answer that question, we'll evaluate the above approach over document question-answering datasets.

## Usage

### Evaluation

Given a document question-answering dataset, Attributor can be used to perform data-driven evaluation of attention attribution in _any_ sufficiently vanilla (GPT, Llama, etc) decoder-only Huggingface Transformer.

See [hotpot_qa](hotpot_qa.py) for an implementation over the HotPotQA dataset.

```sh
python hotpot_qa.py --model your/favorite-hf-model 
```

For example

```
python hotpot_qa.py --model HuggingFaceTB/SmolLM-135M-Instruct --dtype float16 --device_map cuda 
```

### Post-hoc Exploration

After running an evaluation script (e.g. hotpot_qa.py), you can explore the data via an interactive web UI with the following command:

```sh
python explorer.py
```

Then open a browser and navigate to localhost:3000. Open a file and hit `Load` to explore!

### Interactive Visualization

Included is a interactive visualization of the attention flow through the network. Run the python sever

```sh
cd interactive
uvicorn --reload server.app:app
```

Then run the UI

```
cd webapp
npm start
```

Visit localhost:3000. Enter a Huggingface model ID (currently must be a Llama-derived model), enter a device_map (e.g. 'cuda'), a precision (e.g. 'bfloat16') and max tokens (set a high number, it is unused right now)

![interactive-viz](interactive.png)