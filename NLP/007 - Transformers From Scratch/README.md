# Transformers from Scratch

So you have probably heard about the Transformers architecture and Attention Mechanism.
Maybe you have a high level intuition about it, you know the system of keys, queries, values, and you vaguely know how attention works.

But can you confidently answer this question:
- What is causal masking?
- How do self-attention and cross attention differ?
- What are residual connections?
- What is the link between Transformers, Chat-GPT, and BERT?
- What is Multi-Headed attention?
- What is Positional Encoding?
- What does it mean to stack Encoders and Decoders and how do they interact which each other?

If you can answer all of these questions, you are already a King/Queen of Transformers, I have nothing to teach you, congrats! If not, stay with me, you will become an expert.

Prerequisites:

Before diving in this tutorial you should definetly check "The Illustrated Transformer" by Jay Alammar, and at least try to read the "Attention is All you Need" paper, but you should just have an overview, this tutorial is here to make you understand the article fully.

## Thanking

Many thanks to Andrej Karpathy the absolute boss in this field who has a noticeable sharp mind about the topic. Thank you to the authors of "Attention is All You Need", thank you to Jay Alammar for "The Illustrated Transformer", and to Harvard for its "Annotated Transformer".

## Links of the ressources

- [Video Implementation](https://www.youtube.com/watch?v=U0s0f995w14)
- [Attention is All you Need](https://arxiv.org/abs/1706.03762)
