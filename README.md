# PyTorch Dynamic RNN Decoder Tree

This is code I wrote within less than an hour so as to very roughly draft how I would code a Dynamic RNN Decoder Tree.

## The idea 

This decoder tree is meant to take as an input a neural embedding (such as a CNN's last feature map) to decode it into programming code as a decoder tree (for example, generating a HTML tree of code with this RNN decoder tree, for converting a screenshot of a website to the code generating that screenshot). 

For a full implementation of an RNN decoder tree (but without attention mechanisms such as I have also thought about), you may want to check out [that other implementation](https://github.com/XingxingZhang/td-treelstm).

I wrote the code of the current repository after applying to the [AI Grant](https://aigrant.org/) while waiting for a decision. Me and my teammate ended up in the top 10% of applicants with that project, but the number of grants awarded is more limited. 

## Attention Mechanisms

Four different Attention Mechanisms could be used at different key places: 
- In the convolutional feature map between the encoder CNN and the decoder RNN Tree.
- Across depth to capture context.
- Across breadth to keep track of what remains yet to decode. 
- Also, note that it may be possible to generate a (partial) render of the "yet-generated" HTML, so as to pass that to a second CNN encoder on which a fourth attention module could operate. This fourth attention module would be repeated at every depth, as the third module which is also across depth at every depth level. This way, during decoding, it would be possible to update the view for the decoder at every level throughout decoding, thanks to dynamical neural networks (E.G.: TensorFlow Eager mode, or PyTorch). 

## References and related work 
- [Top-down Tree Long Short-Term Memory Networks](https://github.com/XingxingZhang/td-treelstm) - Decoder Tree LSTMs, without attention mechanisms
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) - Quick overview of attention mechanisms in RNNs, along other interesting recent subjects
- [Attention Mechanisms in Recurrent Neural Networks (RNNs) - IGGG](https://www.youtube.com/watch?v=QuvRWevJMZ4) - A talk of mine where I explain attention mechanisms in RNNs and CNNs. 
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) - How to use attention mechanisms on convolutional feature maps
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) - Interesting discovery on how to stack convolutional layers, it has the best paper award at CVPR 2017 (this year) 
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326) - Based off the previous linked paper, here an encoder-decoder CNN architecture is built and the encoder is what interests me for plugging before the RNN Decoder Tree to generate HTML code
- [pix2code](https://github.com/tonybeltramelli/pix2code) - An existing implementation of what I want to do, without attention mechanisms nor any Dynamic RNN Decoder Tree
- [sketchnet](https://github.com/jtoy/sketchnet) - My teammate's work on the same project, before applying to the AI Grant
- [Bootstrap](http://getbootstrap.com/) - What I would use to style the generated HTML code, such that the RNN Decoder Tree outputs both type and styling information at each node of the HTML tree
