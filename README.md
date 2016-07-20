# PixelCNN & PixelRNN in TensorFlow

TensorFlow implementation of [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759). This implementation contains:

![model](./assets/model.png)

1. PixelCNN
  - Masked Convolution (A, B)
2. PixelRNN
  - Row LSTM
  - Diagonal BiLSTM (skew, unskew)
  - Residual Connections
  - Multi-Scale PixelRNN


## Requirements

- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) 0.9+


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a generative model with mnist data:

    $ python main.py --data=mnist --is_train=True


## Results

(in progress)


## References

- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
- [Review by Kyle Kastner](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md)
- [kundan2510/pixelCNN](https://github.com/kundan2510/pixelCNN)
- [zhirongw/pixel-rnn](https://github.com/zhirongw/pixel-rnn)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
