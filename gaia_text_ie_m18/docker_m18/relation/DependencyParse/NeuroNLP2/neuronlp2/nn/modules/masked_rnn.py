__author__ = 'max'

import torch.nn as nn
from .._functions.masked_rnn import AutogradMaskedRNN, AutogradMaskedStep


class MaskedRNNBase(nn.Module):
    def __init__(self, Cell, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, **kwargs):

        super(MaskedRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                cell = self.Cell(layer_input_size, hidden_size, self.bias, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions + direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def forward(self, input, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        lstm = self.Cell is nn.LSTMCell
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
            if lstm:
                hx = (hx, hx)

        func = AutogradMaskedRNN(num_layers=self.num_layers,
                                 batch_first=self.batch_first,
                                 dropout=self.dropout,
                                 train=self.training,
                                 bidirectional=self.bidirectional,
                                 lstm=lstm)

        output, hidden = func(input, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1, )))
        return output, hidden

    def step(self, input, hx=None, mask=None):
        '''
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, input_size): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            mask (batch): the mask tensor of this step.

        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        '''
        assert not self.bidirectional, "step only cannot be applied to bidirectional RNN."
        batch_size = input.size(0)
        lstm = self.Cell is nn.LSTMCell
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            if lstm:
                hx = (hx, hx)

        func = AutogradMaskedStep(num_layers=self.num_layers,
                                 dropout=self.dropout,
                                 train=self.training,
                                 lstm=lstm)

        output, hidden = func(input, self.all_cells, hx, mask)
        return output, hidden


class MaskedRNN(MaskedRNNBase):
    r"""Applies a multi-layer Elman RNN with costomized non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. If nonlinearity='relu', then `ReLU` is used instead
    of `tanh`.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, h_0
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(MaskedRNN, self).__init__(nn.RNNCell, *args, **kwargs)


class MaskedLSTM(MaskedRNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, (h_0, c_0)
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.


    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len
    """

    def __init__(self, *args, **kwargs):
        super(MaskedLSTM, self).__init__(nn.LSTMCell, *args, **kwargs)


class MaskedGRU(MaskedRNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, h_0
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__(nn.GRUCell, *args, **kwargs)
