import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_shape, n_class, dropout_rate=0.5, output_activation='softmax', **kwargs):
        super(MLP, self).__init__(**kwargs)

        input_size = np.prod(input_shape[1:])

        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        if output_activation == 'softmax':
            self.cl = nn.Linear(256, n_class)
            self.out_act = nn.Identity(dim=1) #Softmax() is already performed inside CrossEntropyLoss
        else:
            self.cl = nn.Linear(256, 1)
            self.out_act = nn.Sigmoid()


    def forward(self, inputs):
        output = self.flatten(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.cl(output)

        return self.out_act(output)


class TempCNN(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.5, output_activation='softmax', **kwargs):
        super(TempCNN, self).__init__(**kwargs)
        self.conv1 = nn.LazyConv1d(64,5,padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.LazyConv1d(64,5,padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.LazyConv1d(64,5,padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(dropout_rate)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.flatten = nn.Flatten()
        self.dense = nn.LazyLinear(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dp4 = nn.Dropout(dropout_rate)

        if output_activation == "softmax":
            self.cl = nn.LazyLinear(n_class)
            self.out_act = nn.Identity(dim=1) #Softmax() is already performed inside CrossEntropyLoss
        else:
            self.cl = nn.LazyLinear(1)
            self.out_act = nn.Sigmoid()

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        output1 = self.bn1(output1)
        output1 = self.relu1(output1)
        output1 = self.dp1(output1)

        output2 = self.conv2(output1)
        output2 = self.bn2(output2)
        output2 = self.relu2(output2)
        output2 = self.dp2(output2)

        output3 = self.conv3(output2)
        output3 = self.bn3(output3)
        output3 = self.relu3(output3)
        output3 = self.dp3(output3)

        output = self.gap(output3)

        # Classifier
        output = self.flatten(output)
        output = self.dense(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.dp4(output)
        output = self.cl(output)
        return self.out_act(output)


class InceptionLayer(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_bottleneck=True,
                 bottleneck_size=32, kernel_size=40):
        super(InceptionLayer, self).__init__()

        # self.in_channels = in_channels
        kernel_size_s = [(kernel_size) // (2 ** i) for i in range(3)] # = [40, 20, 10]
        kernel_size_s = [x+1 for x in kernel_size_s] # Avoids warning about even kernel_size with padding="same"
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck


        # Bottleneck layer
        self.bottleneck = nn.LazyConv1d(self.bottleneck_size, kernel_size=1,
                                    stride=1, padding="same", bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck_conv = nn.LazyConv1d(nb_filters, kernel_size=1,
                                         stride=1, padding="same", bias=False)

        # Convolutional layer (several filter lenghts)
        self.conv_list = nn.ModuleList([])
        for i in range(len(kernel_size_s)):
            # Input size could be self.in_channels or self.bottleneck_size (if bottleneck was applied)
            self.conv_list.append(nn.LazyConv1d(nb_filters, kernel_size=kernel_size_s[i],
                                            stride=1, padding='same', bias=False))

        self.bn = nn.BatchNorm1d(4*self.bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        in_channels = input.shape[-2]
        if self.use_bottleneck and int(in_channels) > self.bottleneck_size:
            input_inception = self.bottleneck(input)
        else:
            input_inception = input

        max_pool = self.max_pool(input)
        output = self.bottleneck_conv(max_pool)
        for conv in self.conv_list:
            output = torch.cat((output,conv(input_inception)),dim=1)

        output = self.bn(output)
        output = self.relu(output)

        return output


class Inception(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=40):
        super(Inception, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.fc(gap_layer)


class LSTMFCN(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/sktime and https://github.com/houshd/MLSTM-FCN
    def __init__(self, nb_classes, dim, dropout=0.8, kernel_sizes=(8,5,3),
                 filter_sizes=(128, 256, 128), lstm_size=8, attention=False):
        super(LSTMFCN, self).__init__()

        # self.attention = attention

        self.LSTM = nn.LSTM(dim, lstm_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        conv_layers = []
        for i in range(len(filter_sizes)):
            conv_layers.append(nn.LazyConv1d(filter_sizes[i], kernel_sizes[i], padding="same")) # keras: kernel_initializer="he_uniform"
            conv_layers.append(nn.BatchNorm1d(filter_sizes[i]))
            conv_layers.append(nn.ReLU())
            if i < len(filter_sizes):
                conv_layers.append(SqueezeExciteBlock(filter_sizes[i]))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

    def forward(self, input):
        # Dimension shuffle: input.permute(0,2,1)
        # Unecessary, since LSTM already takes (batch, seq, feature) reversed wrt to our input (batch, var, time), and also wrt the conv1d convention.
        # We want to give all timesteps to LSTM at each step (as proposed in the paper).
        whole_seq_output, _ = self.LSTM(input)
        x = whole_seq_output[:,-1,:] # Take only last time step of size (batch, lstm_size), as pytorch returns the whole sequence
        x = self.dropout(x)

        y = self.conv_layers(input)
        y = self.gap(y)

        output = torch.cat((x,torch.squeeze(y)),dim=1)
        return self.fc(output)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, input_channels):
        super(SqueezeExciteBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // 16)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_channels // 16, input_channels)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.gap(x)
        x_se = x_se.view(x_se.size(0), -1)
        x_se = F.relu(self.fc1(x_se))
        x_se = torch.sigmoid(self.fc2(x_se))
        x_se = x_se.view(x_se.size(0), -1, 1)

        x = x * x_se
        return x

"""
Lightweight Temporal Attention Encoder module

source: github.com/VSainteuf/lightweight-temporal-attention-pytorch/
MODIFICATIONS: 
- input shape is now : Batch size x Seq. length x Emb. dim. (reversed two last dims.)
- in_conv: replaced LayerNorm by BatchNorm, as the former required fixed sequence lenght
- LTAE: forward() receives src_pos as input (sequence position indexes in positional encoding)
- LTAE_clf forward() takes 'dates' and generates 'src_pos' as the day count from January 1st
- LTAE_clf calls LTAE with d_model=32, n_heads=4, n_neuron=[32,16], 
    len_max_seq=input_shape[-1], in_channels=input_shape[-2]

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import copy

class LTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att


        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            #TODO Replace this by 2-layer MLP?
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        # nn.LayerNorm((d_model, len_max_seq)) # Doesn't work with variable sequence lenght!
                                        nn.BatchNorm1d(d_model)
                                        )
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_pos):

        x = x.permute(0, 2, 1) # MODIFIED! To comply with PyTorch standard input shape N x C x T

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # if self.positions is None:
        #     src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        # else:
        #     src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)

        enc_output = x + self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


def get_sinusoid_encoding_table_var(positions, d_hid, clip=4, offset=3, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    x = np.array(positions)

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx + offset // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table = np.sin(sinusoid_table)  # dim 2i
    sinusoid_table[:, clip:] = torch.zeros(sinusoid_table[:, clip:].shape)

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


class LTAE_clf(nn.Module):
    """
    Lightweight Temporal Attention Encoder + MLP decoder (classifier)
    """

    def __init__(self, input_shape, n_classes, n_head=16, d_k=8, d_model=256, mlp_enc=[256, 128], 
                 dropout=0.2, T=1000, dates=None,
                 mlp_dec=[128, 64, 32], return_att=False):
    # def __init__(self, input_shape, n_classes, n_head=4, d_k=8, d_model=32, mlp_enc=[32, 16], 
    #              dropout=0.2, T=1000, dates=None,
    #              mlp_dec=[16, 16, 8], return_att=False):
        super(LTAE_clf, self).__init__()

        self.dates = get_day_count(dates)

        # if dates is not None:
        #     positions = get_day_count(dates) # for this to work, dates must contain all possible dates encountered in test time as well
        positions = 306 # nb. days between Oct 1st and August 1st. This way, all positions between 0 and 305 are computed
        len_max_seq = input_shape[-1] # unused if positions is provided. Better to use positions as len_max_seq also affects in_conv layer

        self.temporal_encoder = LTAE(in_channels=input_shape[-2], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp_enc, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq, positions=positions, return_att=return_att
                                           )

        mlp_dec.append(n_classes)
        self.decoder = get_decoder(mlp_dec)
        self.return_att = return_att

    def forward(self, input, dates=None):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
            
            dates: to be provided whenever different from the training dates
        """
        if dates is None:
            dates = self.dates
        else:
            dates = get_day_count(dates)

        if self.return_att:
            out, att = self.temporal_encoder(input, dates)
            out = self.decoder(out)
            return out, att
        else:
            out = self.temporal_encoder(input, dates)
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

        return total

def get_day_count(dates,ref_day='10-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int)
    return torch.tensor(days_elapsed,dtype=torch.long)

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_decoder(n_neurons):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu
    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
    """
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        if i < (len(n_neurons) - 2):
            layers.extend([
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m    