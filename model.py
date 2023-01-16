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