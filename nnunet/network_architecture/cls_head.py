import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, pooling="avg", dropout=0.2):
        super(ClassificationHead, self).__init__()

        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        self.pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        self.flatten = Flatten()
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        self.linear  = nn.Linear(in_channels, out_channels, bias=True)
        #super().__init__(self.pool, self.flatten, self.dropout, self.linear)
    
    def forward(self, input):
        #input = [batch, last_conv_output_channels, last_conv_output_width, last_conv_output_height] ex: [1, 320, 5, 5, 6]
        avg_pool = self.pool(input) # ex: avg_pool = [1, 320, 1, 1, 1]
        flat = self.flatten(avg_pool) # ex: flat = [1, 320] ← [1, 320*1*1*1]
        drop_out = self.dropout(flat) 
        linear = self.linear(drop_out) # ex: linear = [1, 1]

        return linear