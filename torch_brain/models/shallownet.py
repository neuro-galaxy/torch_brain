"""
Minimal re-implementation of ShallowFBCSPNet architecture.

Inspired by:
- Schirrmeister et al. 2017 
- Braindecode (BSD-3 licensed) implementation:
  https://github.com/braindecode/braindecode

"""

import torch
from torch import nn
from typing import Union

class ShallowNet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        filter_time_length: int = 25,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: Union[int, str] = "auto",
        dropout_p: float = 0.5,
        logsoftmax: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.in_times = in_times
        self.conv_time = nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=True)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (1, in_chans), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat, momentum=0.1)
        self.pool = nn.AvgPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1))
        self.dropout = nn.Dropout(p=dropout_p)

        # analytic final conv length 
        if final_conv_length == "auto":
            L1 = (in_times - filter_time_length) // 1 + 1
            L3 = (L1 - pool_time_length) // pool_time_stride + 1
            self.final_conv_length = L3
        else:
            self.final_conv_length = int(final_conv_length)

        self.conv_classifier = nn.Conv2d(n_filters_spat, n_classes,
                                         kernel_size=(self.final_conv_length, 1), bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=1) if logsoftmax else nn.Identity()

        # init (paper suggests Xavier uniform)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv_spat.weight, gain=1.0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        nn.init.xavier_uniform_(self.conv_classifier.weight, gain=1.0)
        nn.init.constant_(self.conv_classifier.bias, 0.0)
        if self.conv_time.bias is not None:
            nn.init.constant_(self.conv_time.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B, C, T) or (B, C, T, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = x.permute(0, 3, 2, 1)          
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x * x                          
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # safe_log
        x = self.dropout(x)
        x = self.conv_classifier(x) 
        x = self.logsoftmax(x)
        x = x.squeeze(3).squeeze(2)   #(B, n_classes)
        return x
