"""Minimal reproduction: MCDropout*.predict_step() leaves BatchNorm in training mode.

Run:  python repro_mc_dropout_batchnorm.py
"""
import torch
import torch.nn as nn
from lightning_uq_box.uq_methods import MCDropoutClassification

torch.manual_seed(0)

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Dropout(0.2), nn.Linear(8, 4),
)
model = MCDropoutClassification(net, num_mc_samples=10, loss_fn=nn.CrossEntropyLoss(),
                                dropout_layer_names=["5"])
model.eval()

bn = net[1]
bn.running_mean.fill_(0.0)
bn.running_var.fill_(1.0)
mean_before = bn.running_mean.clone()

x = torch.randn(64, 3, 16, 16) * 5 + 3   # data with other statistics than the running stats
model.predict_step(x)

print("BatchNorm in training mode after predict_step:", bn.training)
print("running_mean changed by predict_step:          ", not torch.allclose(mean_before, bn.running_mean))
print("max |change| of running_mean:                  ", (bn.running_mean - mean_before).abs().max().item())

# second, smaller problem: all dropout layers are switched on, not only the listed ones
net2 = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 2))
m2 = MCDropoutClassification(net2, num_mc_samples=2, loss_fn=nn.CrossEntropyLoss(), dropout_layer_names=["3"])
m2.eval()
m2.activate_dropout()
print("dropout '1' active although only '3' was listed:", net2[1].training)
