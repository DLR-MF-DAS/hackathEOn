import torch, torch.nn as nn
from lightning_uq_box.uq_methods import MCDropoutClassification
from lightning_uq_box.uq_methods.mc_dropout import MCDropoutBase


def activate_dropout(self) -> None:
    """Proposed fix: everything in eval mode, then only the listed dropout layers in train mode."""
    self.model.eval()
    found = []
    for name, module in self.model.named_modules():
        if name in self.dropout_layer_names and isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
            found.append(name)
    if not found:
        raise UserWarning("No dropout layers found in model, maybe dropout is implemented via specialized layers?")


MCDropoutBase.activate_dropout = activate_dropout
exec(open(__file__.replace("fix_check.py", "repro_mc_dropout_batchnorm.py")).read().split('"""', 2)[2])
