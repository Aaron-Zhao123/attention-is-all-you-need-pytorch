import torch.nn as nn
import torch
from overriders.base import CustomWrapperBase
from overriders.pruner import Pruner
from overriders.network_wrapper import NetworkWrapperBase, override


class ToyNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(ToyNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)
    # comment out this line to test :p
    # self.linear1.weight = Pruner(data=self.linear1.weight.data, requires_grad=True)
    # tmp = self.linear2.weight
    # self.linear2.weight = Pruner(data=self.linear2.weight.data, requires_grad=True)
    # networkwrap(self, transformer=Pruner)
    # import pdb; pdb.set_trace()

  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


class NetworkWrapper(ToyNet, NetworkWrapperBase):
    def __init__(self, *args, **kwargs):
        self.transformer = kwargs.pop('transformer')
        super(NetworkWrapper, self).__init__(*args, **kwargs)
        self._override(update=True)

    def forward(self, x):
        self._override(update=False)
        return super(NetworkWrapper, self).forward(x)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 101, 102, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

pruner = Pruner()
# Construct our model by instantiating the class defined above.
model = NetworkWrapper(D_in, H, D_out, transformer=pruner)
# model = ToyNet(D_in, H, D_out)
# model_wrapper._override(model, update=False)
# override(model, pruner, update=True)

# for name, param in model.named_parameters():
#         # if self._check_name(name):
#         mask = pruner.get_mask(param.data, name)
#         # mask = torch.zeros(param.data.shape)
#         param.data = param.data.mul_(mask)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
data = model.linear1.weight
print(torch.sum(data==0))

for t in range(100):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)

  # Compute and print loss
  loss = loss_fn(y_pred, y)

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

y_pred = model(x)
plist = model.parameters()
for module in model.modules():
  print(module)
data = model.linear1.weight.data
print(torch.sum(data==0))
import pdb; pdb.set_trace()

