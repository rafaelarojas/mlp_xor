import torch
import torch.nn as nn
import torch.optim as optim

class XORMLP(nn.Module):
    def __init__(self):
        super(XORMLP, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

inputs = torch.tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])

targets = torch.tensor([[0.0],
                        [1.0],
                        [1.0],
                        [0.0]])

model = XORMLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 10000
for epoch in range(epochs):
    for input_data, target in zip(inputs, targets):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for input_data in inputs:
        output = model(input_data)
        print(f"Input: {input_data.numpy()}")
        print(f"Output: {output.numpy()}")
        print(f"---------------------")