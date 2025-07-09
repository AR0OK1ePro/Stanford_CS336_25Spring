import torch
from cs336_basics import train_transformer

def learning_rate_test(lr: float):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = train_transformer.SGD([weights], lr=lr)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        learning_rate_test(lr)
