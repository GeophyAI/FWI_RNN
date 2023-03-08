import torch, functorch
device ="cuda"
model = torch.nn.Linear(2,2).to(device)
inputs = torch.rand(1,2).to(device)
criterion = torch.nn.CrossEntropyLoss()
target=torch.ones(len(inputs), dtype=torch.long).to(device)

func, func_params = functorch.make_functional(model) 

def loss(params):
    out = func(params, inputs) 
    return criterion(out, target)  
print(loss)
print(func_params)
H=functorch.hessian(loss)(func_params)  
#print(H)
