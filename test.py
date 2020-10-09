import torch
import torchvision
import time
import numpy as np

model = torchvision.models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 400, 400)
tt = np.zeros(10)
for i in range(10):
	t = time.time()
	out = model(example)
	xt = time.time() - t
	tt[i] = time.time()-t
	print(xt)
print(tt.mean())
exit()
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")