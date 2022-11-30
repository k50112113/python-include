import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

class torch_AE_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        return self.data[index], self.data[index]

class torch_AE(nn.Module):
    def __init__(self, input_size, encoder_size, latent_size, decoder_size=None, mirror=False):
        super(torch_AE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        
        self.encoder_size = [self.input_size] + encoder_size + [self.latent_size]
        if mirror == False:
            self.encoder_size = [self.input_size] + encoder_size + [self.latent_size]
        else:
            self.decoder_size = [self.latent_size] + encoder_size[::-1] + [self.input_size]
            
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
                
        self.activ = nn.Tanh()
        #self.activ = nn.ReLU()
        for layer_index in range(len(self.encoder_size)-1):
            self.encoder.append(nn.Linear(self.encoder_size[layer_index],self.encoder_size[layer_index+1]))
            
        for layer_index in range(len(self.decoder_size)-1):
            self.decoder.append(nn.Linear(self.decoder_size[layer_index],self.decoder_size[layer_index+1]))
            
    def forward(self, x):
        output = self.encode(x)
        output = self.decode(output)
        return output
        
    def encode(self, x):
        output = x
        for layer_index in range(len(self.encoder_size)-1):
            output = self.encoder[layer_index](output)
            output = self.activ(output)
        return output
        
    def decode(self, x):
        output = x
        for layer_index in range(len(self.encoder_size)-2):
            output = self.decoder[layer_index](output)
            output = self.activ(output)
        output = self.decoder[len(self.encoder_size)-2](output)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_split_seed = 42
num_epochs      = 50
batch_size      = 10000
test_ratio      = 0.01
learning_rate   = 0.001
latent_size     = 2
encoder_size    = [64,256,64]
mirror          = True

adir = "cycloalkane/c6-kabsch"
raw_data_filename = adir+"/dscrp-cyclohexane-bb-600k-kabsch"
model_sufix = "-600k-kabsch"
useweights = False
"""
bb_dscrp = [0,1,2,24,25,26]
h_dscrp = []
for i in range(dim):
    if i not in bb_dscrp:
        h_dscrp.append(i)
weigths = [1 if i in h_dscrp else 0 for i in range(dim)]
weigths = torch.tensor(weigths)
"""
raw_data = []
with open(raw_data_filename,"r") as fin:
    for aline in fin:
        if "Frame" not in aline:
            linelist = aline.strip().split()
            raw_data.append([float(i) for i in linelist[1:]])
            #raw_data.append([float(linelist[i]) for i in range(1,dim+1)])
dim = len(raw_data[0])
print("Input dimension: %d"%(dim))
raw_dataset = torch_AE_Dataset(torch.from_numpy(np.array(raw_data,dtype=np.float32)))
total_n_sample       = len(raw_dataset)
input_size           = len(raw_dataset[0][0])
total_n_test_sample  = int(test_ratio*total_n_sample)
total_n_train_sample = total_n_sample - total_n_test_sample
print('Number of training samples: %d'%(total_n_train_sample))
print('Number of testing samples:  %d'%( total_n_test_sample))
train_dataset, test_dataset = torch.utils.data.random_split(raw_dataset, [total_n_train_sample, total_n_test_sample], generator=torch.Generator().manual_seed(random_split_seed))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)
'''
examples= iter(train_loader)
X, y = examples.next()
print(X.shape,y.shape)
print(X,y)
'''
model = torch_AE(input_size, encoder_size, latent_size, mirror=mirror)
#print("Let's use", torch.cuda.device_count(), "GPUs!")
#model = nn.DataParallel(model)
#odel = MyDataParallel(model)
model.to(device)
if useweights == True:
    lossfunc = nn.MSELoss(reduction='none')
    weigths = weigths.to(device)
else:
    lossfunc = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9 ,dampening = 0.01)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        #weigths = weights.to(device)        
        output = model(X)
        loss = lossfunc(output, y)
        if useweights == True:
            loss = (loss * weigths / weigths.sum()).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        #if i % 10 == 0:
        #    print(i*batch_size)
    #print("")
    print('epoch %d / %d\tloss = %.4e'%(epoch+1,num_epochs,loss.item()))

avg_loss = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        
        output = model(X)
        loss = lossfunc(output, y)
        if useweights == True:
            loss = (loss * weigths / weigths.sum()).mean()

        n_samples += y.shape[0]
        avg_loss.append(loss.item())
    print('number of test samples = %d\naverage loss = %.4e'%(n_samples,np.mean(np.array(avg_loss))))

modelname = adir+'/ae'
for i in [input_size]+encoder_size+[latent_size]:
    modelname += '-%d'%i
modelname += model_sufix+'.pt'
torch.save(model.state_dict(), modelname)

