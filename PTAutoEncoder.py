import torch
import torch.nn as nn
import numpy as np
import time
import os
import CurveFitting as CF

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

class AutoEncoder_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        return self.data[index], self.data[index]

class AutoEncoderModel(nn.Module):
    def __init__(self, input_size, encoder_size, latent_size, decoder_size=[]):
        super(AutoEncoderModel, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        
        self.encoder_size = [self.input_size] + encoder_size + [self.latent_size]
        if len(decoder_size) == 0: decoder_size = encoder_size[::-1]
        self.decoder_size = [self.latent_size] + decoder_size + [self.input_size]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
                
        self.activate = nn.Tanh() #self.activate_relu = nn.ReLU()
        for layer_index in range(len(self.encoder_size)-1):
            self.encoder.append(nn.Linear(self.encoder_size[layer_index],self.encoder_size[layer_index+1]))
            
        for layer_index in range(len(self.decoder_size)-1):
            self.decoder.append(nn.Linear(self.decoder_size[layer_index],self.decoder_size[layer_index+1]))
            
    def forward(self, x):
        output = self.decode(self.encode(x))
        return output
        
    def encode(self, x):
        output = x
        for layer_index in range(len(self.encoder_size)-1):
            output = self.encoder[layer_index](output)
            output = self.activate(output)
        return output
        
    def decode(self, x):
        output = x
        for layer_index in range(len(self.decoder_size)-2):
            output = self.decoder[layer_index](output)
            output = self.activate(output)
        output = self.decoder[len(self.decoder_size)-2](output)
        return output

class Autoencoder:
    def __init__(self, model_name="", input_size=0, encoder_size=[64,256,64], latent_size=2, decoder_size=[],\
                       random_split_seed=42,num_epochs=50,batch_size=10000,\
                       test_ratio=0.1,learning_rate_start=1e-3,learning_rate_end=-1,loss_weigths=[],\
                       regularization="",lambda_regularization=0,\
                       verbose=1,select_cuda_id=0):
        self.ae_model                       = None
        self.verbose                        = verbose
        self.verbose_output_file                    = None
        self.device                         = torch.device('cuda:%d'%(select_cuda_id) if torch.cuda.is_available() else 'cpu')
        self.random_split_seed              = random_split_seed
        self.num_epochs                     = num_epochs      
        self.batch_size                     = batch_size
        self.test_ratio                     = test_ratio     
        self.learning_rate_start            = learning_rate_start
        if learning_rate_end > -1: 
            self.learning_rate_end          = learning_rate_end
        else:
            self.learning_rate_end          = learning_rate_start
        self.learning_rate_lambda_table_    = np.append(np.logspace(np.log10(self.learning_rate_start),np.log10(self.learning_rate_end),num_epochs),[0])/self.learning_rate_start
        self.loss_weigths                   = torch.tensor(loss_weigths).to(self.device)
        self.regularization                 = regularization.lower() #l1/lasso, l2/ridge
        self.lambda_regularization          = lambda_regularization
        self.train_loader                   = None 
        self.test_loader                    = None

        if model_name != "":
            self.LoadModel(model_name)
        elif input_size > 0:
            self.input_size                 = input_size
            self.latent_size                = latent_size
            self.encoder_size               = encoder_size
            if len(decoder_size) > 0:
                self.decoder_size           = decoder_size
            else:
                self.decoder_size           = encoder_size[::-1]
            self.CreateAutoencoderModel()
        else:
            print("The input size cannot be zero.")
    
    def CreateAutoencoderModel(self,model_name=""):
        self.ae_model = AutoEncoderModel(self.input_size, self.encoder_size, self.latent_size, self.decoder_size)
        if model_name != "":
            self.ae_model.load_state_dict(torch.load(model_name+".pt",map_location=self.device))
        self.ae_model.to(self.device)

    def split_dataset(self,dataset,batch_size,test_ratio):
        total_n_test_sample        = int(test_ratio*len(dataset))
        total_n_train_sample       = len(dataset) - total_n_test_sample
        if self.verbose==1:
            print('Number of training samples = %s'%(total_n_train_sample))
            print('Number of testing samples  = %s'%(total_n_test_sample))
        if self.verbose_output_file != None:
            self.verbose_output_file.write('Number of training samples = %s\n'%(total_n_train_sample))
            self.verbose_output_file.write('Number of testing samples  = %s\n'%(total_n_test_sample))

        if total_n_train_sample > 0:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total_n_train_sample, total_n_test_sample], generator=torch.Generator().manual_seed(self.random_split_seed))
            self.train_loader           = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
            self.test_loader            = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)
        else:
            self.test_loader            = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)

    def apply_loss(self):
        self.loss_function = nn.MSELoss(reduction='none')

    def compute_loss(self, output, y, return_mse=False, regulized_term=None):
        loss = self.loss_function(output, y)
        if len(self.loss_weigths) > 0: loss = loss * self.loss_weigths / self.loss_weigths.sum()
        loss = torch.mean(loss)
        loss_mse = torch.clone(loss)
        if self.lambda_regularization > 0:
            if self.regularization == "l1" or self.regularization == "lasso":
                loss += self.lambda_regularization * torch.mean(torch.linalg.norm(regulized_term, ord=1, dim=1))
            elif self.regularization == "l2" or self.regularization == "ridge":
                loss += self.lambda_regularization * torch.mean(torch.linalg.norm(regulized_term, ord=2, dim=1))
            # if self.regularization == "l1" or self.regularization == "lasso":
            #     loss += self.lambda_regularization * torch.norm(self.ae_model.encoder[len(self.encoder_size)].weight, p=1)
            # elif self.regularization == "l2" or self.regularization == "ridge":
            #     loss += self.lambda_regularization * torch.norm(self.ae_model.encoder[len(self.encoder_size)].weight, p=2)
        if return_mse==True:
            return loss, loss_mse
        return loss

    def apply_optimize(self,method='adam'):
        if method == 'adam':
            self.optimization = torch.optim.Adam(self.ae_model.parameters(), lr=self.learning_rate_start)
        elif method == 'sgd':
            self.optimization = torch.optim.SGD(self.ae_model.parameters(), lr=self.learning_rate_start, momentum = 0.3 ,dampening = 0.01)
        self.learning_rate_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimization, lr_lambda=lambda epoch: self.learning_rate_lambda_table_[epoch])

    def train_init(self,raw_data_filename,omit_first_i_column=0,interval=1):
        if not self.ae_model:
            print("No model specified.")
            exit()
        if self.verbose==1:
            print("Training starts_______________________________________")
            print("Random split seed = %s"%(self.random_split_seed))
            print("Total number of epochs = %d"%(self.num_epochs))
            print("Batch size = %d"%(self.batch_size))
            print("Learning rate = %f -> %f"%(self.learning_rate_start, self.learning_rate_end))
        if self.verbose_output_file != None:
            self.verbose_output_file.write("Training starts_______________________________________\n")
            self.verbose_output_file.write("Random split seed = %s\n"%(self.random_split_seed))
            self.verbose_output_file.write("Total number of epochs = %d\n"%(self.num_epochs))
            self.verbose_output_file.write("Batch size = %d\n"%(self.batch_size))
            self.verbose_output_file.write("Learning rate = %f -> %f\n"%(self.learning_rate_start, self.learning_rate_end))
        self.print_model_configuration()
        if raw_data_filename != "":
            raw_dataset = self.read_raw_dataset(raw_data_filename,omit_first_i_column=omit_first_i_column,interval=interval)
            self.split_dataset(raw_dataset, self.batch_size, self.test_ratio)
        elif self.test_loader == None or self.train_loader == None:
            print("No data for training.")
            exit()
        self.apply_loss()
        self.apply_optimize()

    def train_epoch(self):
        train_loss = []
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)   
            output = self.ae_model(X)
            cv = None
            if self.lambda_regularization > 0: cv = self.ae_model.encode(X)
            loss = self.compute_loss(output,y,regulized_term=cv)
            train_loss.append(loss.item())
            self.optimization.zero_grad()
            loss.backward()
            self.optimization.step()
        return np.mean(np.array(train_loss))

    def train_model(self,raw_data_filename,output_model_name="tmp",omit_first_i_column=0,interval=1):
        self.reset_timer()
        self.verbose_output_file = open(output_model_name+".out","w")
        self.train_init(raw_data_filename,omit_first_i_column=omit_first_i_column,interval=interval)
        print("Time for initilization: %.2fs"%(self.timer()))
        if self.verbose==1:
            print("epoch trainloss testloss testlossmse lr time")
        self.verbose_output_file.write("epoch trainloss testloss testlossmse lr time\n")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.learning_rate_schedule.step()
            test_loss, test_loss_mse = self.test_model(return_mse=True)
            dt = self.timer()
            if self.verbose==1:
                print("%d %.4e %.4e %.4e %.4e %.2f"%(epoch+1,train_loss,test_loss,test_loss_mse,self.optimization.param_groups[0]["lr"],dt))
            self.verbose_output_file.write("%d %.4e %.4e %.4e %.4e %.2f\n"%(epoch+1,train_loss,test_loss,test_loss_mse,self.optimization.param_groups[0]["lr"],dt))
        self.verbose_output_file.write("end of trainning.\n")
        test_loss, test_loss_mse = self.test_model(return_mse=True)
        self.verbose_output_file.write("Total_test_loss MSE_test_loss\n")
        self.verbose_output_file.write("%.4e %.4e\n"%(test_loss,test_loss_mse))
        self.SaveModel(model_name=output_model_name)
        self.verbose_output_file.close()
        self.verbose_output_file = None

    def train_sparse_model_on_the_fly(self,raw_data_filename,output_model_name="tmp",omit_first_i_column=0,interval=1,test_loss_mse_target=1.0e-2):
        self.reset_timer()
        check_test_loss_mse_target_freq = 100
        compute_essential_cv_freq = 1
        std_cutoff = 1.0e-2
        alpha = 0.5
        same_latent_size_max = 3
        min_num_epochs = 100
        lambda_ = [self.lambda_regularization]
        loss_ = []
        lin = lambda x, a, b : a*x+b
        lin_func = CF.Curve_fitting(lin)
        self.verbose_output_file = open(output_model_name+".out","w")
        self.learning_rate_lambda_table_ = np.logspace(np.log10(self.learning_rate_start),np.log10(self.learning_rate_end),check_test_loss_mse_target_freq+1)/self.learning_rate_start
        self.learning_rate_lambda_table_ = np.array(list(self.learning_rate_lambda_table_)*(self.num_epochs//check_test_loss_mse_target_freq+1))
        self.train_init(raw_data_filename,omit_first_i_column=omit_first_i_column,interval=interval)
        print("Time for initilization: %.2fs"%(self.timer()))

        n_cv_essential_ = [self.latent_size]
        if self.verbose==1:
            print("epoch trainloss testloss testlossmse lr lambda cv time")
        self.verbose_output_file.write("epoch trainloss testloss testlossmse lr lambda cv time\n")
        epoch = 0
        same_latent_size_count = 0
        while epoch < self.num_epochs:
            epoch += 1
            train_loss = self.train_epoch()
            self.learning_rate_schedule.step()
            test_loss, test_loss_mse = self.test_model(return_mse=True)

            if epoch % compute_essential_cv_freq==0:
                cv_essential = self.compute_essential_cv_from_mse_increase(std_cutoff)
                n_cv_essential = int(cv_essential.size()[0])
            dt = self.timer()
            if self.verbose==1:
                print("%d %.4e %.4e %.4e %.4e %.4e %d %.2f"%(epoch,train_loss,test_loss,test_loss_mse,self.optimization.param_groups[0]["lr"],self.lambda_regularization,n_cv_essential,dt))
            self.verbose_output_file.write("%d %.4e %.4e %.4e %.4e %.4e %d %.2f\n"%(epoch,train_loss,test_loss,test_loss_mse,self.optimization.param_groups[0]["lr"],self.lambda_regularization,n_cv_essential,dt))
            if epoch % check_test_loss_mse_target_freq == 0:
                if len(lambda_) == 1:
                    self.lambda_regularization *= (test_loss_mse_target/test_loss_mse)**alpha
                    lambda_.append(self.lambda_regularization)
                    loss_.append(test_loss_mse)
                else:
                    loss_.append(test_loss_mse)
                    popt,_ = lin_func.fit(lambda_,loss_)
                    self.lambda_regularization = (test_loss_mse_target - popt[1])/popt[0]
                    lambda_.append(self.lambda_regularization)

                n_cv_essential_.append(n_cv_essential)
                if n_cv_essential_[-1] == n_cv_essential_[-2]:
                    same_latent_size_count += 1
                    if same_latent_size_count >= same_latent_size_max:
                        break
                else:
                    same_latent_size_count = 0
                self.CreateAutoencoderModel()
                self.apply_optimize()

        self.verbose_output_file.write("end of trainning.\n")
        test_loss, test_loss_mse = self.test_model(return_mse=True)
        self.verbose_output_file.write("Total_test_loss MSE_test_loss\n")
        self.verbose_output_file.write("%.4e %.4e\n"%(test_loss,test_loss_mse))
        self.SaveModel(model_name=output_model_name+'-otf')
        self.verbose_output_file.close()
        self.verbose_output_file = None

    def compute_essential_cv_from_mse_increase(self,cutoff):
        with torch.no_grad():
            X_stack = []
            y_stack = []
            cv_stack = []
            output_stack = []
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                cv = self.ae_model.encode(X)
                output = self.ae_model(X)
                if torch.is_tensor(X_stack):
                    X_stack = torch.vstack((X_stack,X))
                    y_stack = torch.vstack((y_stack,y))
                    cv_stack = torch.vstack((cv_stack,cv))
                    output_stack = torch.vstack((output_stack,output))
                else:
                    X_stack = torch.clone(X)
                    y_stack = torch.clone(y)
                    cv_stack = torch.clone(cv)
                    output_stack = torch.clone(output)
            X_std = torch.std(X_stack,dim=0,unbiased=True)
            cv_std = torch.std(cv_stack,dim=0,unbiased=True)
            cv_avg = torch.mean(cv_stack,dim=0)
            _, loss_mse = self.compute_loss(output_stack, y_stack, return_mse=True, regulized_term=cv_stack)
            increase_mse_ratio = torch.zeros_like(cv_std)
            for i in range(list(cv_stack.size())[1]):
                tmp = torch.clone(cv_stack[:,i])
                cv_stack[:,i] = cv_avg[i]
                new_output = self.ae_model.decode(cv_stack)
                _, new_loss_mse = self.compute_loss(new_output, y_stack, return_mse=True, regulized_term=cv_stack)
                cv_stack[:,i] = tmp
                increase_mse_ratio[i] = torch.maximum(torch.zeros(1).to(self.device),((new_loss_mse-loss_mse)/loss_mse))
            increase_mse_ratio_truncated = torch.where(increase_mse_ratio > cutoff, increase_mse_ratio, torch.zeros(1).to(self.device))
            cv_essential = torch.flatten(torch.nonzero(increase_mse_ratio_truncated))
            #print(cv_std,increase_mse_ratio)
            return cv_essential
 
    def test_model(self,data=[],return_mse=False,batch_size=-1):
        if len(data) > 0:
            dataset = self.data_to_dataset(data)
            if batch_size == -1: batch_size = len(dataset)
            self.split_dataset(dataset,batch_size,1)
            self.apply_loss()
        if self.test_loader:
            test_loss = []
            test_loss_mse = []
            with torch.no_grad():
                for X, y in self.test_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    output = self.ae_model(X)
                    cv = None
                    if self.lambda_regularization > 0: cv = self.ae_model.encode(X)
                    loss = self.compute_loss(output, y, return_mse=return_mse, regulized_term=cv)
                    if return_mse == True:
                        loss, loss_mse = loss
                        test_loss_mse.append(loss_mse.item())
                    test_loss.append(loss.item())
                if return_mse == True:
                    return np.mean(np.array(test_loss)),np.mean(np.array(test_loss_mse))
                return np.mean(np.array(test_loss))
        else:
            print("No test data loaded.")

    def encode_data(self,x,output_filename="",batch_size=-1):
        if not self.ae_model:
            print("No model specified.")
            exit()
        dataset = self.data_to_dataset(x)
        if batch_size == -1: batch_size = len(dataset)
        self.split_dataset(dataset,batch_size,1)
        with torch.no_grad():
            cv_stack = []
            for X, _ in self.test_loader:
                X = X.to(self.device)
                cv = self.ae_model.encode(X)
                if torch.is_tensor(cv_stack):   cv_stack = torch.vstack((cv_stack,cv))
                else:                           cv_stack = torch.clone(cv)
            cv_stack = cv_stack.detach().numpy()    
            if output_filename != "":
                fout = open(output_filename,"w")
                for a_cv_sample in cv_stack:
                    for a_cv_value in a_cv_sample:
                        fout.write('%.4e '%(a_cv_value))
                    fout.write('\n')
                fout.close()
        return cv_stack
        
    def decode_data(self,cv,output_filename="",batch_size=-1):
        if not self.ae_model:
            print("No model specified.")
            exit()
        dataset = self.data_to_dataset(cv,is_data_latent=True)
        if batch_size == -1: batch_size = len(dataset)
        self.split_dataset(dataset,batch_size,1)
        with torch.no_grad():
            Xp_stack = []
            for cv, _ in self.test_loader:
                cv = cv.to(self.device)
                Xp = self.ae_model.decode(cv)
                if torch.is_tensor(Xp_stack):   Xp_stack = torch.vstack((Xp_stack,Xp))
                else:                           Xp_stack = torch.clone(Xp)
            Xp_stack = Xp_stack.detach().numpy()
            if output_filename != "":
                fout = open(output_filename,"w")
                for a_Xp_sample in Xp_stack:
                    for a_Xp_value in a_Xp_sample:
                        fout.write('%.4e '%(a_Xp_value))
                    fout.write('\n')
                fout.close()
        return Xp_stack
    
    def compute_decoder_jacobian_determinant(self,cv):
        if torch.is_tensor(cv) == False:
            cv = torch.from_numpy(np.array(cv,dtype=np.float32))
        dataset = self.data_to_dataset(cv,is_data_latent=True,ignore_error=True)
        self.split_dataset(dataset,10,1)
        det_stack = []
        for cv, _ in self.test_loader:
            cv = cv.to(self.device)
            jacob = torch.autograd.functional.jacobian(self.ae_model.decode, cv)
            jacob = torch.sum(jacob,dim=2)
            determinant = torch.det(torch.matmul(torch.transpose(jacob,1,2),jacob))
            if torch.is_tensor(det_stack):  det_stack = torch.cat((det_stack,determinant))
            else:                           det_stack = torch.clone(determinant)  
        return det_stack.detach().numpy()

    def read_raw_dataset(self, raw_data_filename, omit_first_i_column=0, interval=1):
        raw_data = []
        with open(raw_data_filename,"r") as fin:
            itr=0
            for aline in fin:
                if "Frame" not in aline and "#" not in aline:
                    itr += 1
                    if itr % interval == 0:
                        linelist = aline.strip().split()
                        raw_data.append([float(i) for i in linelist[omit_first_i_column:]])
                        if self.input_size != len(linelist[omit_first_i_column:]):
                            print("Data input size from %s is not compatible with model input size"%(raw_data_filename))
                            exit()
        if len(raw_data) == 0:
            print("Empty data file")
            exit()            
        return self.data_to_dataset(raw_data)

    def data_to_dataset(self,data,is_data_latent=False,ignore_error=False):
        if ignore_error == False:
            if (is_data_latent == False and self.input_size != len(data[0])) or \
            (is_data_latent == True and self.latent_size != len(data[0])):
                print("Data size is mismatch.")
                exit()
        if self.verbose==1:
            print("Number of samples          = %s"%(len(data)))
        dataset = AutoEncoder_Dataset(torch.from_numpy(np.array(data,dtype=np.float32)))
        return dataset

    def LoadModel(self,model_name):
        if os.path.isfile(model_name+".aedat") and os.path.isfile(model_name+".pt"):
            self.learning_rate_end = -1
            with open(model_name+'.aedat',"r") as fin:
                for aline in fin:
                    if "#" not in aline:
                        linelist = aline.strip().split()
                        if linelist[0]   == "input_size":           self.input_size             = int(linelist[2])
                        elif linelist[0] == "latent_size":          self.latent_size            = int(linelist[2])
                        elif linelist[0] == "encoder_size":         self.encoder_size           = [int(i) for i in linelist[2:]]
                        elif linelist[0] == "decoder_size":         self.decoder_size           = [int(i) for i in linelist[2:]]
                        elif linelist[0] == "random_split_seed":    self.random_split_seed      = int(linelist[2])
                        elif linelist[0] == "num_epochs":           self.num_epochs             = int(linelist[2])
                        elif linelist[0] == "batch_size":           self.batch_size             = int(linelist[2])
                        elif linelist[0] == "test_ratio":           self.test_ratio             = float(linelist[2])
                        elif linelist[0] == "learning_rate" or linelist[0] == "learning_rate_start":    self.learning_rate_start = float(linelist[2])
                        elif linelist[0] == "learning_rate_end":    self.learning_rate_end      = float(linelist[2])
                        elif linelist[0] == "loss_weigths":         self.loss_weigths           = torch.tensor([int(i) for i in linelist[2:]]).to(self.device)
                        elif linelist[0] == "regularization":
                            if len(linelist) > 2:                   self.regularization = linelist[2].lower()
                            else:                                   self.regularization = ""
                        elif linelist[0] == "lambda_regularization":self.lambda_regularization = float(linelist[2])

            if self.learning_rate_end == -1: self.learning_rate_end = self.learning_rate_start
            self.learning_rate_lambda_table_ = np.append(np.logspace(np.log10(self.learning_rate_start),np.log10(self.learning_rate_end),self.num_epochs),[0])/self.learning_rate_start
            self.CreateAutoencoderModel(model_name=model_name)
            return True
        else:
            print("Model %s not found.\nBoth %s.aedat and %s.pt should exist."%(model_name,model_name,model_name))
            return False

    def SaveModel(self,model_name):
        torch.save(self.ae_model.state_dict(), model_name+".pt")
        with open(model_name+'.aedat',"w") as fout:
            fout.write("#--------------------------------------------------\n")
            fout.write("input_size              = %s\n"%(self.input_size))
            fout.write("latent_size             = %s\n"%(self.latent_size))
            fout.write("encoder_size            = ")
            for w in self.encoder_size: fout.write("%s "%(w))
            fout.write("\n")
            fout.write("decoder_size            = ")
            for w in self.decoder_size: fout.write("%s "%(w))
            fout.write("\n")
            fout.write("random_split_seed       = %s\n"%(self.random_split_seed))
            fout.write("num_epochs              = %s\n"%(self.num_epochs))   
            fout.write("batch_size              = %s\n"%(self.batch_size))
            fout.write("test_ratio              = %s\n"%(self.test_ratio))     
            fout.write("learning_rate_start     = %s\n"%(self.learning_rate_start)) 
            fout.write("learning_rate_end       = %s\n"%(self.learning_rate_end))
            fout.write("regularization          = %s\n"%(self.regularization))
            fout.write("lambda_regularization   = %s\n"%(self.lambda_regularization))
            if len(self.loss_weigths) > 0:
                fout.write("loss_weigths            = ")
                for w in self.loss_weigths.detach().numpy(): fout.write("%s "%(w))
                fout.write("\n")
        if self.verbose==1: print("Model saved to %s.pt and %s.aedat"%(model_name,model_name))
        if self.verbose_output_file != None: self.verbose_output_file.write("Model saved to %s.pt and %s.aedat\n"%(model_name,model_name))

    def LRP_encoder(self, x, i):
        if torch.is_tensor(x) == False:
            x = torch.from_numpy(np.array(x,dtype=np.float32))
        x.to(self.device)
        with torch.no_grad():
            eps = 1e-9
            eps = 10
            A = [x]
            for layer_index in range(len(self.ae_model.encoder_size)-1):
                A.append(self.ae_model.activate(self.ae_model.encoder[layer_index](A[-1])))
            R = [None]*len(A)
            # mask = torch.zeros_like(A[-1])
            # mask[:, i] = 1.0
            # R[-1] = A[-1]*mask
            R[-1] = torch.zeros_like(A[-1])
            R[-1].to(self.device)
            R[-1][:, i] = 1.0
            for layer_index in range(len(self.ae_model.encoder_size)-2,-1,-1):
                #z = A[layer_index+1]
                w = self.ae_model.encoder[layer_index].weight
                #w = torch.maximum(torch.zeros_like(w),w)*0.25+w
                #w = torch.minimum(torch.zeros_like(w),w)*0.25+w
                z = torch.mm(A[layer_index],torch.transpose(w,0,1))+self.ae_model.encoder[layer_index].bias
                s = R[layer_index+1]/(eps+z)
                c = torch.mm(s,w)
                R[layer_index] = A[layer_index]*c
        # R_return = R[0].detach().numpy()
        # R_return /= np.sum(R_return,axis=1)[:,None]
        # return R[0] / torch.sum(R[0],dim=1)[:, None]
        return R[0]
        
    def LRP_decoder(self, x, i):
        if torch.is_tensor(x) == False:
            x = torch.from_numpy(np.array(x,dtype=np.float32))
        x.to(self.device)
        with torch.no_grad():
            eps = 1e-9
            A = [x]
            for layer_index in range(len(self.ae_model.decoder_size)-2):
                A.append(self.ae_model.activate(self.ae_model.decoder[layer_index](A[-1])))
            A.append(self.ae_model.decoder[len(self.ae_model.decoder_size)-2](A[-1]))
            R = [None]*len(A)
            # mask = torch.zeros_like(A[-1])
            # mask[:, i] = 1.0
            # R[-1] = A[-1]*mask
            R[-1] = torch.zeros_like(A[-1])
            R[-1].to(self.device)
            R[-1][:, i] = 1.0
            for layer_index in range(len(self.ae_model.decoder_size)-2,-1,-1):
                w = self.ae_model.decoder[layer_index].weight
                #w = torch.maximum(torch.zeros_like(w),w)*0.25+w
                #w = torch.minimum(torch.zeros_like(w),w)*0.25+w
                z = torch.mm(A[layer_index],torch.transpose(w,0,1))+self.ae_model.decoder[layer_index].bias
                s = R[layer_index+1]/(eps+z)
                c = torch.mm(s,w)
                R[layer_index] = A[layer_index]*c
        # R_return /= np.sum(R_return,axis=1)[:,None]
        # return R[0] / torch.sum(R[0],dim=1)[:, None]
        return R[0]

    def print_model_configuration(self):
        if self.verbose == 1:
            print("Encoder configuration:")
            for i in [self.input_size]+self.encoder_size+[self.latent_size]: print("%s\t"%(i),end="")
            print("\n")
            print("Decoder configuration:")
            for i in [self.latent_size]+self.encoder_size+[self.input_size]: print("%s\t"%(i),end="")
            print("\n")
            if len(self.loss_weigths) > 0:
                print("Use weights: ")
                for w in self.loss_weigths.detach().numpy(): print("%s "%(w),end="")
                print("\n")
            if self.regularization != "":
                print("Use %s regularization with lambda=%s."%(self.regularization,self.lambda_regularization))
        if self.verbose_output_file != None:
            self.verbose_output_file.write("Encoder configuration:\n")
            for i in [self.input_size]+self.encoder_size+[self.latent_size]: self.verbose_output_file.write("%s\t"%(i))
            self.verbose_output_file.write("\n")
            self.verbose_output_file.write("Decoder configuration:\n")
            for i in [self.latent_size]+self.encoder_size+[self.input_size]: self.verbose_output_file.write("%s\t"%(i))
            self.verbose_output_file.write("\n")
            if len(self.loss_weigths) > 0:
                self.verbose_output_file.write("Use weights: ")
                for w in self.loss_weigths.detach().numpy(): self.verbose_output_file.write("%s "%(w))
                self.verbose_output_file.write("\n")
            if self.regularization != "":
                self.verbose_output_file.write("Use %s regularization with lambda=%s.\n"%(self.regularization,self.lambda_regularization))
    
    def timer(self):
        self.time2 = time.perf_counter()
        dt = self.time2-self.time1
        self.time1 = self.time2
        return dt

    def reset_timer(self):
        self.time1 = time.perf_counter()
        self.time2 = None

        '''
        examples= iter(train_loader)
        X, y = examples.next()
        print(X.shape,y.shape)
        print(X,y)
        '''

        """
        bb_dscrp = [0,1,2,24,25,26]
        h_dscrp = []
        for i in range(dim):
            if i not in bb_dscrp:
                h_dscrp.append(i)
        loss_weigths = [1 if i in h_dscrp else 0 for i in range(dim)]
        loss_weigths = torch.tensor(loss_weigths)
        """
    """
    def compute_essential_cv(self,R_std_cutoff):
        with torch.no_grad():
            Rweighted, X_std, cv_std = self.compute_weighted_relevance()
            R_std = torch.std(Rweighted,dim=0,unbiased=True)
            R_std_normalized = R_std/torch.max(R_std)
            R_std_normalized_truncated = torch.where(R_std_normalized > R_std_cutoff, R_std_normalized, torch.zeros(1))
            cv_essential = torch.flatten(torch.nonzero(R_std_normalized_truncated))
            return cv_essential

    def compute_weighted_relevance(self,return_unweighted=False):
        with torch.no_grad():
            X_stack = []
            cv_stack = []
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                cv = self.ae_model.encode(X)
                if torch.is_tensor(X_stack):
                    X_stack = torch.vstack((X_stack,X))
                    cv_stack = torch.vstack((cv_stack,cv))
                else:
                    X_stack = torch.clone(X)
                    cv_stack = torch.clone(cv)

            X_std = torch.std(X_stack,dim=0,unbiased=True)
            cv_std = torch.std(cv_stack,dim=0,unbiased=True)
            R = torch.zeros_like(cv_stack)
            if return_unweighted == True:
                Runweighted = torch.zeros_like(cv_stack)
            for i in range(list(X.size())[1]):
                Rtmp = self.LRP_decoder(cv_stack,i)
                if return_unweighted == False:
                    #R += Rtmp*X_std[i]**2/torch.sum(X_std**2)*cv_std**2/torch.sum(cv_std**2)
                    R += Rtmp*X_std[i]/torch.sum(X_std)*cv_std/torch.sum(cv_std)
                else:
                    R += Rtmp
            # R /= torch.sum(R,dim=1)[:,None]
            return R, X_std, cv_std
    
    def change_latent_size(self,cv_essential):
        # encoder_weight_list = []
        # for layer_index in range(len(self.ae_model.encoder_size)-2):
        #     encoder_weight_list.append(torch.nn.Parameter(torch.clone(self.ae_model.encoder[layer_index].weight)))
        # encoder_weight_list.append(torch.nn.Parameter(torch.clone(torch.index_select(self.ae_model.encoder[len(self.ae_model.encoder_size)-2].weight, 0, cv_essential))))
        
        # decoder_weight_list = []
        # decoder_weight_list.append(torch.nn.Parameter(torch.clone(torch.index_select(self.ae_model.decoder[0].weight, 1, cv_essential))))
        # for layer_index in range(1,len(self.ae_model.decoder_size)-1):
        #     decoder_weight_list.append(torch.nn.Parameter(torch.clone(self.ae_model.decoder[layer_index].weight)))
        
        self.latent_size = int(cv_essential.size()[0])
        self.CreateAutoencoderModel()
        
        # for layer_index in range(len(self.ae_model.encoder_size)-1):
        #     self.ae_model.encoder[layer_index].weight = encoder_weight_list[layer_index]
        # for layer_index in range(len(self.ae_model.decoder_size)-1):
        #     self.ae_model.decoder[layer_index].weight = decoder_weight_list[layer_index] 
    """     

class HierarchicalAutoencoderModel(nn.Module):
    def __init__(self, input_size, encoder_size, latent_size, decoder_size=[]):
        super(HierarchicalAutoencoderModel, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder_size = [self.input_size] + encoder_size + [self.latent_size]
        if len(decoder_size) == 0: decoder_size = encoder_size[::-1]
        self.decoder_sizes_ = [[i] + decoder_size + [self.input_size] for i in range(1,self.latent_size+1)]

        self.encoder = nn.ModuleList()
        self.decoders_ = nn.ModuleList([nn.ModuleList() for _ in range(self.latent_size)])
                
        self.activate = nn.Tanh()
        for layer_index in range(len(self.encoder_size)-1):
            self.encoder.append(nn.Linear(self.encoder_size[layer_index],self.encoder_size[layer_index+1]))
        
        for latent_index in range(self.latent_size):
            for layer_index in range(len(self.decoder_sizes_[latent_index])-1):
                self.decoders_[latent_index].append(nn.Linear(self.decoder_sizes_[latent_index][layer_index],self.decoder_sizes_[latent_index][layer_index+1]))

    def forward(self, x):
        outputs_ = self.decode(self.encode(x))
        return outputs_
        
    def encode(self, x):
        output = x
        for layer_index in range(len(self.encoder_size)-1):
            output = self.encoder[layer_index](output)
            output = self.activate(output)
        return output
        
    def decode(self, x, output_last=False):
        if output_last == False:
            outputs_ = []
            indices = torch.arange(x.size()[1]).to(x.device)
            for latent_index in range(x.size()[1]):
                output = torch.index_select(x, -1, indices[:latent_index+1])
                for layer_index in range(len(self.decoder_sizes_[latent_index])-2):
                    output = self.decoders_[latent_index][layer_index](output)
                    output = self.activate(output)
                output = self.decoders_[latent_index][len(self.decoder_sizes_[latent_index])-2](output)
                if latent_index == 0:
                    outputs_ = torch.unsqueeze(output,0)
                else:
                    outputs_ = torch.vstack((outputs_,torch.unsqueeze(output,0)))
            return outputs_
        else:
            latent_index = x.size()[1]-1
            output = torch.clone(x)
            for layer_index in range(len(self.decoder_sizes_[latent_index])-2):
                output = self.decoders_[latent_index][layer_index](output)
                output = self.activate(output)
            output = self.decoders_[latent_index][len(self.decoder_sizes_[latent_index])-2](output)
            return output


class HierarchicalAutoencoder(Autoencoder):
    def __init__(self, model_name="", input_size=0, encoder_size=[64,256,64], latent_size=2, decoder_size=[],\
                       random_split_seed=42,num_epochs=50,batch_size=10000,\
                       test_ratio=0.1,learning_rate_start=1e-3,learning_rate_end=-1,loss_weigths=[],hierarchical_weights=[],\
                       verbose=1,select_cuda_id=0):
        super(HierarchicalAutoencoder, self).__init__(model_name=model_name, input_size=input_size, encoder_size=encoder_size, latent_size=latent_size, decoder_size=decoder_size,\
                                                           random_split_seed=random_split_seed,num_epochs=num_epochs,batch_size=batch_size,\
                                                           test_ratio=test_ratio,learning_rate_start=learning_rate_start,learning_rate_end=learning_rate_end,loss_weigths=loss_weigths,\
                                                           verbose=verbose,select_cuda_id=select_cuda_id)
        self.hierarchical_weights = torch.tensor(hierarchical_weights).to(self.device)
        # for p in self.ae_model.parameters():
        #     print(type(p), p.size())
        # exit()

    def CreateAutoencoderModel(self,model_name=""):
        self.ae_model = HierarchicalAutoencoderModel(self.input_size, self.encoder_size, self.latent_size, self.decoder_size)
        if model_name != "":
            self.ae_model.load_state_dict(torch.load(model_name+".pt",map_location=self.device))
        self.ae_model.to(self.device)
    
    def compute_loss(self, outputs_, y, return_hierarchical_components=False):
        loss = None
        loss_components = []
        for latent_index in range(self.latent_size):
            loss_tmp = self.loss_function(outputs_[latent_index], y)
            if len(self.loss_weigths) > 0:          loss_tmp = loss_tmp * self.loss_weigths / self.loss_weigths.sum()
            if len(self.hierarchical_weights) > 0:  loss_tmp = loss_tmp * self.hierarchical_weights[latent_index]
            loss_components.append(torch.mean(loss_tmp))
            if torch.is_tensor(loss) == True:
                loss = loss + loss_components[-1]
            else:
                loss = loss_components[-1]
        if return_hierarchical_components == True: return loss, torch.tensor(loss_components)
        return loss

    def train_epoch(self):
        train_loss = []
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)   
            outputs_ = self.ae_model(X)
            loss = self.compute_loss(outputs_,y)
            train_loss.append(loss.item())
            self.optimization.zero_grad()
            loss.backward()
            self.optimization.step()
        return np.mean(np.array(train_loss))
    
    def train_model(self,raw_data_filename="",output_model_name="tmp",omit_first_i_column=0,interval=1):
        self.reset_timer()
        self.verbose_output_file = open(output_model_name+".out","w")
        self.train_init(raw_data_filename,omit_first_i_column=omit_first_i_column,interval=interval)
        min_test_loss = 1000
        not_improve_count = 0
        not_improve_limit = 10
        improve_criteria = 1e-3
        print("Time for initilization: %.2fs"%(self.timer()))
        if self.verbose==1:
            print("epoch trainloss testloss ",end="")
            for i in range(self.latent_size): print("test_loss_%d "%(i+1),end="")
            print("lr time")
        self.verbose_output_file.write("epoch trainloss testloss ")
        for i in range(self.latent_size): self.verbose_output_file.write("test_loss_%d "%(i+1))
        self.verbose_output_file.write("lr time\n")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.learning_rate_schedule.step()
            test_loss, test_loss_components = self.test_model(return_hierarchical_components=True)
            dt = self.timer()
            if self.verbose==1:
                print("%d %.4e %.4e "%(epoch+1,train_loss,test_loss),end="")
                for a_test_loss_components in test_loss_components: print("%.4e "%(a_test_loss_components),end="")
                print("%.4e %.2f"%(self.optimization.param_groups[0]["lr"],dt))
            self.verbose_output_file.write("%d %.4e %.4e "%(epoch+1,train_loss,test_loss))
            for a_test_loss_components in test_loss_components: self.verbose_output_file.write("%.4e "%(a_test_loss_components))
            self.verbose_output_file.write("%.4e %.2f\n"%(self.optimization.param_groups[0]["lr"],dt))

            if epoch > 0 and (min_test_loss-test_loss)/min_test_loss < improve_criteria:
                not_improve_count += 1
                if not_improve_count == not_improve_limit:
                    break
            else:
                not_improve_count = 0
            min_test_loss = min(min_test_loss, test_loss)

        self.verbose_output_file.write("end of trainning.\n")
        test_loss, test_loss_components = self.test_model(return_hierarchical_components=True)
        self.verbose_output_file.write("Total_test_loss\n")
        self.verbose_output_file.write("%.4e "%(test_loss))
        for a_test_loss_components in test_loss_components: self.verbose_output_file.write("%.4e "%(a_test_loss_components))
        self.verbose_output_file.write("\n")
        self.SaveModel(model_name=output_model_name)
        self.verbose_output_file.close()
        self.verbose_output_file = None

    def test_model(self,data=[],batch_size=-1,return_hierarchical_components=False):
        if len(data) > 0:
            dataset = self.data_to_dataset(data)
            if batch_size == -1: batch_size = len(dataset)
            self.split_dataset(dataset,batch_size,1)
            self.apply_loss()
        if self.test_loader:
            test_loss = []
            test_loss_components = []
            with torch.no_grad():
                for X, y in self.test_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    output = self.ae_model(X)
                    loss = self.compute_loss(output, y, return_hierarchical_components=return_hierarchical_components)
                    if return_hierarchical_components == True:
                        loss, loss_components = loss
                        if torch.is_tensor(test_loss_components) == True:
                            test_loss_components = torch.vstack((test_loss_components,loss_components))
                        else:
                            test_loss_components = torch.clone(loss_components)
                    test_loss.append(loss.item())
                if return_hierarchical_components == True:
                    if len(test_loss_components.size()) == 2: test_loss_components = torch.mean(test_loss_components, 0)
                    return np.mean(np.array(test_loss)), test_loss_components.detach().numpy()
                return np.mean(np.array(test_loss))
        else:
            print("No test data loaded.")

    def decode_data(self,cv,output_filename="",batch_size=-1):
        if not self.ae_model:
            print("No model specified.")
            exit()
        dataset = self.data_to_dataset(cv,is_data_latent=True,ignore_error=True)
        if batch_size == -1: batch_size = len(dataset)
        self.split_dataset(dataset,batch_size,1)
        with torch.no_grad():
            Xp_stack = []
            for cv, _ in self.test_loader:
                cv = cv.to(self.device)
                Xp = self.ae_model.decode(cv)
                if torch.is_tensor(Xp_stack):   Xp_stack = torch.vstack((Xp_stack,Xp))
                else:                           Xp_stack = torch.clone(Xp)
            Xp_stack = Xp_stack.detach().numpy()
            #torch.transpose(Xp_stack,0,1)
            if output_filename != "":
                fout = open(output_filename,"w")
                for i, a_Xp_correction in enumerate(Xp_stack):
                    fout.write("#use first %d CV(s)\n"%(i))
                    for a_Xp_sample in a_Xp_correction:
                        for a_Xp_value in a_Xp_sample:
                            fout.write('%.4e '%(a_Xp_value))
                        fout.write('\n')
                fout.close()
        return Xp_stack
    
    def compute_decoder_jacobian_determinant(self,cv):
        if torch.is_tensor(cv) == False:
            cv = torch.from_numpy(np.array(cv,dtype=np.float32))
        dataset = self.data_to_dataset(cv,is_data_latent=True,ignore_error=True)
        self.split_dataset(dataset,10,1)
        new_decode = lambda cv: self.ae_model.decode(cv, output_last=True)
        det_stack = []
        for cv, _ in self.test_loader:
            cv = cv.to(self.device)
            jacob = torch.autograd.functional.jacobian(new_decode, cv)
            jacob = torch.sum(jacob,dim=2)
            determinant = torch.det(torch.matmul(torch.transpose(jacob,1,2),jacob))
            if torch.is_tensor(det_stack):  det_stack = torch.cat((det_stack,determinant))
            else:                           det_stack = torch.clone(determinant)  
        return det_stack.detach().numpy()

    def LoadModel(self,model_name):
        if os.path.isfile(model_name+".haedat") and os.path.isfile(model_name+".pt"):
            with open(model_name+'.haedat',"r") as fin:
                for aline in fin:
                    if "#" not in aline:
                        linelist = aline.strip().split()
                        if linelist[0]   == "input_size":           self.input_size             = int(linelist[2])
                        elif linelist[0] == "latent_size":          self.latent_size            = int(linelist[2])
                        elif linelist[0] == "encoder_size":         self.encoder_size           = [int(i) for i in linelist[2:]]
                        elif linelist[0] == "decoder_size":         self.decoder_size           = [int(i) for i in linelist[2:]]
                        elif linelist[0] == "random_split_seed":    self.random_split_seed      = int(linelist[2])
                        elif linelist[0] == "num_epochs":           self.num_epochs             = int(linelist[2])
                        elif linelist[0] == "batch_size":           self.batch_size             = int(linelist[2])
                        elif linelist[0] == "test_ratio":           self.test_ratio             = float(linelist[2])
                        elif linelist[0] == "learning_rate" or linelist[0] == "learning_rate_start":    self.learning_rate_start = float(linelist[2])
                        elif linelist[0] == "learning_rate_end":    self.learning_rate_end      = float(linelist[2])
                        elif linelist[0] == "loss_weigths":         self.loss_weigths           = torch.tensor([int(i) for i in linelist[2:]]).to(self.device)
                        elif linelist[0] == "hierarchical_weights": self.hierarchical_weights   = torch.tensor([int(i) for i in linelist[2:]]).to(self.device)

            self.learning_rate_lambda_table_ = np.append(np.logspace(np.log10(self.learning_rate_start),np.log10(self.learning_rate_end),self.num_epochs),[0])/self.learning_rate_start
            self.CreateAutoencoderModel(model_name=model_name)
            return True
        else:
            print("Model %s not found.\nBoth %s.haedat and %s.pt should exist."%(model_name,model_name,model_name))
            return False

    def SaveModel(self,model_name):
        torch.save(self.ae_model.state_dict(), model_name+".pt")
        with open(model_name+'.haedat',"w") as fout:
            fout.write("#--------------------------------------------------\n")
            fout.write("input_size              = %s\n"%(self.input_size))
            fout.write("latent_size             = %s\n"%(self.latent_size))
            fout.write("encoder_size            = ")
            for w in self.encoder_size: fout.write("%s "%(w))
            fout.write("\n")
            fout.write("decoder_size            = ")
            for w in self.decoder_size: fout.write("%s "%(w))
            fout.write("\n")
            fout.write("random_split_seed       = %s\n"%(self.random_split_seed))
            fout.write("num_epochs              = %s\n"%(self.num_epochs))   
            fout.write("batch_size              = %s\n"%(self.batch_size))
            fout.write("test_ratio              = %s\n"%(self.test_ratio))     
            fout.write("learning_rate_start     = %s\n"%(self.learning_rate_start)) 
            fout.write("learning_rate_end       = %s\n"%(self.learning_rate_end))
            if len(self.loss_weigths) > 0:
                fout.write("loss_weigths            = ")
                for w in self.loss_weigths.detach().numpy(): fout.write("%s "%(w))
                fout.write("\n")
            if len(self.loss_weigths) > 0:
                fout.write("hierarchical_weights    = ")
                for w in self.hierarchical_weights.detach().numpy(): fout.write("%s "%(w))
                fout.write("\n")
        if self.verbose==1: print("Model saved to %s.pt and %s.haedat"%(model_name,model_name))
        if self.verbose_output_file != None: self.verbose_output_file.write("Model saved to %s.pt and %s.haedat\n"%(model_name,model_name))    
