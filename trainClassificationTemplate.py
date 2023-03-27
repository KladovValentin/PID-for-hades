
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import uproot
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from tqdm import trange
from pympler import asizeof
from models.model import DANN
from models.model import Model
from dataHandling import My_dataset, DataManager, load_dataset



def train_model(model, train_loader, loss, optimizer, num_epochs, valid_loader, scheduler=None):
    print("start model nn train")
    loss_history = []
    train_history = []
    validLoss_history = []

    for epoch in range(num_epochs):
        model.train()

        loss_train = 0
        accuracy_train = 0
        isteps = 0
        tepoch = tqdm(train_loader,unit=" batch")
        for i_step, (x, y) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            prediction = model(x)
            running_loss = loss(prediction, y)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            indices = torch.max(prediction, 1)[1]
            running_acc = torch.sum(indices==y)/y.shape[0]
            if i_step > len(train_loader)*3./4.:
                accuracy_train += running_acc
                loss_train += float(running_loss)
                isteps += 1

            loss_history.append(float(running_loss))

            tepoch.set_postfix(loss=float(running_loss), accuracy=float(running_acc)*100)
            del indices, prediction, x, y, running_acc, running_loss

        accuracy_train = accuracy_train/isteps
        loss_train = loss_train/isteps

        #<<<< Validation >>>>#
        model.eval()
        loss_valid = 0
        accuracy_valid = 0
        validLosses = []
        validAccuracies = []
        with torch.no_grad():
            for v_step, (x, y) in enumerate(valid_loader):
                prediction = model(x)
                validLosses.append(float(loss(prediction, y)))
                indices = torch.max(prediction, 1)[1]
                validAccuracies.append(float(torch.sum(indices==y))/ y.shape[0])

            loss_valid = np.mean(np.array(validLosses))
            accuracy_valid = np.mean(np.array(validAccuracies))
        model.train() 


        if scheduler is not None:
            #scheduler.step(ave_valid_loss)
            scheduler.step()


        #<<<< Printing and drawing >>>>#
        #loss_history.append(loss_train)
        train_history.append(accuracy_train)
        validLoss_history.append(float(loss_valid))
        ep = np.arange(1,(epoch+1)*(i_step+1)+1,1)
        lv = np.array(validLoss_history)
        lt = np.array(loss_history)
        plt.clf()
        plt.plot(ep,lt,"blue",label="train")
        #plt.plot(ep,lv,"orange",label="validation")
        plt.legend(loc=[0.5,0.6])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if ((epoch+1)%1 == 0):
            plt.show()

        print("Average loss: %f, valid loss: %f, Train accuracy: %f, V acc: %f, epoch: %f" % (loss_train, loss_valid, accuracy_train*100, accuracy_valid*100, epoch+1))
    
    return 1


def train_DANNmodel(model, sim_loader, exp_loader, val_exp_loader, val_sim_loader, lossClass, lossDomain, optimizer, num_epochs, scheduler=None):
    loss_history = []
    train_history = []
    validLoss_history = []
    #ti loh
    len_dataloader = min(len(sim_loader), len(exp_loader))
    len_dataloader1 = min(len(val_sim_loader), len(val_exp_loader))
    for epoch in range(num_epochs):
        sim_iter = iter(sim_loader)
        exp_iter = iter(exp_loader)
        tepoch = tqdm(range(len_dataloader), total=len_dataloader)
        for i_step in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            (s_x, s_y) = next(sim_iter)
            (e_x, _)   = next(exp_iter)

            domain_label = torch.zeros(len(s_y)).long()
            s_class, s_domain = model(s_x)
            s_loss = lossClass(s_class, s_y) + lossDomain(s_domain, domain_label)

            domain_label = torch.ones(len(e_x)).long()
            _, e_domain = model(e_x)
            e_loss = lossDomain(e_domain, domain_label)

            running_loss = s_loss + e_loss

            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            indices = torch.max(s_class, 1)[1]
            running_acc = torch.sum(indices==s_y)/s_y.shape[0]

            train_history.append(1-running_acc)
            loss_history.append(float(running_loss))

            tepoch.set_postfix(loss=float(running_loss), acc=float(running_acc)*100)

        # validation step
        loss_valid = 0
        accuracy_valid = 0
        validLosses = []
        validAccuracies = []
        with torch.no_grad():
            model.eval()
            val_sim_iter = iter(val_sim_loader)
            val_exp_iter = iter(val_exp_loader)
            for j_step in range(len_dataloader1):
                    
                (vs_x, vs_y) = next(val_sim_iter)
                (ve_x, _)   = next(val_exp_iter)
                vs_y = vs_y.long().flatten()

                vdomain_label = torch.zeros(len(vs_y)).long()
                vs_class, vs_domain = model(vs_x)
                #print(s_class)
                vs_loss = lossClass(vs_class, vs_y) + lossDomain(vs_domain, vdomain_label)

                vdomain_label = torch.ones(len(ve_x)).long()
                _, ve_domain = model(ve_x)
                ve_loss = lossDomain(ve_domain, vdomain_label)

                vrunning_loss = vs_loss + ve_loss

                vindices = torch.max(vs_class, 1)[1]
                vrunning_acc = torch.sum(vindices==vs_y)/vs_y.shape[0]

                validLosses.append(vrunning_loss)
                validAccuracies.append(vrunning_acc)

        loss_valid = np.mean(np.array(validLosses))
        accuracy_valid = np.mean(np.array(validAccuracies))
        model.train() 

        if scheduler is not None:
            scheduler.step()

        print("Valid loss: %f, V acc: %f, epoch: %f" % (loss_valid, accuracy_valid*100, epoch+1))
        #asdsadjasd
        #asdsadjasd
        # drawing step
        ep = np.arange(1,(epoch+1)*(i_step+1)+1,1)
        lt = np.array(loss_history)
        at = np.array(train_history)
        plt.clf()
        #plt.plot(ep,at,"orange",label="1-acc")
        plt.plot(ep,lt,"blue",label="loss")
        plt.legend(loc=[0.5,0.6])
        plt.xlabel('step')
        plt.ylabel('Loss')
        plt.show()
    
    model.eval()

    return 1


def train_NN(simulation_path="simu1.parquet", experiment_path="expu1.parquet"):
    print("start nn training")
    
    batch_size = 16000

    dftCorr = pandas.read_parquet(simulation_path).sample(frac=1.0).reset_index(drop=True) # shuffling
    dataTable = dftCorr.sample(frac=0.8).sort_index()
    validTable = dftCorr.drop(dataTable.index)

    dftCorrExp = pandas.read_parquet(experiment_path).sample(frac=1.0).reset_index(drop=True) # shuffling
    dataTableExp = dftCorrExp.sample(frac=0.8).sort_index()
    validTableExp = dftCorrExp.drop(dataTableExp.index)
    
    train_dataset = My_dataset(load_dataset(dataTable))
    valid_dataset = My_dataset(load_dataset(validTable))

    exp_dataset = My_dataset(load_dataset(dataTableExp))
    exp_valset = My_dataset(load_dataset(validTableExp))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)

    exp_dataLoader = DataLoader(exp_dataset, batch_size=batch_size, drop_last=True)
    exp_valLoader = DataLoader(exp_valset, batch_size=batch_size, drop_last=True)

    nClasses = dftCorr[list(dftCorr.columns)[-1]].nunique()
    input_dim = train_dataset[0][0].shape[0]

    del dataTable, validTable, train_dataset, valid_dataset, dftCorr, batch_size
    del exp_dataset, dftCorrExp, exp_valset, dataTableExp, validTableExp

    #nn_model = Model(input_dim=input_dim, output_dim=nClasses)
    nn_model = DANN(input_dim=input_dim, output_dim=nClasses).type(torch.FloatTensor)

    loss = nn.CrossEntropyLoss()
    #loss = nn.MSELoss()
    loss_domain = nn.NLLLoss()

    #optimizer = optim.SGD(nn_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.05)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.00001, betas=(0.5, 0.9), weight_decay=0.0)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.2, factor=0.2)

    print("prepared to train nn")
    #train_model(nn_model, train_loader, loss, optimizer, 10, valid_loader, scheduler = scheduler)
    train_DANNmodel(nn_model, train_loader, exp_dataLoader, exp_valLoader, valid_loader, loss, loss_domain, optimizer, 10, scheduler=scheduler)

    print("trained nn")
    torch.save(nn_model.state_dict(), "tempModel.pt")


print("start_train_python")

dataManager = DataManager()
dataManager.manageDataset("train_dann")

train_NN()

