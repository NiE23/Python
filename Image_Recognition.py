import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

from matplotlib import pyplot as plt
from datetime import datetime
from zipfile  import ZipFile
import numpy as np
import os

import requests
import io

### CHANGE HERE: ###
#Hardware-Einstellungen
num_workers = 0
use_gpu = False

#Anzeige von Bildern
show_pics = True

#Path for Save
save_if = 70 #percent when the ai should be saved
path_save = ""

#Load existing AI
is_load = False
path_load_ai = ""
path_savedata = ""

#Testdata
load_testdata = False
path_test = ""

#Neuronal Network
epochs = 2
lrate = 0.0002

hidden_nodes_1 = 250
hidden_nodes_2 = 150
hidden_nodes_3 = 75
### END OF CHANGE ###

#Constants
momentum = 0.9
input_nodes = 256
output_nodes = 43

columns = 5
rows = 2
showing_images = int(round(columns * rows))

#Datasets
url_traindata = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip"
url_testdata = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

#Classes
classes = [None] * 43
classes[0] = 'Geschwindigkeit 20'
classes[1] = 'Geschwindigkeit 30'
classes[2] = 'Geschwindigkeit 50'
classes[3] = 'Geschwindigkeit 60'
classes[4] = 'Geschwindigkeit 70'
classes[5] = 'Geschwindigkeit 80'
classes[6] = 'Aufhebung Geschwindigkeit'
classes[7] = 'Geschwindigkeit 100'
classes[8] = 'Geschwindigkeit 120'
classes[9] = 'Überholverbot PKW'
classes[10] = 'Überholverbot LKW'
classes[11] = 'Vorfahrt Kreuzung'
classes[12] = 'Vorfahrt Straße'
classes[13] = 'Vorfahrt gewähren'
classes[14] = 'STOP'
classes[15] = 'Durchfahrt verboten'
classes[16] = 'Durchfahrt verboten LKW'
classes[17] = 'Durchfahrt verboten Einbahnstraße'
classes[18] = 'Achtung'
classes[19] = 'Achtung steile links Kurve'
classes[20] = 'Achtung steile rechts Kurve'
classes[21] = 'Achtung S-Kurve'
classes[22] = 'Achtung Bodenwelle'
classes[23] = 'Achtung Rutschgefahr'
classes[24] = 'Achtung Engstelle'
classes[25] = 'Achtung Baustelle'
classes[26] = 'Achtung Ampel'
classes[27] = 'Achtung Fußgänger'
classes[28] = 'Achtung Kinder'
classes[29] = 'Achtung Fahrrad'
classes[30] = 'Achtung Schnee und Eis'
classes[31] = 'Achtung Wild'
classes[32] = 'Aufhebung alles'
classes[33] = 'Nur rechts abbiegen'
classes[34] = 'Nur links abbiegen'
classes[35] = 'Nur geradeaus'
classes[36] = 'Nur rechts oder geradeaus'
classes[37] = 'Nur links oder geradeaus'
classes[38] = 'Nur rechts'
classes[39] = 'Nur links'
classes[40] = 'Kreisverkehr'
classes[41] = 'Aufhebung Überholverbot PKW'
classes[42] = 'Aufhebung Überholverbot LKW'

#Startmethod
started = datetime.now()
print("{0} // AI started...".format(started.strftime("%d/%m/%Y %H:%M:%S")))
print(("\n\033[1mParameter\033[0m\n"+ 
       "Date: {0}\n" + 
       "Epochs: {1}\n" + 
       "Learning Rate: {2}\n" + 
       "Hidden 1: {3}\n" +
       "Hidden 2: {4}\n" + 
       "Hidden 3: {5}\n").format(started.strftime("%d/%m/%Y %H:%M:%S"),
                                 epochs, 
                                 lrate, 
                                 hidden_nodes_1, 
                                 hidden_nodes_2, 
                                 hidden_nodes_3))

class Network(nn.Module):
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, output_nodes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.input_nodes = input_nodes
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.hidden_nodes_3 = hidden_nodes_3
        self.output_nodes = output_nodes

        self.fc1 = nn.Linear(self.input_nodes, self.hidden_nodes_1)
        self.fc2 = nn.Linear(self.hidden_nodes_1, self.hidden_nodes_2)
        self.fc3 = nn.Linear(self.hidden_nodes_2, self.hidden_nodes_3)
        self.fc4 = nn.Linear(self.hidden_nodes_3, self.output_nodes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_nodes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

net = Network(input_nodes, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, output_nodes)

if use_gpu == True:
    net.cuda()

if is_load == True:
   net.load_state_dict(torch.load(path_load_ai))
else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((48,48)), 
                                    transforms.Normalize((0.5, 0.5, 0.5), 
                                                         (0.5, 0.5, 0.5))])
    
    print("\n{0} // Download started...".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    r = requests.get(url_traindata)
    z = ZipFile(io.BytesIO(r.content))
    path_train = path_savedata + "/Training"
    try:
        os.makedirs(path_train, exist_ok=True)
    except:
        pass
    z.extractall(path_train)

    print("\n{0} // Finished Get Trainingsdata...".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    if load_testdata == False:
        r = requests.get(url_testdata)
        z = ZipFile(io.BytesIO(r.content))
        path_test = path_savedata + "/Test"
        try:
            os.makedirs(path_test, exist_ok=True)
        except:
            pass
        z.extractall(path_test)

        test_dataset = torchvision.datasets.ImageFolder(root=path_test, 
                                                    transform=transform)

        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size = 1, 
                                                  num_workers=num_workers, 
                                                  shuffle=True)                                                     

    train_dataset = torchvision.datasets.ImageFolder(root=path_train, 
                                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size = 1, 
                                               num_workers=num_workers, 
                                               shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
                          lr=lrate, 
                          momentum=momentum)

    for e in range(epochs):
        run_acc = 0.0
        run_loss = 0.0
        percent = 0
        for i, data in enumerate(train_loader, 0):
            line = {}
            inputs, labels = data
            if use_gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            _, pred = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            equals = pred == labels
            accItem = torch.mean(equals.type(torch.FloatTensor)).item()
            run_acc += accItem
            run_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i == 0:
                print ("\n\tEpoche {0} von {1} - Progress: {2} %".format(e + 1, epochs, percent), end="")

            if i % int((len(train_loader)/100)) == 0 and i != 0:
                percent += 1
                print ("\r\tEpoche {0} von {1} - Progress: {2} %".format(e + 1, epochs, percent), end="")
                pass
            pass
        pass

score_percent = 0
score = 0
percent = 0

net.eval()
if show_pics == False:
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        if use_gpu == True:
            inputs, labels = inputs.cuda(), labels.cuda()
            pass

        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        guess = int(preds[0])
        right = int(labels[0])

        if guess == right:
            score += 1
            pass

        if i == 0:
            print("\tProgress: {0} %".format(percent), end="")

        if i % int((len(test_loader)/100)) == 0 and i != 0:
            percent += 1
            print ("\r\tProgress: {0} %".format(percent), end="")
            pass
        pass
        
    score_percent = (score/len(test_loader)) * 100
else:
    #Wenn weniger Bilder verfügbar, dann sollen weniger Druchläufe stattfinden
    if len(test_loader) < showing_images:
        showing_images = len(test_loader)

    #Wenn weniger Bilder als Spalten, dann braucht nur eine Reihe sein
    if showing_images <= columns:
        rows = 1
    else:
        rows = int(round(showing_images / columns))

    score = 0
    magic_nbr = 6
    height = rows * magic_nbr
    width = columns * magic_nbr
    fig=plt.figure(figsize=(width, height))

    for i in range(showing_images):
        dataiter = iter(test_loader)
        inputs, labels = dataiter.next()
        if use_gpu == True:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        guess_nbr = int(preds[0])
        guess = classes[guess_nbr]
        right_nbr = int(labels[0])
        right = classes[right_nbr]

        if guess_nbr == right_nbr:
            score += 1
            pass

        fig.add_subplot(rows, columns, i + 1)
        title = "Guess: ({0}) - {1}\nRight: ({2}) - {3}".format(guess_nbr, guess, right_nbr, right)
        plt.title(title)
        if use_gpu == True:
            plt.imshow(imshow(inputs.cpu()[0]))
        else:
            plt.imshow(imshow(inputs[0]))
        pass

    plt.show()
    score_percent = (score/showing_images) * 100

if score_percent > save_if:
    os.makedirs(path_save, exist_ok=True)
    filepath = '{0}/AI_Traffic.pth'.format(path_save)
    torch.save(net.state_dict(), filepath)

    file_parameters = "{0}/Parameters.csv".format(path_save)

    if os.path.exists(file_parameters) == False:
        with open(file_parameters, 'w') as datei:
            datei.write('"ID","Datum","Epochen","Lernrate","Hidden 1","Hidden 2","Hidden 3","Score"')

    with open(file_parameters, 'a') as datei:
        datei.write('\n"{0}","{1}","{2}","{3}","{4}","{5}","{6}%"'.format(started.strftime("%d.%m.%Y %H:%M:%S"), 
                                                                                epochs,
                                                                                "{0:.5f}".format(lrate).replace('.', ','),
                                                                                hidden_nodes_1, 
                                                                                hidden_nodes_2,
                                                                                hidden_nodes_3,     
                                                                                "{0:.2f}".format(score_percent).replace('.', ',')))

ended = datetime.now()
gesamt = (ended - started).total_seconds()
min = int(gesamt / 60)
sec = int(gesamt - (min * 60))

durchlauf = 'Zeit des Durchlaufes: {0} Min und {1} Sek'.format(min, sec)
print("{0} // KI beendet - {1}".format(ended.strftime("%d/%m/%Y %H:%M:%S"), durchlauf))