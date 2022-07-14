# IMPORTS
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from fastai import *
from fastai.vision import *
from fastai.callback import *
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import scipy
import time
import copy
import pickle
import sklearn.metrics
import os
import csv

# DATA PATHS
DATA_CSV_PATH    = '*data.csv'
TRAIN_DATA_PATH  = '*/Data/train'
DATA_CSV = open(os.path.expanduser(DATA_CSV_PATH))
train_labels = pd.read_csv(DATA_CSV)

# SPLITTING TRAIN/VALIDATION DATA
fracture_ratio = np.sum(train_labels['presence_of_fracture'].values == 1)/train_labels.shape[0]
train_ratio = 0.6
val_ratio = 0.4
test_ratio = 0.5
random.seed(69) # NOICE
fract_tol = 0.1
keep_splitting = True

# FIRST SPLIT
while keep_splitting:
  val_indicies = random.sample(list(range(train_labels.shape[0])), int(val_ratio * train_labels.shape[0]))
  val_fract_ratio = np.sum(train_labels.loc[val_indicies,'presence_of_fracture'] == 1)/len(val_indicies)
  if abs(val_fract_ratio - fracture_ratio) < fract_tol:
    keep_splitting = False

# SECOND SPLIT
keep_splitting = True
while keep_splitting:
  test_indicies = random.sample(list(range(len(val_indicies))), int(test_ratio * len(val_indicies)))
  test_fract_ratio = np.sum(train_labels.loc[test_indicies,'presence_of_fracture'] == 1)/len(test_indicies)
  if abs(test_fract_ratio - fracture_ratio) < fract_tol:
    keep_splitting = False

train_labels['Set'] = 0
train_labels.loc[val_indicies, 'Set'] = 1
train_labels.loc[test_indicies, 'Set'] = 2

# TRANSFORMS
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10, expand=False),
    transforms.Resize((224, 224)),
    transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    ])

normalize_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Normalize((.485, .456, .406),(.229, .224, .225))
    ])

# STANDARDIZING IMAGES
def standardize_image(img):
    standardized_img = (img - np.min(img))/(np.max(img) - np.min(img))
    return standardized_img

# READING IMAGES
def read_image(fname, crop_side_dim):
  img = Image.open(f"{TRAIN_DATA_PATH}/{fname}")
  if len(np.array(img).shape) > 2:
    img = img.convert('L')
  img = np.array(img)
  img = scipy.ndimage.zoom(img,crop_side_dim/img.shape[0],order = 3)
  return img

# BATCH SIZE FOR TRAINING
batch_size  = 20

# NUMBER OF WORKERS FOR TRAINING
num_workers = 0

# RESIZING IMAGE FOR MODEL INPUT
resize = 224

# DATA CLASS
class Data(Dataset):
    def __init__(self, labels_df, set_name, transforms = None,
        resized_image_length = resize):
        
        if set_name == 'train':
            self.labels_df = labels_df.loc[labels_df.Set == 0, :]
        elif set_name == 'val':
            self.labels_df = labels_df.loc[labels_df.Set == 1, :]
        elif set_name == 'test':
            self.labels_df = labels_df.loc[labels_df.Set == 2, :]
        else:
            print("Wrong set name was given")
        
        self.images     = list(self.labels_df['img'])
        self.labels     = list(self.labels_df['presence_of_fracture'])
        self.len        = len(self.labels)
        self.transforms = transforms
        self.resized_image_length = resized_image_length

    def __getitem__(self, index):
        image_path = self.images[index]
        label      = self.labels[index]
        image = read_image(image_path,self.resized_image_length)

        # Normalize the image
        image = standardize_image(image)

        # Stack image 3 times for 3 input channels
        image = np.stack([image,image,image],axis = 0)
        image = image.astype(np.float64)
        image = torch.from_numpy(image)

        # Transform
        if self.transforms:
            image = self.transforms(image)

        label = torch.from_numpy(np.asarray(label))
        return image.type(torch.FloatTensor), label.type(torch.LongTensor)

    def __len__(self):
        return self.len

# BUILDING TRAINING AND VALIDATION SET
trainset = Data(labels_df=train_labels, set_name='train', transforms = transform)
valset   = Data(labels_df=train_labels, set_name='val', transforms = normalize_transform)
testset  = Data(labels_df=train_labels, set_name='test', transforms = normalize_transform)

# BUILDING TRAINING AND VALIDATION DATA LOADERS
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers,
                             drop_last=True)

valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             drop_last=False)

testset_loader = DataLoader(testset, batch_size=1, shuffle=False,
                             num_workers=num_workers,
                             drop_last=False)

# GET MODEL
model = models.efficientnet_b1(pretrained=True)
model.classifier._modules['1'] = nn.Linear(in_features=1280, out_features=2, bias=True)

# SET DATA TO CUDA GPU IF AVAILABLE
print("GPU available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Model successfully assigned to:", device)
#print(model)

# HYPER PARAMETERS

# NUMBER OF EPOCHS
num_epochs = 50

# LEARNING RATE
lr = 0.001

# WEIGHTS FOR CONTROL/COVID (DUE TO DATA DISTRIBUTION)
class_weights = torch.tensor([0.1,5.]).cuda()

# LOSS FUNCTION
criterion = nn.CrossEntropyLoss(weight=class_weights)

# OPTIMIZER
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# SCHEDULER FOR LEARNING RATE DECAY
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# TRAINING FUNCTION
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25, return_stats=False):
    since = time.time()

    # INITIALIZE best_model_wts TO KEEP TRACK OF BEST MODEL
    best_model_wts = copy.deepcopy(model.state_dict())

    # PERFORMANCE TRACKERS
    best_avg = 1e-3
    epoch_losses = {'train': [], 'val': []}
    epoch_accs = {'train': [], 'val': []}
    epoch_sens = {'train': [], 'val': []}
    epoch_specs = {'train': [], 'val': []}

    # START TRAINING
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # SEPARATE TRAIN AND VALIDATION PHASES
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_sens_corrects = 0
            running_spec_corrects = 0
            total_positives = 1e-4
            total_normals   = 1e-4

            # ITERATE OVER DATA
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # ZERO GRADIENTS
                optimizer.zero_grad()

                # FORWARD
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs, labels)
                    loss = criterion(outputs, labels)
                    # BACKPROPAGATION AND OPTIMIZER STEP FOR TRAINING PHASE
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # PERFORMANCE STATISTICS
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_sens_corrects += torch.sum(preds[labels.data == 1] == labels.data[labels.data == 1])
                running_spec_corrects += torch.sum(preds[labels.data == 0] == labels.data[labels.data == 0])

                total_positives += torch.sum(labels.data == 1)
                total_normals   += torch.sum(labels.data == 0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc  = running_corrects.double() / (total_positives + total_normals)
            epoch_sen  = running_sens_corrects.double() / total_positives
            epoch_spec = running_spec_corrects.double() / total_normals
            epoch_avg = (epoch_sen + epoch_spec)/2

            epoch_losses[phase].append(epoch_loss)
            epoch_accs[phase].append(epoch_acc)
            epoch_sens[phase].append(epoch_sen)
            epoch_specs[phase].append(epoch_spec)

            print('{} loss: {:.4f} acc: {:.2f}% sens: {:.2f}% spec: {:.2f}%'.format(
                phase, epoch_loss, 100*epoch_acc, 100*epoch_sen, 100*epoch_spec))

            # COPY BEST MODEL WEIGHTS
            if phase == 'val' and epoch_avg > best_avg:
                print('Updating best model')
                best_avg = epoch_avg
                best_model_wts = copy.deepcopy(model.state_dict())

        
        time_elapsed = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best avg: {:4f}'.format(100*best_avg))

    # LOAD BEST MODEL FOR FINAL STATISTICS
    model.load_state_dict(best_model_wts)

    if return_stats:
        return model, epoch_losses, epoch_accs, epoch_sens, epoch_spec
    else:
        return model

# START TRAINING
dataloaders = {'train': trainset_loader, 'val': valset_loader}
model, losses, accs, sens, spec = train_model(dataloaders, model, criterion, 
                            optimizer, scheduler, num_epochs, return_stats=True)

# SAVE BEST MODEL
#save_path = ""
#torch.save(model.state_dict(), save_path)

# GET STATS
#stats = {'losses': losses, 'accs': accs, 'sens': sens, 'spec': spec}

# DUMP STATS
#with open('.pkl', 'wb') as handle:
    #pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

# LOADING EXISTING MODEL AND STATISTICS
#odel.load_state_dict(torch.load(""))
#model.eval()

#ith open('.pkl', 'rb') as handle:
    #stats = pickle.load(handle)
#losses = stats['losses']
#accs  = stats['accs']


# PERFORMANCE PLOTS
# LOSSES
plt.figure()
plt.plot(losses['train'])
plt.plot(losses['val'])
plt.xlabel('epoch')
plt.xticks(np.arange(0, len(losses['train'])))
plt.ylabel('loss')
plt.legend(['train', 'val'])

# ACCURACIES
accs_train = [k.cpu() for k in accs['train']]
accs_val = [k.cpu() for k in accs['val']]
plt.figure()
plt.plot(accs_train)
plt.plot(accs_val)
plt.xlabel('epoch')
plt.xticks(np.arange(0, len(accs['train'])))
plt.ylabel('accuracy')
plt.legend(['train', 'val'])

# FINAL VALIDATION METRICS
def metrics(dataloader, model):
    sm = torch.nn.Softmax(dim = 1)
    correct = 0
    total = 0

    correct_pos = 0
    correct_neg = 0
    true_pos = 0
    true_neg = 0

    all_pred_logits = torch.zeros(0)
    all_labels = torch.zeros(0)

    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            all_pred_logits = torch.cat((all_pred_logits,sm(outputs)[:,1].cpu()),dim = 0)
            all_labels = torch.cat((all_labels,labels.cpu()),dim = 0)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            true_pos += (labels == 1).sum().item()
            true_neg += (labels == 0).sum().item()

            correct_pos += (predicted[labels == 1] == labels[labels == 1]).sum().item()
            correct_neg += (predicted[labels == 0] == labels[labels == 0]).sum().item()
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, 100 * correct_pos / true_pos, 100 * correct_neg / true_neg, all_pred_logits, all_labels

overall_acc, overall_sens, overall_spec, preds, labels = metrics(valset_loader, model)

print('Accuracy of the network on the test images: %d %%' % overall_acc)
print('Sensitivity of the network on the test images: %d %%' % overall_sens)
print('Specificity of the network on the test images: %d %%' % overall_spec)

# AUC AND ROC CALCULATION AND PLOT
fpr, tpr, _ = sklearn.metrics.roc_curve(labels.numpy(), preds.numpy(), pos_label = 1)
roc_auc = sklearn.metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Vertebral Fracture classification ROC Curve: Validation')
plt.legend(loc="lower right")
plt.show()

# FOR TEST SET
final_acc, final_sens, final_spec, final_preds, final_labels = metrics(testset_loader, model)

print('Accuracy of the network on the test images: %d %%' % final_acc)
print('Sensitivity of the network on the test images: %d %%' % final_sens)
print('Specificity of the network on the test images: %d %%' % final_spec)

# AUC AND ROC CALCULATION AND PLOT
fpr, tpr, _ = sklearn.metrics.roc_curve(labels.numpy(), preds.numpy(), pos_label = 1)
roc_auc = sklearn.metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Vertebral Fracture classification ROC Curve: Test')
plt.legend(loc="lower right")
plt.show()