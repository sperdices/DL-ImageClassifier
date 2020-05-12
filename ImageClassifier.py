import json
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

# TODO: Fix bug with the epoch print in the last batch of the epochs
# TODO: Change steps to number of images
# TODO: Extract dataset name from path

class ImageClassifier:
    def __init__(self):
        # Device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Device to bring back to cpu
        self.device_cpu = torch.device("cpu")
        
        # data_directory expects: /train /valid /test
        self.data_directory = None
        # dataset folder name
        self.dataset = 'Flowers'
        
        # Transforms for the training, validation, and testing sets
        self.transform = {}
        # ImageFolder data training, validation, and testing sets
        self.data = {}
        # Data loaders for the training, validation, and testing sets
        self.batch_size = 32
        self.loader = {}
        
        # Normalization parameters
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        # DL model
        self.model = None
        # DL architecture to load (default value)
        self.arch = 'vgg13'
        # Hidden units of the classifier (default value)
        self.hidden_units = [512, 256]
        # Number of classes of the classifier (set from data['train'].class_to_idx)
        self.nclasses = None
        
        # Criterion and probability function
        self.criterion = nn.CrossEntropyLoss() 
        self.prob_func = nn.Softmax(dim=1)
        
        # Criterion and probability function
        #self.criterion = nn.NLLLoss()
        #self.prob_func = torch.exp()

        # Optimizer (Adam)
        self.optimizer = None

        # Optimizer learning_rate (default value)
        self.learning_rate = 0.001

        # Training settings
        self.trainer = {    'epochs_to_train': 10,
                            'print_every': 10,
                            'mute': False,
                            'train_losses': [],
                            'validation_losses': [],
                            'accuracy_partials': [],
                            'valid_loss_min': np.inf,
                            'epoch_best': -1,
                            'epoch': [],
                            'step': [],
                            'epochs_acum': 0 }
        
        # Training stats
        self.running_train_loss = 0
        self.step_cur = 0
        self.step_last = 0
        self.epochs_start = 0
        self.epochs_last = 0
        self.training_start_time = 0
        self.training_last_time = 0
        self.valid_time = 0

        # Dictionaries
        self.class_to_idx = None
        self.idx_to_class = None
        self.class_to_name = None

        # Default checkpoint values
        self.save_directory = 'checkpoints'
        self.get_default_ckeckpoint_name = lambda: ('ckp_' + self.dataset 
                                                    + '_' + self.arch 
                                                    + '_' + "_".join(str(x) for x in self.hidden_units) 
                                                    + '_' + str(self.learning_rate))
                                                    #+ '_' + str(self.trainer['epochs_acum']))

    def use_gpu(self, gpu):
        if gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")


    def load_data(self, data_directory):       
        try:
            self.data_directory = os.path.expanduser(data_directory)
            self.dataset = self.dataset # TODO: get dataset name as the folder name of the dataset

            print("Loading dataset {} from {}".format(self.dataset, self.data_directory))

            train_dir = os.path.join(self.data_directory, 'train')
            valid_dir = os.path.join(self.data_directory, 'valid')
            test_dir = os.path.join(self.data_directory, 'test')

            # Define your transforms for the training, validation, and testing sets
            self.transform['train'] = transforms.Compose([transforms.RandomRotation(30),
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(self.norm_mean, self.norm_std)])

            self.transform['valid'] = transforms.Compose([transforms.Resize(255),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(self.norm_mean, self.norm_std)])

            self.transform['test'] = transforms.Compose([transforms.Resize(255),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(self.norm_mean, self.norm_std)])

            # Load the datasets with ImageFolder    
            self.data['train'] = datasets.ImageFolder(train_dir, transform=self.transform['train'])
            self.data['valid'] = datasets.ImageFolder(valid_dir, transform=self.transform['valid'])
            self.data['test'] = datasets.ImageFolder(test_dir, transform=self.transform['test'])

            # Using the image datasets and the trainforms, define the dataloaders
            self.loader['train'] = torch.utils.data.DataLoader(self.data['train'], batch_size=self.batch_size, shuffle=True)
            self.loader['valid'] = torch.utils.data.DataLoader(self.data['valid'], batch_size=self.batch_size)
            self.loader['test'] = torch.utils.data.DataLoader(self.data['test'], batch_size=self.batch_size, shuffle=True)

            # Save class_to_idx and idx_to_class
            self.class_to_idx = self.data['train'].class_to_idx
            self.idx_to_class = { v: k for k, v in self.class_to_idx.items() }

            # set classifier number of classes
            self.nclasses = len(self.data['train'].class_to_idx)
            
            return True
        except Exception as e:
            print("[ERR] Loading data:", str(e))
            return False  


    def load_class_names(self, filepath):
        filepath = os.path.expanduser(filepath)
        try:
            with open(filepath, 'r') as f:
                self.class_to_name = json.load(f)
            return True
        except Exception as e:
            print("[ERR] Loading class names json:", str(e))
            return False  


    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio (thumbnail or resize)
        image.thumbnail((256, 256))
        
        # Crop Center
        width, height = image.size
        new_width, new_height = 224, 224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        image = image.crop((left, top, right, bottom))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1: 
        image_np = image_np / image_np.max()
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        
        # Move color channel from 3rd to 1st
        image_np = image_np.transpose(2, 1, 0)

        return image_np


    def crete_output_classifier(self, in_units):
        units = self.hidden_units.copy()
        units.insert(0,in_units)
        layers_dict =  OrderedDict([])
        for i in range(len(units)-1):
            layers_dict['fc'+str(i+1)] = nn.Linear(units[i], units[i+1])
            #layers_dict['relu'+str(i+1)] = nn.ReLU(inplace=True)
            layers_dict['relu'+str(i+1)] = nn.ReLU()
            layers_dict['drop'+str(i+1)] = nn.Dropout(0.2)

        layers_dict['fc'+str(len(self.hidden_units)+1)] = nn.Linear(units[-1], self.nclasses)   
        #layers_dict['output'] =  nn.LogSoftmax(dim=1)
        return nn.Sequential(layers_dict)


    def create_model(self, arch=None, hidden_units=None):
        self.arch = arch if arch is not None else self.arch
        self.hidden_units = hidden_units if hidden_units is not None else self.hidden_units

        print('Creating model:', self.arch, 'with hidden_units:', " ".join(str(x) for x in self.hidden_units), 'and nclass:', self.nclasses)
        
        # create self.model from pre-trained network
        if self.arch == 'vgg13':
            self.model = models.vgg13(pretrained=True) 
            # Freeze parameters so we don't backprop through them
            for param in self.model.parameters():
                    param.requires_grad = False
            # Replace last part of the pre-trained network
            start_units = self.model.classifier[0].in_features #25088
            self.model.classifier = self.crete_output_classifier(start_units)
            self.model.param_to_optimize = self.model.classifier.parameters()

        elif self.arch == 'vgg16_bn':
            self.model = models.vgg16_bn(pretrained=True) 
            # Freeze parameters so we don't backprop through them
            for param in self.model.parameters():
                    param.requires_grad = False
            # Replace last part of the pre-trained network
            start_units = self.model.classifier[0].in_features #25088
            self.model.classifier = self.crete_output_classifier(start_units)
            self.model.param_to_optimize = self.model.classifier.parameters()

        elif self.arch == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            # Freeze parameters so we don't backprop through them
            for param in self.model.parameters():
                    param.requires_grad = False
            # Replace last part of the pre-trained network
            start_units = self.model.classifier.in_features #1024
            self.model.classifier = self.crete_output_classifier(start_units)
            self.model.param_to_optimize = self.model.classifier.parameters()
            
        elif self.arch == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            # Freeze parameters so we don't backprop through them
            for param in self.model.parameters():
                    param.requires_grad = False
            # Replace last part of the pre-trained network
            start_units = self.model.fc.in_features #2048
            self.model.fc = self.crete_output_classifier(start_units)
            self.model.param_to_optimize = self.model.fc.parameters()
        
        else:
            print("[ERR] creating_model invalid arch:", self.arch)
            return None

        self.model = self.model.to(self.device)
        return self.model


    def create_optimizer(self, lr=None):
        self.learning_rate = lr if lr is not None else self.learning_rate

        # Only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.param_to_optimize, lr=self.learning_rate)    
        return self.optimizer


    def print_stats(self):
        e = self.trainer['epochs_acum']
        nstep = len(self.loader['train'])

        step_remaining = (self.epochs_last - e)*nstep - self.step_cur
        time_spend = time.time() - self.training_last_time
        speed = (self.step_cur - self.step_last)/time_spend
        time_remaining = step_remaining / speed

        print("Epoch: {}/{}.. ".format(e+1, self.epochs_last),
            "Step: {}/{}.. ".format(self.step_cur, nstep),
            "Train Loss: {:.3f}.. ".format(self.trainer['train_losses'][-1]),
            "Valid Loss: {:.3f}.. ".format(self.trainer['validation_losses'][-1]),
            "Valid Accuracy: {:.3f}.. ".format(self.trainer['accuracy_partials'][-1]),
            "Time: {}s/{}s/{}m{}s".format(int(self.valid_time), int(time_spend), int(time_remaining//60), int(time_remaining%60)) )
            #"Time: {}s".format(int(time_spend)),
            #"Remainig: {}m{}s".format(int(time_remaining//60), int(time_remaining%60)))


    def validation(self, save_ckp=False):
        valid_loss, accuracy = 0, 0
        self.valid_time = time.time()
        self.model.eval() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        with torch.no_grad():
            for images, labels in self.loader['valid']:
                # Move input and label tensors to the default self.device
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model.forward(images)
                valid_loss += self.criterion(outputs, labels).item() 
                
                # Calculate accuracy
                _, top_class = torch.max(outputs, 1)
                accuracy += torch.mean((top_class == labels.data).type(torch.FloatTensor)).item()  
        self.model.train() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.valid_time = time.time()-self.valid_time
        
        valid_loss /= len(self.loader['valid']) 
        accuracy /= len(self.loader['valid'])

        # Save results
        self.trainer['train_losses'].append(self.running_train_loss/(self.step_cur-self.step_last)) 
        self.trainer['validation_losses'].append(valid_loss)
        self.trainer['accuracy_partials'].append(accuracy)
        self.trainer['epoch'].append(self.trainer['epochs_acum'])
        self.trainer['step'].append(self.step_cur)

        # Print results
        if not self.trainer['mute']:
            self.print_stats()

        # Save checkpoint
        if save_ckp:
            if valid_loss < self.trainer['valid_loss_min']:
                self.trainer['valid_loss_min'] = valid_loss
                self.trainer['epoch_best'] = self.trainer['epochs_acum']
                self.save_checkpoint(best=True)
            else:
                self.save_checkpoint(best=False)


    def train(self, epochs_to_train = None, save_directory = None, print_every = None):       
        self.trainer['epochs_to_train'] =  epochs_to_train if epochs_to_train is not None else self.trainer['epochs_to_train']
        self.trainer['print_every'] =  print_every if print_every is not None else self.trainer['print_every']
        self.save_directory = save_directory if save_directory is not None else self.save_directory
        self.verify_directory()

        print("Training {} epoch using {}".format(self.trainer['epochs_to_train'], self.device))      

        # Set variables for the training
        self.running_train_loss, self.step_cur, self.step_last = 0, 0, 0
        self.training_start_time, self.training_last_time = time.time(), time.time()
        self.epochs_start = self.trainer['epochs_acum']
        self.epochs_last = self.trainer['epochs_acum'] + self.trainer['epochs_to_train']
        
        try:
            # model in training mode, dropout is on
            self.model.train() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
                    
            for e in range(self.epochs_start, self.epochs_last):   
                for images, labels in self.loader['train']:
                    self.step_cur += 1
                    
                    # Move input and label tensors to the default self.device
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()          
                    output = self.model.forward(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    self.optimizer.step()                    
                    self.running_train_loss += loss.item() 

                    if self.step_cur % self.trainer['print_every'] == 0:
                        # Run validation pass, save and results
                        self.validation(save_ckp=False)                  
                        
                        # Reset variables per end of print
                        self.running_train_loss = 0
                        self.step_last = self.step_cur
                        self.training_last_time = time.time()
                # End of epoch
                else:                
                    self.trainer['epochs_acum'] += 1

                    # Run validation pass, save and results
                    self.validation(save_ckp=True)
                    
                    # Reset variables per end of epoch                   
                    self.running_train_loss = 0
                    self.step_cur = 0
                    self.step_last = 0                
                    self.training_last_time = time.time()
                    
        except KeyboardInterrupt:
            print("Exiting training: KeyboardInterrupt")

            # Run validation pass, save and results
            print("Running final validation step...")
            self.validation(save_ckp=True)

        finally:
            # Print the training time
            time_duration = time.time()-self.training_start_time
            print("Training duration: {}m{}s".format(int(time_duration//60), int(time_duration%60)))

            # Plot the results
            plt.plot(self.trainer['train_losses'], label='Training loss')
            plt.plot(self.trainer['validation_losses'], label='Validation loss')
            plt.plot(self.trainer['accuracy_partials'], label='Acuracy')
            plt.legend(frameon=False)
            plt.savefig(os.path.join(self.save_directory, self.get_default_ckeckpoint_name()+'.png'))
            plt.show()          


    def test(self, topk = 2, show_failures=False):
        print(f"Testing using: {str(self.device)}")
        
        corrects_acum, accuracy_count = 0, 0
        #images, labels = next(iter(self.loader['test']))
        for images, labels in self.loader['test']:           
            # Move input and label tensors to the default self.device
            images, labels = images.to(self.device), labels.to(self.device)

            # Disable dropouts and turn off gradients to speed up this part
            self.model.eval() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            with torch.no_grad():
                outputs = self.model.forward(images)

                _, top_class = outputs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                corrects = equals.sum()
                corrects_acum += corrects
                accuracy_count += images.size(0)
                accuracy = float(corrects) / images.size(0) * 100
                print('Accuracy partial: {}/{}[{:.2f}%]'.format(corrects, images.size(0), accuracy))           
                
            # if show_failures and accuracy < 1:
            #     print("The following are the mistakes:")

            #     # Bring back to cpu
            #     images, ps_all = images.to(self.device_cpu), ps_all.to(self.device_cpu)
            #     top_class, labels = top_class.to(self.device_cpu), labels.to(self.device_cpu)
                
            #     # Show the failures
            #     for i,img in enumerate(images):
            #         if top_class[i] != labels[i]:
            #             top_p_i, top_class_i = ps_all[i].topk(topk)
            #             self.view_classify(img, top_p_i, top_class_i, correct=labels[i].item())

        accuracy_total = float(corrects_acum) / accuracy_count * 100
        print('\nAccuracy total: {}/{}[{:.2f}%]'.format(corrects_acum, accuracy_count, accuracy_total))           


    def predict(self, image_path, topk=1, show_image=True):
        ''' Predict the class (or classes) of an image using a trained deep learning self.model.
        '''
        image_path = os.path.expanduser(image_path)      
        # Load image
        try:
            img_pil = Image.open(image_path)
        except Exception as e:
            print('[ERR] In predict opening image: ' + str(e))
            return None, None

        # Process image
        image_np = self.process_image(img_pil)
        image = torch.from_numpy(image_np).unsqueeze(0)
        
        print("Predict using: ", self.device)
        # input and label tensors to the default self.device
        image = image.to(self.device, dtype=torch.float)

        # Turn off gradients to speed up this part
        self.model.eval() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
        with torch.no_grad():
            # If model's output is log-softmax, take exponential to get the probabilities
            # If model's output is linear, take softmax to get the probabilities
            ps = self.prob_func(self.model.forward(image))
            top_p, top_class = ps.topk(topk, dim=1)

        # Bring back to CPU
        image, top_p, top_class = image.to(self.device_cpu), top_p.to(self.device_cpu), top_class.to(self.device_cpu)

        image, top_p, top_class = image.squeeze(0), top_p.squeeze(0), top_class.squeeze(0)
        
        if show_image:
            self.view_classify(image, top_p, top_class)

        return {'top_p': top_p, 'top_class': top_class}


    def print_predictions(self, predictions):
        top_class = predictions['top_class'].numpy()
        top_p = predictions['top_p'].numpy()

        top_class_print = [self.idx_to_class[i] for i in top_class]
        
        if self.class_to_name is not None:
            top_class_print = [self.class_to_name[i] for i in top_class_print]

        for p, c in zip(top_p, top_class_print):
            print('[{}]: {:.2f}%'.format(c, p*100))


    def view_classify(self, img, top_p, top_class, correct=None):
        ''' Function for viewing an image and it's predicted classes.
        '''
        topk = len(top_p)

        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1 = self.imshow(img, ax1)   
        ax1.axis('off')
        ax2.barh(np.arange(topk), top_p)
        ax2.set_yticks(np.arange(topk))
        ax2.set_aspect(0.1)   
        if self.class_to_name is None:
            ax2.set_yticklabels(["{}[{}]".format(self.idx_to_class.get(i),i) for i in top_class.numpy()], size='small') 
        else:
            ax2.set_yticklabels(["{}[{}]".format(self.class_to_name.get(self.idx_to_class.get(i)),i) for i in top_class.numpy()], size='small') 
        if correct is not None:
            if self.class_to_name is None:
                ax2.set_title('Class Prob. [correct:{}[{}]]'.format(self.idx_to_class.get(correct),correct))
            else:
                ax2.set_title('Class Prob. [correct:{}[{}]]'.format(self.class_to_name.get(self.idx_to_class.get(correct)),correct))
        else:
            ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()


    def imshow(self, image, ax=None, title=None, normalize=True):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))

        if normalize:
            mean = np.array(self.norm_mean)
            std = np.array(self.norm_std)
            image = std * image + mean
            image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        return ax


    def verify_directory(self):
        if not os.path.isdir(self.save_directory):
            os.mkdir(self.save_directory)


    def save_checkpoint(self, save_directory=None, best=False):        
        filepath_base = os.path.join(self.save_directory, self.get_default_ckeckpoint_name())
        filepath = filepath_base + "_last.pth"

        checkpoint = {
                        'arch': self.arch,
                        'hidden_units': self.hidden_units,
                        'nclasses': self.nclasses,
                        'class_to_idx': self.class_to_idx,
                        'idx_to_class': self.idx_to_class,
                        'model_state_dict': self.model.state_dict(),
                        
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'learning_rate': self.learning_rate,

                        'trainer': self.trainer
                    }
        torch.save(checkpoint, filepath)
        print("Checkpoint saved:", filepath)
       
        if best:
            filepath = filepath_base + "_best.pth"
            torch.save(checkpoint, filepath)
            print("Checkpoint saved:", filepath)
        

    def load_checkpoint(self, filepath='checkpoint.pth'):   
        filepath = os.path.expanduser(filepath)      
        if os.path.isfile(filepath):
            print("Loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            
            # Build a model with the checkpoint data
            self.arch = checkpoint['arch']
            self.hidden_units = checkpoint['hidden_units']
            self.nclasses = checkpoint['nclasses']
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = checkpoint['idx_to_class']
            # create_model() needs to be here for model.to(device) be called before creating the optimizer... 
            self.create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(self.model)

            self.learning_rate = checkpoint['learning_rate']
            print("Optimizer:\n", self.create_optimizer())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.trainer = checkpoint['trainer']

            # Print checkpoint info
            print('Trainer epoch_best/epochs_acum: {}/{}'.format(self.trainer['epoch_best'],
                                                                self.trainer['epochs_acum']))
            print('Trainer min_loss:', self.trainer['valid_loss_min'])
            print('Trainer accuracy Last/Max: {:.2f}%/{:.2f}%'.format(self.trainer['accuracy_partials'][-1]*100, 
                                                                np.max(self.trainer['accuracy_partials'])*100))                                                       
            return True
        else:
            print("[ERR] Loading checkpoint path '{}'".format(filepath))
            return False
