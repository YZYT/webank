import os
from pprint import pprint

import torch
import torch.optim as optim
from torch import nn

# import passport_generator
from dataset import prepare_dataset, prepare_wm
from dataloader import prep_dataloader, toy_dataloader
from experiments.base import Experiment
from experiments.trainer import Trainer, Tester
from experiments.trainer_private import TesterPrivate
# from experiments.utils import construct_passport_kwargs, load_passport_model_to_normal_model, \
#     load_normal_model_to_passport_model, load_normal_model_to_normal_model
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport import AlexNetPassport
from models.resnet_normal import ResNet18, ResNet9
from models.lenet import LeNet, LeNet_passport, ToyNet
# from models.resnet import ResNet18
from models.resnet_passport import ResNet18Passport, ResNet9Passport
from optimizers.SWA import SWA
from optimizers.Lookahead import Lookahead

from configs import lr_configs


class ClassificationExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256,
            'imagenet1000': 1000,
            'mnist': 10
        }[self.dataset]

        """
        use mine data augumentation
        """
        self.train_datas, self.valid_data = prepare_dataset(self.args)
        # self.train_datas, self.valid_data = toy_dataloader(self.args)
        # self.train_data, self.valid_data = prep_dataloader(self.args)

  
        self.construct_models()
        
        self.construct_optimizers()


        if len(self.lr_config['scheduler']) != 0:  # if no specify steps, then scheduler = None
            self.schedulers = [optim.lr_scheduler.StepLR(optimizer, **self.lr_config['scheduler']) for optimizer in self.optimizers]
        else:
            self.schedulers = None

        self.trainer = Trainer(self.models, self.optimizers, self.schedulers, self.device)

        self.makedirs_or_load()


    def construct_optimizers(self):

        # self.optimizers = [
        #     optim.SGD([
        #         {'params': model.encoder.parameters(), 'lr': 0.0001},
        #         {'params': model.net1.parameters()},
        #         {'params': model.net2.parameters()},
        #         ], **self.lr_config['optimizer']) 
        #     for model in self.models]

        self.optimizers = [optim.SGD([
                {'params': model.encoder.parameters(), 'lr': 0.01},
                {'params': model.net1.parameters(), 'lr': 0.01},
                {'params': model.net2.parameters(), 'lr': 0.01},
        ], **self.lr_config['optimizer']) for model in self.models]
        
        if self.args['SWA']:
            print('SWA training ...')
            SWA_config = getattr(lr_configs, self.args['SWA_config'])
            steps_per_epoch = int(len(self.train_data.dataset) / self.args['batch_size'])
            print("update per epoch:", steps_per_epoch)
            self.swa_start = self.args['epochs'] * SWA_config['SWA_ratio']
            self.optimizer = SWA(self.optimizer, swa_start=self.swa_start * steps_per_epoch,
                            swa_freq=steps_per_epoch, swa_lr=SWA_config['SWA_lr'])
            print(self.optimizer)
        
        elif self.args['LA']:
            print('Lookahead training ...')
            LA_config = getattr(lr_configs, self.args['LA_config'])
            self.optimizer = Lookahead(self.optimizer, **LA_config)
            print(self.optimizer)

    def construct_models(self):
        print('Construct Model ...')

        def load_pretrained():
            if self.pretrained_path is not None:
                sd = torch.load(self.pretrained_path)
                model.load_state_dict(sd)


        self.is_baseline = True

        if self.arch == 'alexnet':
            self.models = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
        elif self.arch == 'lenet':
            self.models = [LeNet_passport(self.in_channels, self.num_classes).cuda() for _ in range(self.K)]
            # self.models = [LeNet(self.in_channels, self.num_classes).cuda() for _ in range(self.K)]
            # self.models = [ToyNet().cuda() for _ in range(self.K)]
        else:
            ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
            model = ResNetClass(num_classes=self.num_classes, norm_type=self.norm_type)

        load_pretrained()
        # self.model = model.to(self.device)

        # pprint(self.model)



    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        print('Start training ...')

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(ep, self.train_datas)
            # train_metrics = self.trainer.train_one(ep, self.train_datas[0])
            
            if self.args['SWA'] and ep >= self.swa_start:
                # Batchnorm update
                self.optimizer.swap_swa_sgd()
                self.optimizer.bn_update(self.train_data, self.model, device='cuda')
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')
                self.optimizer.swap_swa_sgd()
            
            elif self.args['LA']:
                self.optimizer._backup_and_load_cache()
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')
                self.optimizer._clear_and_load_backup()
            else:
                valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

           
            metrics = {'epoch': ep}
            # for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                # self.save_model('best.pth')

            # self.save_last_model()

    def evaluate(self):
        self.trainer.test(self.valid_data)
