import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MOSNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal
import time


# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class MemoryTaskSelector(nn.Module):
    def __init__(self, feature_dim, num_tasks, hidden_dim=128, device="cuda"):
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        self.device = device  # Store device information

        # 🔹 Move memory to the correct device
        self.memory = nn.Parameter(torch.randn(num_tasks, feature_dim).to(device))  # Move memory to device

        # 🔹 Task Selector Network (MLP)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        ).to(device)  # Move model to the correct device

    def forward(self, features, task_id=None):
        """
        Args:
            features (torch.Tensor): Input feature representation [B, feature_dim]
            task_id (int, optional): If given, uses memory regularization for this task
        
        Returns:
            torch.Tensor: Task probabilities (shape [B, num_tasks])
            torch.Tensor (optional): Memory loss for regularization
        """
        # 🔹 Ensure `features` is on the correct device
        features = features.to(self.device)
    
        # 🔹 Predict task probabilities
        task_logits = self.fc(features)  # Shape: [B, num_tasks]
        task_probs = F.softmax(task_logits, dim=-1)  
    
        # 🔹 Memory Regularization (Only in Training)
        if task_id is not None:
            # 🔹 Expand memory vector to match batch size
            memory_loss = F.mse_loss(features, self.memory[task_id].unsqueeze(0).expand_as(features).to(self.device))
            return task_probs, memory_loss
        else:
            return task_probs  # Only return probabilities in inference





class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.class_frequencies=None
        self._network = MOSNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ensemble = args["ensemble"]
        self.task_selector = MemoryTaskSelector(768, args["nb_tasks"])
        self.task_optimizer=optim.Adam(
                self.task_selector.parameters(),
            )
        self.eval_cnn_times = [] 

        
        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
    
    def replace_fc(self):       
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task
        
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.replace_fc()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    
    def compute_class_frequencies(self, dataset):
        """
        Compute the frequency of each class in the dataset.
    
        Parameters:
            dataset: The dataset from which to compute class frequencies.
    
        Returns:
            A dictionary with class labels as keys and their frequencies as values.
        """
        class_counts = {}
        for _, label in dataset:
            if label.item() not in class_counts:
                class_counts[label.item()] = 0
            class_counts[label.item()] += 1
    
        # Convert counts to a tensor (or keep as a dictionary if preferred)
        total_samples = sum(class_counts.values())
        class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}
    
        return class_frequencies

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._network.backbone.adapter_update()

        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr *0.1, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params, 
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler
        

    # def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    #     prog_bar = tqdm(range(self.args['tuned_epoch']))
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.backbone.train()

    #         losses = 0.0
    #         correct, total = 0, 0
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
            
    #             output = self._network(inputs, adapter_id=self._cur_task, train=True)
    #             logits = output["logits"][:, :self._total_classes]
    #             logits[:, :self._known_classes] = float('-inf')

    #             loss = F.cross_entropy(logits, targets.long())
    #             loss += self.orth_loss(output['pre_logits'], targets)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             # # using EMA method to merge adapters
    #             if self.args["adapter_momentum"] > 0:
    #                 self._network.backbone.adapter_merge(self.class_frequencies)
                
    #             losses += loss.item()

    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #             total += len(targets)

    #         if scheduler:
    #             scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
    #             self._cur_task,
    #             epoch + 1,
    #             self.args['tuned_epoch'],
    #             losses / len(train_loader),
    #             train_acc,
    #         )
    #         prog_bar.set_description(info)

    #     logging.info(info)

    # def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    #     prog_bar = tqdm(range(self.args['tuned_epoch']))
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.backbone.train()
    
    #         losses = 0.0
    #         correct, total = 0, 0
    
    #         # Initialize class-specific loss tracker
    #         class_losses = torch.zeros(self._total_classes, device=self._device)
    #         class_counts = torch.zeros(self._total_classes, device=self._device)
    
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    
    #             # Forward pass
    #             output = self._network(inputs, adapter_id=self._cur_task, train=True)
    #             logits = output["logits"][:, :self._total_classes]
    #             logits[:, :self._known_classes] = float('-inf')
    
    #             # Compute cross-entropy loss
    #             loss = F.cross_entropy(logits, targets.long(), reduction='none')  # Per-sample loss
    #             loss += self.orth_loss(output['pre_logits'], targets)
    
    #             optimizer.zero_grad()
    #             loss.mean().backward()  # Mean loss for optimization
    #             optimizer.step()
    
    #             # Update EMA for adapters if momentum > 0
    #             if self.args["adapter_momentum"] > 0:
    #                 self._network.backbone.adapter_merge()
    
    #             losses += loss.sum().item()  # Sum loss over batch
    
    #             # Track class-specific losses
    #             for class_id in range(self._total_classes):
    #                 class_mask = (targets == class_id)  # Mask for current class
    #                 class_counts[class_id] += class_mask.sum()
    #                 class_losses[class_id] += loss[class_mask].sum()  # Add losses for the class
    
    #             # Compute training accuracy
    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets).cpu().sum()
    #             total += len(targets)
    
    #         # Normalize class losses (avoid division by zero)
    #         class_losses /= (class_counts + 1e-8)
    
    #         if scheduler:
    #             scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    #         # Pass normalized class losses to the network
    #         self._network.backbone.class_losses = class_losses
    
    #         # Logging progress
    #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
    #             self._cur_task,
    #             epoch + 1,
    #             self.args['tuned_epoch'],
    #             losses / len(train_loader),
    #             train_acc,
    #         )
    #         prog_bar.set_description(info)
    
    #     logging.info(info)
    def dro_loss(self,ce_loss, logits, targets, alpha=0.1):
        """
        Distributionally Robust Optimization (DRO) Loss
        Higher weight for difficult samples
        """
        # Compute per-sample cross-entropy loss
        loss =ce_loss
        
        # Compute weights based on the loss, giving more weight to difficult samples
        weights = (1.0 / (loss + alpha)).detach()  # Avoid division by zero (alpha)
        
        # Weighted loss for each sample
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self.task_selector.train()  # Ensure task selector is in train mode
    
            losses = 0.0
            correct, total = 0, 0
    
            # Initialize class-specific loss tracker
            class_losses = torch.zeros(self._total_classes, device=self._device)
            class_counts = torch.zeros(self._total_classes, device=self._device)
    
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
    
                # 🔹 Step 1: Forward Pass for Main Model (Use correct adapter)
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
    
                # 🔹 Step 2: Compute Classification Loss
                ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')  # Per-sample loss
                loss=ce_loss#self.dro_loss(ce_loss,logits, targets)
                loss += self.orth_loss(output['pre_logits'], targets)  # Add orthogonality loss
    
                optimizer.zero_grad()
                # loss.mean().backward()  # Mean loss for optimization
                loss.mean().backward()
                optimizer.step()
    
                # 🔹 Step 3: Update EMA for Adapters (if applicable)
                if self.args["adapter_momentum"] > 0:
                    self._network.backbone.adapter_merge()
    
                losses += loss.sum().item()  # Sum loss over batch
    
                # Track class-specific losses
                for class_id in range(self._total_classes):
                    class_mask = (targets == class_id)  # Mask for current class
                    class_counts[class_id] += class_mask.sum()
                    class_losses[class_id] += ce_loss[class_mask].sum()  # Add losses for the class
    
                # 🔹 Step 4: Train Task Selector (Using Task-Specific Memory)
                with torch.no_grad():
                    shared_features = self._network.backbone(inputs)["features"].to(self._device)  # ✅ Move to correct device
                
                task_probs, memory_loss = self.task_selector(shared_features, task_id=self._cur_task)  

                
                # Compute task selector loss (task prediction + memory stability)
                task_labels = torch.tensor([self.cls2task[t.item()] for t in targets], device=self._device)
                task_loss = F.cross_entropy(task_probs, task_labels) + 0.1 * memory_loss  # Balance classification & memory loss
                # print(f'task loss: {task_loss}')
                # Optimize task selector
                self.task_optimizer.zero_grad()
                task_loss.backward()
                self.task_optimizer.step()
    
                # 🔹 Compute training accuracy
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
    
            # Normalize class losses (avoid division by zero)
            class_losses /= (class_counts + 1e-8)
    
            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
            # Pass normalized class losses to the network
            self._network.backbone.class_losses = class_losses
    
            # Logging progress
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
    
        logging.info(info)


    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )
            
            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                # print(features_per_cls.shape)
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                # print(features_per_cls.shape)
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"] # 10
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)
                
                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    def classifer_align(self, model):
        model.train()
        
        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([class_idx] * num_sampled_pcls)

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError


            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)
                
                _, preds = torch.max(logits, dim=1)
                
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
         
        logging.info(info)

    def orth_loss(self, features, targets):
        if self.cls_mean:
            # orth loss of this batch
            sample_mean = []
            for k, v in self.cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            # print(loss)
            return self.args["reg"] * loss
            # return 0.1 * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss
            # return 0.0
    
    # def _eval_cnn(self, loader):
    #     self._network.eval()
    #     y_pred, y_true = [], []
    #     orig_y_pred = []
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
    #             orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
    #             orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
                
    #             # test the accuracy of the original model
    #             orig_y_pred.append(orig_preds)
                
    #             all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim, device=self._device)
    #             for t_id in range(self._cur_task + 1):
    #                 t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
    #                 all_features[:, t_id, :] = t_features
                
    #             # self-refined
    #             final_logits = []
                
    #             MAX_ITER = 4
    #             for x_id in range(len(inputs)):
    #                 loop_num = 0
    #                 prev_adapter_idx = orig_idx[x_id]
    #                 while True:
    #                     loop_num += 1
    #                     cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0) # shape=[1, 768]
    #                     cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
    #                     cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
    #                     cur_adapter_idx = torch.tensor([self.cls2task[v] for v in cur_pred], device=self._device)[0]
                        
    #                     if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx:
    #                         break
    #                     else:
    #                         prev_adapter_idx = cur_adapter_idx
                        
    #                 final_logits.append(cur_logits)
    #             final_logits = torch.cat(final_logits, dim=0).to(self._device)
    
    #             if self.ensemble:
    #                 final_logits = F.softmax(final_logits, dim=1)
    #                 orig_logits = F.softmax(orig_logits / (1/(self._cur_task+1)), dim=1)
    #                 outputs = final_logits + orig_logits
    #             else:
    #                 outputs = final_logits
                
    #         predicts = torch.topk(
    #             outputs, k=self.topk, dim=1, largest=True, sorted=True
    #         )[
    #             1
    #         ]  # [bs, topk]
    #         y_pred.append(predicts.cpu().numpy())
    #         y_true.append(targets.cpu().numpy())
    
    #     orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
    #     logging.info("the accuracy of the original model:{}".format(np.around(orig_acc, 2)))
    #     return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    # def _eval_cnn(self, loader):
    #     self._network.eval()
    #     y_pred, y_true = [], []
    #     orig_y_pred = []
    
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             # 1️⃣ Compute Original Model Logits & Predictions
    #             orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
    #             orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
    #             orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
    
    #             # Store original predictions
    #             orig_y_pred.append(orig_preds)
    
    #             # 2️⃣ Store All Adapter Features
    #             all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim, device=self._device)
    #             for t_id in range(self._cur_task + 1):
    #                 t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
    #                 all_features[:, t_id, :] = t_features
    
    #             # 3️⃣ Compute Soft Teacher Targets for Knowledge Distillation
    #             T = 2.0  # Temperature for distillation
    #             orig_teacher_probs = F.softmax(orig_logits / T, dim=1).detach()  # Detach to avoid gradient flow
    
    #             # 4️⃣ Self-Refinement Process with KD + Enhancements
    #             final_logits = []
    #             MAX_ITER = 4
    
    #             for x_id in range(len(inputs)):
    #                 loop_num = 0
    #                 prev_adapter_idx = orig_idx[x_id]
    #                 prev_logits = orig_logits[x_id].unsqueeze(0)  # Store original logits
                    
    #                 while True:
    #                     loop_num += 1
    #                     cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0)  # Extract feature
    #                     cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
                        
    #                     # Compute soft probabilities for refined logits
    #                     cur_probs = F.log_softmax(cur_logits / T, dim=1)  # Log-softmax for KL-div
                        
    #                     # Compute KL-Divergence Loss (Knowledge Distillation)
    #                     distillation_loss = F.kl_div(cur_probs, orig_teacher_probs[x_id].unsqueeze(0), reduction='batchmean')
    
    #                     # Adaptive Distillation Weighting based on confidence score
    #                     confidence_score = torch.max(orig_teacher_probs[x_id])
    #                     weight = 0.1 * (1 - confidence_score)  # Higher weight for uncertain predictions
    #                     adjusted_logits = cur_logits - (weight * distillation_loss)  # Apply KD loss
    
    #                     # Update cur_adapter_idx based on refined prediction
    #                     cur_pred = torch.max(adjusted_logits, dim=1)[1].cpu().numpy()
    #                     cur_adapter_idx = torch.tensor([self.cls2task[v] for v in cur_pred], device=self._device)[0]
    
    #                     # Compute entropy difference to determine early stopping
    #                     entropy_diff = torch.sum(-cur_probs * cur_probs.exp()) - torch.sum(-prev_logits * prev_logits.exp())
                        
    #                     # Stop if converged, reached max iterations, or entropy change is small
    #                     if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx or entropy_diff < 0.01:
    #                         break
    #                     else:
    #                         prev_adapter_idx = cur_adapter_idx
    #                         prev_logits = adjusted_logits  # Save logits for next iteration
    
    #                 final_logits.append(adjusted_logits)
    
    #             # 5️⃣ Combine Final Logits
    #             final_logits = torch.cat(final_logits, dim=0).to(self._device)
    
    #             # 6️⃣ Ensemble Strategy (Optional)
    #             if self.ensemble:
    #                 final_logits = F.softmax(final_logits, dim=1)
    #                 orig_logits = F.softmax(orig_logits / (1/(self._cur_task+1)), dim=1)
    #                 outputs = final_logits + orig_logits
    #             else:
    #                 outputs = final_logits
    
    #         # 7️⃣ Prediction & Accuracy Calculation
    #         predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    #         y_pred.append(predicts.cpu().numpy())
    #         y_true.append(targets.cpu().numpy())
    
    #     # Compute original accuracy
    #     orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
    #     logging.info("The accuracy of the original model: {}".format(np.around(orig_acc, 2)))
    
    #     return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    def _eval_cnn(self, loader):
        start_time = time.time()
        self._network.eval()
        self.task_selector.eval()  # Ensure task selector is in eval mode
    
        y_pred, y_true = [], []
        orig_y_pred = []
    
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 🔹 Step 1: Get Initial Predictions from Original Model
                orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
    
                # 🔹 Store original predictions for accuracy evaluation
                orig_y_pred.append(orig_preds)
    
                # 🔹 Step 2: Extract Shared Features
                shared_features = self._network.backbone(inputs)["features"]
    
                # 🔹 Step 3: Use Task Selector to Predict the Best Adapter
                task_probs = self.task_selector(shared_features)  # Shape: [B, num_tasks]
                adapter_idx = torch.argmax(task_probs, dim=1)  # Select highest probability adapter
    
                # 🔹 Step 4: Extract Features Using the Selected Adapter
                final_logits = []
                for i in range(inputs.shape[0]):  # Iterate over the batch
                    selected_features = self._network.backbone(
                        inputs[i].unsqueeze(0),  # Extract single sample
                        adapter_id=int(adapter_idx[i].item()),  # Convert tensor to int
                        train=False
                    )["features"]
    
                    # Get logits for this sample
                    logits = self._network.backbone(selected_features, fc_only=True)["logits"][:, :self._total_classes]
                    final_logits.append(logits)
    
                # 🔹 Stack logits to reconstruct the batch
                final_logits = torch.cat(final_logits, dim=0).to(self._device)
    
                # 🔹 Step 5: Ensemble (if enabled)
                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1 / (self._cur_task + 1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits
    
            # 🔹 Step 6: Get Final Predictions
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [batch_size, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
    
        # 🔹 Step 7: Compute Original Model Accuracy
        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
                # Store execution time and calculate average
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.eval_cnn_times.append(elapsed_time)
        avg_time = sum(self.eval_cnn_times) / len(self.eval_cnn_times)

        print(f"Time taken for _eval_cnn: {elapsed_time:.4f} seconds")
        print(f"Average time for _eval_cnn calls: {avg_time:.4f} seconds")
        logging.info("The accuracy of the original model: {}".format(np.around(orig_acc, 2)))
    
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

