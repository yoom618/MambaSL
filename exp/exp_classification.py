from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        self.train_data, self.train_loader = self._get_data(flag='TRAIN')
        if self.args.is_training:
            self.vali_data, self.vali_loader = self._get_data(flag='TEST')
        self.test_data, self.test_loader = self._get_data(flag='TEST')

        self.args.seq_len = max(self.train_data.max_seq_len, self.test_data.max_seq_len)
        self.args.label_len = 0
        self.args.pred_len = 0
        self.args.enc_in = self.train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, flag='val'):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                if isinstance(batch_x, tuple):
                    # for TSCMamba, batch_x is a list of [XCWT, XROCKET]
                    batch_x = [x.float().to(self.device) for x in batch_x]
                    padding_mask = None
                else:
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs[0], outputs[1]  # for visualization

                pred = outputs.detach().clone().cpu()
                loss = criterion(pred, label.long().squeeze(-1).cpu())
                if self.args.model in ['InterpretGN']:
                    loss += self.model.loss().mean().cpu()  # add sbm loss
                total_loss.append(loss)

                preds.append(outputs.detach().clone())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        if (self.args.model == 'MambaSingleLayer') and self.args.save_log:
            self.writer.add_pr_curve(f'pr_curve_{flag}', trues, predictions)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        # train_data, train_loader = self._get_data(flag='TRAIN')
        # vali_data, vali_loader = self._get_data(flag='TEST')
        # test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.use_pretrained and os.path.exists(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')):
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        if self.args.model == 'MambaSingleLayer' and self.args.save_log:
            self.writer = SummaryWriter(os.path.join(self.args.checkpoints, self.args.model_id, 'train_logs', setting))
            self.writer.add_text('model', setting)  # save model name in tensorboard
        
        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if isinstance(batch_x, tuple):
                    # for TSCMamba, batch_x is a list of [XCWT, XROCKET]
                    batch_x = [x.float().to(self.device) for x in batch_x]
                    padding_mask = None
                else:
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs[0], outputs[1]  # for visualization
                
                loss = criterion(outputs, label.long().squeeze(-1))
                if self.args.model in ['InterpretGN']:
                    loss += self.model.loss().mean().cpu()  # add sbm loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                

                if (self.args.model == 'MambaSingleLayer') and self.args.save_log:
                    if i == 0 and epoch == 0:
                        self.writer.add_graph(self.model, (batch_x, padding_mask))  # save model graph

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(self.vali_data, self.vali_loader, criterion, 'val')
            test_loss, test_accuracy = self.vali(self.test_data, self.test_loader, criterion, 'test')


            if (self.args.model == 'MambaSingleLayer') and self.args.save_log:
                self.writer.add_scalar('loss/train', train_loss, epoch+1)
                self.writer.add_scalar('loss/val', vali_loss, epoch+1)
                self.writer.add_scalar('accuracy/val', val_accuracy, epoch+1)
                self.writer.add_histogram('gating_values', torch.tensor(self.model.gating_values), epoch+1)  # save gating values histogram
                self.writer.add_histogram('attn_weights', self.model.attn_weight[0].weight, epoch+1)  # save attention weights histogram
                self.writer.add_histogram('attn_bias', self.model.attn_weight[0].bias, epoch+1)  # save attention bias histogram
                self.writer.add_histogram('mamba_A', self.model.mamba[0].A_log, epoch+1)  # save mamba_A histogram
                self.writer.add_histogram('mamba_dt_bias', self.model.mamba[0].dt_proj.bias, epoch+1)  # save mamba_dt_bias histogram
                if self.args.tv_dt:
                    self.writer.add_histogram('mamba_dt_weight', self.model.mamba[0].dt_proj.weight @
                        self.model.mamba[0].x_proj.weight[:self.model.mamba[0].tv_proj_dim[0]], epoch+1)  # save mamba_dt_weight histogram
                if self.args.tv_B:
                    self.writer.add_histogram('mamba_B_weight', self.model.mamba[0].x_proj.weight[
                        self.model.mamba[0].tv_proj_dim[0]:self.model.mamba[0].tv_proj_dim[0] + self.model.mamba[0].tv_proj_dim[1]], epoch+1)  # save mamba_B_weight histogram
                else:
                    self.writer.add_histogram('mamba_B', self.model.mamba[0].B, epoch+1)  # save mamba_B histogram
                if self.args.tv_C:
                    self.writer.add_histogram('mamba_C_weight', self.model.mamba[0].x_proj.weight[
                        self.model.mamba[0].tv_proj_dim[0] + self.model.mamba[0].tv_proj_dim[1]:], epoch+1)  # save mamba_C_weight histogram
                else:
                    self.writer.add_histogram('mamba_C', self.model.mamba[0].C, epoch+1)  # save mamba_C histogram
                if self.args.use_D:
                    self.writer.add_histogram('mamba_D', self.model.mamba[0].D, epoch+1)  # save mamba_D histogram

            if (self.args.model == 'MambaSingleLayer') and (self.args.mamba_projection_type == 'gating'):
                print(f"gating_value: {np.mean(self.model.gating_values, axis=0)}")
                self.model.gating_values.clear()

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            vali_loss = np.inf if np.isnan(vali_loss) else vali_loss
            early_stopping(-val_accuracy + 1e-5*vali_loss, self.model, path)
            if early_stopping.early_stop or val_accuracy == 1.0 or np.isinf(vali_loss):
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        if self.args.model == 'MambaSingleLayer' and self.args.save_log:
            hdict = {
                k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self.args.__dict__.items()
            }
            self.writer.add_hparams(hdict, {'hparam/val_accuracy': val_accuracy,
                                            'hparam/val_loss': vali_loss})
            self.writer.close()

        return self.model

    def test(self, setting, test=0):
        # test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        
        if self.args.model == 'MambaSingleLayer' and self.args.save_log:
            self.writer = SummaryWriter(os.path.join(self.args.checkpoints, self.args.model_id, 'test_logs', setting))
            self.writer.add_text('model', setting)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                if isinstance(batch_x, tuple):
                    # for TSCMamba, batch_x is a tuple of [XCWT, XROCKET]
                    batch_x = [x.float().to(self.device) for x in batch_x]
                    padding_mask = None
                else:
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                if self.args.model == 'MambaSingleLayer' and self.args.save_log and i == 0:
                    self.writer.add_graph(self.model, (batch_x, padding_mask))  # save model graph

                outputs = self.model(batch_x, padding_mask, None, None)

                if isinstance(outputs, tuple):
                    outputs, remainders = outputs[0], outputs[1]  # for visualization
                    if self.args.model == 'MambaSingleLayer' and self.args.save_csv:
                        save_csv_dir = os.path.join(self.args.checkpoints, setting, 'csv')
                        if i == 0:
                            os.makedirs(save_csv_dir, exist_ok=True)
                            print(f'saving csv to {save_csv_dir}')

                        ### save each sample's results
                        import pandas as pd
                        np_x_enc = (batch_x * padding_mask.unsqueeze(2)).cpu().numpy()  # (B, L_in, C_in)
                        np_mamba_out = remainders[0].cpu().numpy()  # (B, L_in, D)
                        np_logit_out = remainders[1].cpu().numpy()  # (B, L_in, C_out)
                        np_w_out = remainders[2].cpu().numpy()  # (B, L_in, 1)
                        np_out = outputs.detach().clone().argmax(dim=1).cpu().numpy()  # (B, C_out) -> (B, )
                        np_label = label.squeeze(1).cpu().numpy()  # (B, C_out) -> (B, )
                        for batch_idx in range(np_x_enc.shape[0]):
                            df = pd.concat([
                                pd.DataFrame(np_x_enc[batch_idx], columns=[f'feature_{i}' for i in range(np_x_enc.shape[2])]),
                                pd.DataFrame(np_mamba_out[batch_idx], columns=[f'hidden_{i}' for i in range(np_mamba_out.shape[2])]),
                                pd.DataFrame(np_logit_out[batch_idx], columns=[f'logit_{i}' for i in range(np_logit_out.shape[2])]),
                                pd.DataFrame(np_w_out[batch_idx], columns=[f'attn_{i}' for i in range(np_w_out.shape[2])])
                                ], axis=1)
                            df.to_csv(os.path.join(save_csv_dir,
                                f'{self.args.model_id}_{i*self.args.batch_size + batch_idx}_label{np_label[batch_idx]:d}_pred{np_out[batch_idx]:d}.csv'), index=False)


                preds.append(outputs.detach().clone())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        
        if (self.args.model == 'MambaSingleLayer') and self.args.save_log:
            hdict = {
                k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self.args.__dict__.items()
            }
            self.writer.add_hparams(hdict, {'hparam/accuracy': accuracy})  # save hyperparameters in tensorboard
            self.writer.add_histogram('gating_values', torch.tensor(self.model.gating_values), self.args.train_epochs)  # save gating values histogram
            self.writer.add_histogram('attn_weights', self.model.attn_weight[0].weight, self.args.train_epochs)  # save attention weights histogram
            self.writer.add_histogram('attn_bias', self.model.attn_weight[0].bias, self.args.train_epochs)  # save attention bias histogram
            self.writer.add_histogram('mamba_A', self.model.mamba[0].A_log, self.args.train_epochs)  # save mamba_A histogram
            self.writer.add_histogram('mamba_dt_bias', self.model.mamba[0].dt_proj.bias, self.args.train_epochs)  # save mamba_dt_bias histogram
            if self.args.tv_dt:
                self.writer.add_histogram('mamba_dt_weight', self.model.mamba[0].dt_proj.weight @
                    self.model.mamba[0].x_proj.weight[:self.model.mamba[0].tv_proj_dim[0]], self.args.train_epochs)  # save mamba_dt_weight histogram
            if self.args.tv_B:
                self.writer.add_histogram('mamba_B_weight', self.model.mamba[0].x_proj.weight[
                    self.model.mamba[0].tv_proj_dim[0]:self.model.mamba[0].tv_proj_dim[0] + self.model.mamba[0].tv_proj_dim[1]], self.args.train_epochs)  # save mamba_B_weight histogram
            else:
                self.writer.add_histogram('mamba_B', self.model.mamba[0].B, self.args.train_epochs)  # save mamba_B histogram
            if self.args.tv_C:
                self.writer.add_histogram('mamba_C_weight', self.model.mamba[0].x_proj.weight[
                    self.model.mamba[0].tv_proj_dim[0] + self.model.mamba[0].tv_proj_dim[1]:], self.args.train_epochs)  # save mamba_C_weight histogram
            else:
                self.writer.add_histogram('mamba_C', self.model.mamba[0].C, self.args.train_epochs)  # save mamba_C histogram
            if self.args.use_D:
                self.writer.add_histogram('mamba_D', self.model.mamba[0].D, self.args.train_epochs)  # save mamba_D histogram
            self.writer.close()  # close the writer

        if (self.args.model == 'MambaSingleLayer') and (self.args.mamba_projection_type == 'gating'):
            print(f"gating_value: {np.mean(self.model.gating_values, axis=0)}")
            self.model.gating_values.clear()

        print('accuracy:{}'.format(accuracy))
        print(f'model parameter : {sum(p.numel() for p in self.model.parameters())}')
        print(f'model size : {os.path.getsize(os.path.join(self.args.checkpoints, setting, "checkpoint.pth")) / 1024 / 1024:.2f}MB')


        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
