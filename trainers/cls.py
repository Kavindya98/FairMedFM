import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from trainers.base import BaseTrainer

from utils.metrics import evaluate_binary, organize_results, log_metrics_to_wandb
from utils.basics import creat_folder
from utils.lr_sched import adjust_learning_rate


class CLSTrainer(BaseTrainer):
    def __init__(self, args, model, test_loader, logger) -> None:
        super().__init__(args, model, logger)
        self.test_loader = test_loader
    
    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        worst_auc = float('-inf')

        while self.epoch < self.total_epochs:
            adjust_learning_rate(self.optimizer, self.epoch + 1, self.args)

            organized_metrics = self.train_epoch(train_dataloader)
            self.logger.info("epoch {}: lr {}, loss {}".format(
                self.epoch, self.optimizer.param_groups[0]["lr"], organized_metrics["loss"]))
            log_metrics_to_wandb(self.epoch, organized_metrics, panel="train")

            if val_dataloader is not None:
                organized_metrics_val = self.evaluate(val_dataloader)
                log_metrics_to_wandb(self.epoch, organized_metrics_val, panel="val")
                if organized_metrics_val["worst-auc"] > worst_auc:
                    worst_auc = organized_metrics_val["worst-auc"]
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "epoch": self.epoch,
                        },
                        os.path.join(self.args.save_folder, "ckpt_best.pth"),
                    )
                    organized_metrics_test = self.evaluate(self.test_loader)
                    log_metrics_to_wandb(self.epoch, organized_metrics_test, panel="test")


            
            # if self.epoch % 5 == 0:
            #     self.evaluate(self.test_loader)

            # save model
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                },
                os.path.join(self.args.save_folder, "ckpt.pth"),
            )

            if self.args.early_stopping and val_dataloader is not None:
                if self.epoch > 10 and (max(self.last_five_auc) - min(self.last_five_auc) < 1e-5):
                    break

            self.epoch += 1
            
    def train_epoch(self, train_dataloader):
        loss_epoch = []
        prob_list = []
        target_list = []
        sensitive_list = []
        for minibatch in train_dataloader:
            if hasattr(train_dataloader, "class_weights_y"):
                loss_batch, outcome = self.update_batch(minibatch, train_dataloader.class_weights_y)
            else:
                loss_batch, outcome = self.update_batch(minibatch, None)

            loss_epoch.append(loss_batch.item())
            prob_list.append(outcome[0])
            target_list.append(outcome[1])
            sensitive_list.append(outcome[2])
        
        prob_list = torch.concat(prob_list).squeeze().cpu().numpy()
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)
        overall_metrics, subgroup_metrics = evaluate_binary(prob_list[:, 1], target_list, sensitive_list)
        organized_metrics = organize_results(overall_metrics, subgroup_metrics)
        organized_metrics["loss"] = np.mean(loss_epoch)

        return organized_metrics

    def update_batch(self, minibatch, class_weights=None):
        x = minibatch["img"].to(self.device)
        y = minibatch["label"].to(self.device)

        if self.args.is_3d:
            # input shape: B x N_slice x 3 x H x W
            # output shape: B x N_slice x N_classes
            # logits list for each slice
            logits_sliced = torch.stack([self.model(x[:, i]) for i in range(x.shape[1])], dim=1)
            prob_sliced = F.softmax(logits_sliced, dim=-1)
            indices = torch.argmax(prob_sliced[:, :, 1], dim=1)

            logits = torch.stack([logits_sliced[i, idx] for i, idx in enumerate(indices)])
            if class_weights is not None :
                loss = F.cross_entropy(logits, y.long().squeeze(-1), weight=class_weights.to(self.device))
            else:
                loss = F.cross_entropy(logits, y.long().squeeze(-1))
        else:
            logits = self.model(x)
            if class_weights is not None:
                loss = F.cross_entropy(logits, y.long().squeeze(-1), weight=class_weights.to(self.device))
            else:
                loss = F.cross_entropy(logits, y.long().squeeze(-1))
        prob = F.softmax(logits, dim=-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss,[prob.detach(), y, minibatch["sensitive"]]

    def evaluate(self, dataloader, save_path=None):
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        logits_list = []
        prob_list = []
        target_list = []
        sensitive_list = []

        for minibatch in dataloader:
            x = minibatch["img"].to(self.device)
            y = minibatch["label"].to(self.device)
            a = minibatch["sensitive"].to(self.device)

            with torch.no_grad():
                if self.args.is_3d:
                    logits_sliced = torch.stack([self.model(x[:, i]) for i in range(x.shape[1])], dim=1)
                    prob_sliced = F.softmax(logits_sliced, dim=-1)
                    indices = torch.argmax(prob_sliced[:, :, 1], dim=1)

                    logits = torch.stack([logits_sliced[i, idx] for i, idx in enumerate(indices)])
                else:
                    logits = self.model(x)
            
            loss = F.cross_entropy(logits, y.long().squeeze(-1))
            prob = F.softmax(logits, dim=-1)
            total_loss += loss.item()
            num_batches += 1
            logits_list.append(logits)
            prob_list.append(prob)
            target_list.append(y)
            sensitive_list.append(a)

        average_loss = total_loss / num_batches if num_batches > 0 else 0
        logits_list = torch.concat(logits_list).squeeze().cpu().numpy()
        prob_list = torch.concat(prob_list).squeeze().cpu().numpy()
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)
        overall_metrics, subgroup_metrics = evaluate_binary(prob_list[:, 1], target_list, sensitive_list)
        organized_metrics = organize_results(overall_metrics, subgroup_metrics)

        if self.args.early_stopping == 1:
            if len(self.last_five_auc) >= 5:
                self.last_five_auc.pop(0)

            self.last_five_auc.append(overall_metrics["auc"])

        self.logger.info("----------------------------------------------".format(self.epoch))
        self.logger.info("----------------eva epoch {}------------------".format(self.epoch))
        self.logger.info(
            "{}".format(
                ", ".join("{}: {}".format(k, v) for k, v in organized_metrics.items()),
            )
        )
        self.logger.info("----------loss {}-------------".format(average_loss))   
        self.logger.info("-----------------meta info-------------------")
        self.logger.info(
            "overall metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in overall_metrics.items()),
            )
        )
        self.logger.info(
            "subgroup metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in subgroup_metrics.items()),
            )
        )
        self.logger.info("----------------------------------------------".format(self.epoch))

        # save predictions
        if save_path is not None:
            creat_folder(save_path)

            with open(os.path.join(save_path, "metrics.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "overall": overall_metrics, "subgroup": subgroup_metrics}, f)

            with open(os.path.join(save_path, "predictions.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "logits": logits_list, "label": target_list}, f)
        organized_metrics["loss"] = average_loss
        return organized_metrics
    
    
