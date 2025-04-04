from collections import Counter, defaultdict
import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score



from trainers.base import BaseTrainer

from utils.metrics import evaluate_binary, organize_results, log_metrics_to_wandb, find_threshold, binary_classification_report, find_best_threshold_youden
from utils.basics import creat_folder
from utils.lr_sched import adjust_learning_rate


class UnSupCLSTrainer(BaseTrainer):
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
        predicted_list = []

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
            predicted_list.append(prob.argmax(dim=-1))



        average_loss = total_loss / num_batches if num_batches > 0 else 0
        logits_list = torch.concat(logits_list).squeeze().cpu().numpy()
        prob_list = torch.concat(prob_list).squeeze().cpu().numpy()
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)
        overall_metrics, subgroup_metrics = evaluate_binary(prob_list[:, 1], target_list, sensitive_list)
        organized_metrics = organize_results(overall_metrics, subgroup_metrics)
        predicted_list = torch.concat(predicted_list).squeeze().cpu().numpy().astype(int)

        tensor_target_list = torch.tensor(target_list)
        tensor_predicted_list = torch.tensor(predicted_list)

        clas_accuracy = (tensor_target_list == tensor_predicted_list).float().mean().item() * 100
        print(f"Class label Accuracy by tensor: {clas_accuracy:.2f}%")
        print(f'Class label Accuracy by confusion matrix: {overall_metrics["acc"]:.2f}%')
        print(f'Class label Accuracy by confusion matrix: {overall_metrics["acc@best_f1"]:.4f}%')
        

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
    
    def confidence_indices(self, target_prob, sensitive_prob, confidence_threshold=0.5):
        # Compute confidence scores
        # Get probabilities of class 0 and class 1
        # target_confidence_class_0 = target_prob[:, 0]
        # target_confidence_class_1 = target_prob[:, 1]

        # # Select indices where either class 0 or class 1 has confidence above the threshold
        # high_confidence_indices_target = np.where(
        #     (target_confidence_class_0 > confidence_threshold) | (target_confidence_class_1 > confidence_threshold)
        # )[0]

        confidence = target_prob.max(axis=1)
       
        high_confidence_indices_target = np.where(confidence > confidence_threshold)[0]

        

        return high_confidence_indices_target
    
    def pseudo_sample_filtering(self, high_confidence_indices_target, pseudo_target_list,pseudo_sensitive_list,target_list,sensitive_list, confidence_threshold):
        
        log = {}

        

        # Extract corresponding samples
        high_confidence_pseudo_targets = pseudo_target_list[high_confidence_indices_target]
        high_confidence_pseudo_sensitive = pseudo_sensitive_list[high_confidence_indices_target]

        n_samples = len(high_confidence_indices_target)
        log["n_samples"] = n_samples
        

        if n_samples == 0:
            log["class_accuracy"] = 0
            log["attribute_accuracy"] = 0
            log["subgroup_accuracy"] = {}
            return log
        else:
            actual_targets = target_list[high_confidence_indices_target]
            actual_sensitive = sensitive_list[high_confidence_indices_target]

            log["class_accuracy"] = accuracy_score(actual_targets, high_confidence_pseudo_targets)*100
            log["attribute_accuracy"] = accuracy_score(actual_sensitive, high_confidence_pseudo_sensitive)*100

            subgroup_correct, subgroup_total = defaultdict(int), defaultdict(int)

            for actual_cls, actual_attr, pseudo_cls, pseudo_attr in zip(
                actual_targets.tolist(), actual_sensitive.tolist(), high_confidence_pseudo_targets.tolist(),
                high_confidence_pseudo_sensitive.tolist()
            ):
                key = (actual_cls, actual_attr)
                subgroup_total[key] += 1
                if actual_cls == pseudo_cls and actual_attr == pseudo_attr:
                    subgroup_correct[key] += 1

            # Compute and print subgroup accuracy
            subgroup_accuracy = {k: (subgroup_correct[k] / subgroup_total[k]) * 100 for k in subgroup_total}
            subgroup_count = {k: f"predicted_{subgroup_correct[k]}_correct{subgroup_total[k]}" for k in subgroup_total}
            log["subgroup_accuracy"] = subgroup_accuracy
            log["subgroup_count"] = subgroup_count
            return log



    
    def pseudo_label_generation(self, dataloader, print_predictions=False, threshold=0.5, confidence_threshold=[0.9,0.8,0.7,0.5]):
        """Generates pseudo labels and evaluates their accuracy against actual labels."""
        
        self.model.eval()
        
        # Initialize lists for storing labels
        target_prob, sensitive_prob = [], []
        pseudo_target_list, pseudo_sensitive_list = [], []
        target_list, sensitive_list = [], []
        
        # Iterate through batches in dataloader
        with torch.no_grad():
            for minibatch in dataloader:
                x = minibatch["img"].to(self.device)
                y = minibatch["label"]
                a = minibatch["sensitive"]
                
                # Forward pass
                cls_logits = self.model(x)  # Class logits
                att_logits = self.model.forward_att(x)  # Attribute logits

                cls_prob = F.softmax(cls_logits, dim=-1)
                att_prob = F.softmax(att_logits, dim=-1)

                target_prob.append(cls_prob)
                sensitive_prob.append(att_prob)

                # Store actual labels
                target_list.extend(y.squeeze().tolist())
                sensitive_list.extend(a.squeeze().tolist())

                # print("Class Logits Sample:", cls_logits[:5])
                # print("Class Logits Sample:", cls_logits[:5].argmax(dim=-1).cpu().tolist()) 
                # print("Attribute Logits Sample:", att_logits[:5]) 
                # print("Attribute Logits Sample:", att_logits[:5].argmax(dim=-1).cpu().tolist())  

                # Get predicted class and attribute labels
                #pseudo_target_list.extend(cls_logits.argmax(dim=-1).cpu().tolist())
                #pseudo_sensitive_list.extend(att_logits.argmax(dim=-1).cpu().tolist())

                

        target_prob = torch.concat(target_prob).squeeze().cpu().numpy()
        sensitive_prob = torch.concat(sensitive_prob).squeeze().cpu().numpy()
        
        pseudo_target_list = (target_prob[:, 1] > threshold).astype(int)
        pseudo_sensitive_list = (sensitive_prob[:, 1] > threshold).astype(int)
        target_list = np.array(target_list)  
        sensitive_list = np.array(sensitive_list)
        # Convert lists to tensors for computation
        # actual_class_labels = torch.tensor(target_list)
        # actual_attribute_labels = torch.tensor(sensitive_list)
        # pseudo_class_labels = torch.tensor(pseudo_target_list)
        # pseudo_attribute_labels = torch.tensor(pseudo_sensitive_list)

        # **Print Predictions & Accuracy Calculation**
        
        overral_log = {}
        if print_predictions:
            self.logger.info(f"Total_samples: {len(target_list)}")
            for confidence in confidence_threshold:
                high_confidence_indices_target= self.confidence_indices(target_prob, sensitive_prob, confidence)
                log = self.pseudo_sample_filtering(high_confidence_indices_target,pseudo_target_list,pseudo_sensitive_list,target_list,sensitive_list, confidence)
                overral_log[confidence] = log
                self.logger.info(f"Confidence Threshold: {confidence}")
                self.logger.info(f"Number of samples: {log['n_samples']}")
                self.logger.info(f"Class label Accuracy: {log['class_accuracy']:.2f}%")
                self.logger.info(f"Attribute label Accuracy: {log['attribute_accuracy']:.2f}%")
                self.logger.info("\n✅ **Subgroup Accuracy:**")
                for (cls, attr), acc in sorted(log["subgroup_accuracy"].items()):
                    self.logger.info(f"Class {cls}, Attribute {attr} Accuracy: {acc:.2f}% {log['subgroup_count'][(cls, attr)]}")
                self.logger.info("\n")



            

            # clas_accuracy = (actual_class_labels == pseudo_class_labels).float().mean().item() * 100
            # self.logger.info(f"Pseudo Class label Accuracy: {clas_accuracy:.2f}%")
            # cm = confusion_matrix(target_list, pseudo_target_list)
            # header = "      " + "  ".join([f"Pred {i}" for i in range(cm.shape[1])])
            # rows = [f"True {i}  " + "   ".join(f"{num:5d}" for num in row) for i, row in enumerate(cm)]
            # formatted_cm = "\n".join([header] + rows)# Log the formatted confusion matrix
            # self.logger.info("\nConfusion Matrix:\n" + formatted_cm)

            # attr_accuracy = (actual_attribute_labels == pseudo_attribute_labels).float().mean().item() * 100
            # self.logger.info(f"Pseudo Attribute label Accuracy: {attr_accuracy:.2f}%")
            # cm2 = confusion_matrix(sensitive_list, pseudo_sensitive_list)
            # header = "      " + "  ".join([f"Pred {i}" for i in range(cm2.shape[1])])
            # rows = [f"True {i}  " + "   ".join(f"{num:5d}" for num in row) for i, row in enumerate(cm2)]
            # formatted_cm2 = "\n".join([header] + rows)# Log the formatted confusion matrix
            # self.logger.info("\nConfusion Matrix:\n" + formatted_cm2)
            
            # # Count occurrences in (class, attribute) pairs
            # actual_counts = Counter(map(tuple, zip(target_list, sensitive_list)))
            # pseudo_counts = Counter(map(tuple, zip(pseudo_target_list, pseudo_sensitive_list)))
            # validation_counts = Counter(map(tuple, zip(pseudo_target_list, sensitive_list)))

            # self.logger.info("\n🔹 Actual Labels Distribution:")
            # for (cls, attr), count in actual_counts.items():
            #     self.logger.info(f"Class {cls}, Attribute {attr}: {count} samples")
            
            # self.logger.info("\n🔹 Pseudo Labels Distribution:")
            # for (cls, attr), count in pseudo_counts.items():
            #     self.logger.info(f"Class {cls}, Attribute {attr}: {count} samples")
            
            # self.logger.info("\n🔹 Pseudo class label Actual atrribute label Distribution:")
            # for (cls, attr), count in validation_counts.items():
            #     self.logger.info(f"Class {cls}, Attribute {attr}: {count} samples")

            # # Compute per-group accuracy
            # subgroup_correct, subgroup_total = defaultdict(int), defaultdict(int)

            # for actual_cls, actual_attr, pseudo_cls, pseudo_attr in zip(
            #     actual_class_labels.tolist(), actual_attribute_labels.tolist(),
            #     pseudo_class_labels.tolist(), pseudo_attribute_labels.tolist()
            # ):
            #     key = (actual_cls, actual_attr)
            #     subgroup_total[key] += 1
            #     if actual_cls == pseudo_cls and actual_attr == pseudo_attr:
            #         subgroup_correct[key] += 1

            # # Compute and print subgroup accuracy
            # subgroup_accuracy = {k: (subgroup_correct[k] / subgroup_total[k]) * 100 for k in subgroup_total}

            # self.logger.info("\n✅ **Subgroup Accuracy:**")
            # for (cls, attr), acc in sorted(subgroup_accuracy.items()):
            #     self.logger.info(f"Class {cls}, Attribute {attr} Accuracy: {acc:.2f}%")
    
    def best_n_pseudo_label_per_group_generation(self, dataloader, n_samples=[20,30,40,50]):
        """Generates pseudo labels and evaluates their accuracy against actual labels."""
        
        self.model.eval()
        
        # Initialize lists for storing labels
        target_prob = []
        sensitive_prob = []
        target_list = []
        sensitive_list = []
        pred_target_list = []
        pred_sensitive_list = []
        
        # Iterate through batches in dataloader
        with torch.no_grad():
            for minibatch in dataloader:
                x = minibatch["img"].to(self.device)
                y = minibatch["label"]
                a = minibatch["sensitive"]
                
                # Forward pass
                cls_logits = self.model(x)  # Class logits
                att_logits = self.model.forward_att(x)  # Attribute logits

                cls_prob = F.softmax(cls_logits, dim=-1)  # Convert logits to probabilities
                att_prob = F.softmax(att_logits, dim=-1)

                # Get predicted labels
                pred_target = torch.argmax(cls_prob, dim=-1)
                pred_sensitive = torch.argmax(att_prob, dim=-1)

                target_prob.append(cls_prob)
                sensitive_prob.append(att_prob)

                # Store actual and predicted labels
                target_list.extend(y.squeeze().tolist())
                sensitive_list.extend(a.squeeze().tolist())
                pred_target_list.extend(pred_target.squeeze().tolist())
                pred_sensitive_list.extend(pred_sensitive.squeeze().tolist())

        # Convert lists to tensors
        target_prob = torch.cat(target_prob, dim=0)  # [N, num_classes]
        sensitive_prob = torch.cat(sensitive_prob, dim=0)  # [N, num_sensitive_attrs]
        target_list = torch.tensor(target_list)
        sensitive_list = torch.tensor(sensitive_list)
        pred_target_list = torch.tensor(pred_target_list)
        pred_sensitive_list = torch.tensor(pred_sensitive_list)

        # Dictionary to store top-n confident samples per (class, sensitive) group
        best_samples = defaultdict(list)

        # Iterate over all samples
        for i in range(len(target_list)):
            class_label = int(target_list[i].item())
            sensitive_label = int(sensitive_list[i].item())

            # Extract confidence for the correct class and sensitive attribute
            class_confidence = target_prob[i][class_label].item()
            sensitive_confidence = sensitive_prob[i][sensitive_label].item()

            # Compute overall confidence as the product of both probabilities
            overall_confidence = class_confidence * sensitive_confidence

            group_key = (class_label, sensitive_label)  # (y, a) as key

            # Append sample index and its confidence to the group
            best_samples[group_key].append((i, overall_confidence))
        
        for n in n_samples:
            top_n_samples_per_group = {}

            for group, sample_list in best_samples.items():
                # Sort samples by confidence (highest first)
                sorted_samples = sorted(sample_list, key=lambda x: x[1], reverse=True)

                # Keep only the top 20
                top_n_samples_per_group[group] = sorted_samples[:n]

            # Compute accuracy metrics
            total_correct_class = 0
            total_correct_spurious = 0
            total_samples = 0

            group_accuracies = {}

            for group, samples in top_n_samples_per_group.items():
                correct_class_preds = 0
                correct_spurious_preds = 0
                correct_group_preds = 0
                total_samples += len(samples)  # Count total top-20 samples for all groups

                for idx, _ in samples:
                    correct_class = pred_target_list[idx] == target_list[idx]
                    correct_spurious = pred_sensitive_list[idx] == sensitive_list[idx]

                    if correct_class:
                        correct_class_preds += 1
                    if correct_spurious:
                        correct_spurious_preds += 1
                    if correct_class and correct_spurious:
                        correct_group_preds += 1

                # Update overall counts
                total_correct_class += correct_class_preds
                total_correct_spurious += correct_spurious_preds

                # Store accuracy per group
                group_accuracies[group] = {
                    "Group Accuracy": correct_group_preds / len(samples) if len(samples) > 0 else 0,
                    "Correctly Predicted Samples": correct_group_preds,
                }

            # Compute overall class and spurious accuracy
            overall_class_accuracy = total_correct_class / total_samples if total_samples > 0 else 0
            overall_spurious_accuracy = total_correct_spurious / total_samples if total_samples > 0 else 0

            # Print overall accuracies
            self.logger.info(f"Top-{n} Samples Per Group #########")
            self.logger.info(f"\nOverall Class Accuracy: {overall_class_accuracy:.2%}")
            self.logger.info(f"\nOverall Spurious Accuracy: {overall_spurious_accuracy:.2%}")

            # Print each group's accuracy
            for group, metrics in group_accuracies.items():
                self.logger.info(f"\nGroup {group}:")
                self.logger.info(f"  - Group Accuracy: {metrics['Group Accuracy']:.2%}")
                self.logger.info(f"  - Correctly Predicted Samples: {metrics['Correctly Predicted Samples']}/{n}")
            self.logger.info("\n")

    def best_accuracies_after_tuning(self, dataloader):
        """Generates pseudo labels and evaluates their accuracy against actual labels."""
        
        self.model.eval()
        
        # Initialize lists for storing labels
        target_prob = []
        sensitive_prob = []
        target_list = []
        sensitive_list = []
        pred_target_list = []
        pred_sensitive_list = []
        
        # Iterate through batches in dataloader
        with torch.no_grad():
            for minibatch in dataloader:
                x = minibatch["img"].to(self.device)
                y = minibatch["label"]
                a = minibatch["sensitive"]
                
                # Forward pass
                cls_logits = self.model(x)  # Class logits
                att_logits = self.model.forward_att(x)  # Attribute logits

                cls_prob = F.softmax(cls_logits, dim=-1)  # Convert logits to probabilities
                att_prob = F.softmax(att_logits, dim=-1)

               
                target_prob.append(cls_prob)
                sensitive_prob.append(att_prob)

                # Store actual and predicted labels
                target_list.append(y)
                sensitive_list.append(a)
        
        target_prob = torch.concat(target_prob).squeeze().cpu().numpy() # [N, num_classes]
        sensitive_prob = torch.concat(sensitive_prob).squeeze().cpu().numpy()  # [N, num_sensitive_attrs]
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)
        # target_list = torch.concat(target_list)
        # sensitive_list = torch.concat(sensitive_list)

        #target_threshold = find_threshold(target_prob[:,1], target_list) 
        #sensitive_threshold = find_threshold(sensitive_prob[:,1], sensitive_list)

        #target_threshold = find_best_threshold_youden(target_prob[:,1], target_list) 
        #sensitive_threshold = find_best_threshold_youden(sensitive_prob[:,1], sensitive_list)

        target_threshold = 0.5
        sensitive_threshold = 0.5

        
        pseudo_target_list = (target_prob[:, 1] > target_threshold).astype(int)
        pseudo_sensitive_list = (sensitive_prob[:, 1] > sensitive_threshold).astype(int)
        
        target_binary_classification = binary_classification_report(target_prob[:,1], target_list, threshold=target_threshold)
        sensitive_binary_classification = binary_classification_report(sensitive_prob[:,1], sensitive_list, threshold=sensitive_threshold)

        self.logger.info(f"Target Accuracy: {target_binary_classification['acc']:.2f}%")
        self.logger.info(f"Sensitive Accuracy: {sensitive_binary_classification['acc']:.2f}%")
        self.logger.info("\n")
        self.logger.info(f"Target AUC: {target_binary_classification['auc']:.2f}")
        self.logger.info(f"Sensitive AUC: {sensitive_binary_classification['auc']:.2f}")
        self.logger.info("\n")
        self.logger.info(f"Target ECE: {target_binary_classification['ece']:.2f}")
        self.logger.info(f"Sensitive ECE: {sensitive_binary_classification['ece']:.2f}")
        self.logger.info("\n")

         # Compute per-group accuracy
        subgroup_correct, subgroup_total = defaultdict(int), defaultdict(int)

        for actual_cls, actual_attr, pseudo_cls, pseudo_attr in zip(
            target_list, sensitive_list,
            pseudo_target_list, pseudo_sensitive_list
        ):
            key = (actual_cls, actual_attr)
            subgroup_total[key] += 1
            if actual_cls == pseudo_cls and actual_attr == pseudo_attr:
                subgroup_correct[key] += 1

        # Compute and print subgroup accuracy
        subgroup_accuracy = {k: (subgroup_correct[k] / subgroup_total[k]) * 100 for k in subgroup_total}

        self.logger.info("\n✅ **Subgroup Accuracy:**")
        for (cls, attr), acc in sorted(subgroup_accuracy.items()):
            self.logger.info(f"Class {cls}, Attribute {attr} Accuracy: {acc:.2f}%")

    def oracle_n_pseudo_label_per_group_generation(self, dataloader, n_samples=[20,30,40,50]):
        """Generates pseudo labels and evaluates their accuracy against actual labels."""
        
        self.model.eval()
        
        # Initialize lists for storing labels
        target_prob = []
        sensitive_prob = []
        target_list = []
        sensitive_list = []
        pred_target_list = []
        pred_sensitive_list = []
        
        # Iterate through batches in dataloader
        with torch.no_grad():
            for minibatch in dataloader:
                x = minibatch["img"].to(self.device)
                y = minibatch["label"]
                a = minibatch["sensitive"]
                
                # Forward pass
                cls_logits = self.model(x)  # Class logits
                att_logits = self.model.forward_att(x)  # Attribute logits

                cls_prob = F.softmax(cls_logits, dim=-1)  # Convert logits to probabilities
                att_prob = F.softmax(att_logits, dim=-1)

                # Get predicted labels
                

                target_prob.append(cls_prob)
                sensitive_prob.append(att_prob)

                # Store actual and predicted labels
                target_list.append(y)
                sensitive_list.append(a)
                
        # Convert lists to tensors
        target_prob = torch.concat(target_prob).squeeze().cpu().numpy() # [N, num_classes]
        sensitive_prob = torch.concat(sensitive_prob).squeeze().cpu().numpy()   # [N, num_sensitive_attrs]
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)

        target_threshold = 0.5
        sensitive_threshold = 0.5

        pred_target_list = (target_prob[:, 1] > target_threshold).astype(int)
        pred_sensitive_list = (sensitive_prob[:, 1] > sensitive_threshold).astype(int)

        #target_threshold = find_best_threshold_youden(target_prob[:,1], target_list) 
        #sensitive_threshold = find_best_threshold_youden(sensitive_prob[:,1], sensitive_list)

        

        # Dictionary to store top-n confident samples per (class, sensitive) group
        best_samples = defaultdict(list)

        # Iterate over all samples
        for i in range(len(target_list)):
            class_label = int(target_list[i].item())
            sensitive_label = int(sensitive_list[i].item())

            # Extract confidence for the correct class and sensitive attribute
            class_confidence = max(target_prob[i])
            sensitive_confidence = max(sensitive_prob[i])

            # Compute overall confidence as the product of both probabilities
            overall_confidence = class_confidence * sensitive_confidence
            #overall_confidence = class_confidence

            group_key = (class_label, sensitive_label)  # (y, a) as key

            # Append sample index and its confidence to the group
            best_samples[group_key].append((i, overall_confidence))
        
        for n in n_samples:
            top_n_samples_per_group = {}
            indexes = []

            for group, sample_list in best_samples.items():
                # Sort samples by confidence (highest first)
                sorted_samples = sorted(sample_list, key=lambda x: x[1], reverse=True)

                # Keep only the top 20
                top_n_samples_per_group[group] = sorted_samples[:n]
                indexes.extend([idx for idx, _ in sorted_samples[:n]])

            # Compute accuracy metrics
            total_correct_class = 0
            total_correct_spurious = 0
            total_samples = 0

            group_accuracies = {}

            

            target_report = binary_classification_report(target_prob[indexes,1], target_list[indexes], threshold=target_threshold)
            sensitive_report = binary_classification_report(sensitive_prob[indexes,1], sensitive_list[indexes], threshold=sensitive_threshold)

            self.logger.info(f"Top-{n} Samples Per Group #########")
            
            self.logger.info(f"Target Accuracy from Report: {target_report['acc']:.2f}%")
            self.logger.info(f"Sensitive Accuracy from Report: {sensitive_report['acc']:.2f}%")
            self.logger.info("\n")

            self.logger.info(f"Target AUC from Report: {target_report['auc']:.2f}")
            self.logger.info(f"Sensitive AUC from Report: {sensitive_report['auc']:.2f}")
            self.logger.info("\n")

            self.logger.info(f"Target ECE from Report: {target_report['ece']:.2f}")
            self.logger.info(f"Sensitive ECE from Report: {sensitive_report['ece']:.2f}")
            self.logger.info("\n")

            

            for group, samples in top_n_samples_per_group.items():
                correct_class_preds = 0
                correct_spurious_preds = 0
                correct_group_preds = 0
                total_samples += len(samples)  # Count total top-20 samples for all groups

                for idx, _ in samples:
                    correct_class = pred_target_list[idx] == target_list[idx]
                    correct_spurious = pred_sensitive_list[idx] == sensitive_list[idx]

                    if correct_class:
                        correct_class_preds += 1
                    if correct_spurious:
                        correct_spurious_preds += 1
                    if correct_class and correct_spurious:
                        correct_group_preds += 1

                # Update overall counts
                total_correct_class += correct_class_preds
                total_correct_spurious += correct_spurious_preds

                # Store accuracy per group
                group_accuracies[group] = {
                    "Group Accuracy": correct_group_preds / len(samples) if len(samples) > 0 else 0,
                    "Correctly Predicted Samples": correct_group_preds,
                }

            # Compute overall class and spurious accuracy
            overall_class_accuracy = total_correct_class / total_samples if total_samples > 0 else 0
            overall_spurious_accuracy = total_correct_spurious / total_samples if total_samples > 0 else 0

            # Print overall accuracies
           
            self.logger.info(f"\nOverall Class Accuracy: {overall_class_accuracy:.2%}")
            self.logger.info(f"\nOverall Spurious Accuracy: {overall_spurious_accuracy:.2%}")

            # Print each group's accuracy
            for group, metrics in group_accuracies.items():
                self.logger.info(f"\nGroup {group}:")
                self.logger.info(f"  - Group Accuracy: {metrics['Group Accuracy']:.2%}")
                self.logger.info(f"  - Correctly Predicted Samples: {metrics['Correctly Predicted Samples']}/{n}")


    def n_pseudo_label_per_group_generation(self, dataloader, n_samples=[20,40]):
        """Generates pseudo labels and evaluates their accuracy against actual labels."""
        
        self.model.eval()
        
        # Initialize lists for storing labels
        target_prob = []
        sensitive_prob = []
        target_list = []
        sensitive_list = []
        pred_target_list = []
        pred_sensitive_list = []
        
        # Iterate through batches in dataloader
        with torch.no_grad():
            for minibatch in dataloader:
                x = minibatch["img"].to(self.device)
                y = minibatch["label"]
                a = minibatch["sensitive"]
                
                # Forward pass
                cls_logits = self.model(x)  # Class logits
                att_logits = self.model.forward_att(x)  # Attribute logits

                cls_prob = F.softmax(cls_logits, dim=-1)  # Convert logits to probabilities
                att_prob = F.softmax(att_logits, dim=-1)

                # Get predicted labels
                

                target_prob.append(cls_prob)
                sensitive_prob.append(att_prob)

                # Store actual and predicted labels
                target_list.append(y)
                sensitive_list.append(a)
                
        # Convert lists to tensors
        target_prob = torch.concat(target_prob).squeeze().cpu().numpy() # [N, num_classes]
        sensitive_prob = torch.concat(sensitive_prob).squeeze().cpu().numpy()   # [N, num_sensitive_attrs]
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)

        target_threshold = 0.5
        sensitive_threshold = 0.5

        pred_target_list = (target_prob[:, 1] > target_threshold).astype(int)
        pred_sensitive_list = (sensitive_prob[:, 1] > sensitive_threshold).astype(int)

        #target_threshold = find_best_threshold_youden(target_prob[:,1], target_list) 
        #sensitive_threshold = find_best_threshold_youden(sensitive_prob[:,1], sensitive_list)

        

        # Dictionary to store top-n confident samples per (class, sensitive) group
        best_samples = defaultdict(list)

        # Iterate over all samples
        for i in range(len(pred_target_list)):
            class_label = int(pred_target_list[i].item())
            sensitive_label = int(pred_sensitive_list[i].item())

            # Extract confidence for the correct class and sensitive attribute
            class_confidence = max(target_prob[i])
            sensitive_confidence = max(sensitive_prob[i])

            # Compute overall confidence as the product of both probabilities
            overall_confidence = class_confidence * sensitive_confidence
            #overall_confidence = class_confidence

            group_key = (class_label, sensitive_label)  # (y, a) as key

            # Append sample index and its confidence to the group
            best_samples[group_key].append((i, overall_confidence))
        
        for n in n_samples:
            top_n_samples_per_group = {}
            indexes = []

            for group, sample_list in best_samples.items():
                # Sort samples by confidence (highest first)
                sorted_samples = sorted(sample_list, key=lambda x: x[1], reverse=True)

                # Keep only the top 20
                top_n_samples_per_group[group] = sorted_samples[:n]
                indexes.extend([idx for idx, _ in sorted_samples[:n]])

            # Compute accuracy metrics
            total_correct_class = 0
            total_correct_spurious = 0
            total_samples = 0

            group_accuracies = {}

            

            target_report = binary_classification_report(target_prob[indexes,1], target_list[indexes], threshold=target_threshold)
            sensitive_report = binary_classification_report(sensitive_prob[indexes,1], sensitive_list[indexes], threshold=sensitive_threshold)

            self.logger.info(f"Top-{n} Samples Per Group #########")
            
            self.logger.info(f"Target Accuracy from Report: {target_report['acc']:.2f}%")
            self.logger.info(f"Sensitive Accuracy from Report: {sensitive_report['acc']:.2f}%")
            self.logger.info("\n")

            self.logger.info(f"Target AUC from Report: {target_report['auc']:.2f}")
            self.logger.info(f"Sensitive AUC from Report: {sensitive_report['auc']:.2f}")
            self.logger.info("\n")

            self.logger.info(f"Target ECE from Report: {target_report['ece']:.2f}")
            self.logger.info(f"Sensitive ECE from Report: {sensitive_report['ece']:.2f}")
            self.logger.info("\n")

            

            for group, samples in top_n_samples_per_group.items():
                correct_class_preds = 0
                correct_spurious_preds = 0
                correct_group_preds = 0
                total_samples += len(samples)  # Count total top-20 samples for all groups

                for idx, _ in samples:
                    correct_class = pred_target_list[idx] == target_list[idx]
                    correct_spurious = pred_sensitive_list[idx] == sensitive_list[idx]

                    if correct_class:
                        correct_class_preds += 1
                    if correct_spurious:
                        correct_spurious_preds += 1
                    if correct_class and correct_spurious:
                        correct_group_preds += 1

                # Update overall counts
                total_correct_class += correct_class_preds
                total_correct_spurious += correct_spurious_preds

                # Store accuracy per group
                group_accuracies[group] = {
                    "Group Accuracy": correct_group_preds / len(samples) if len(samples) > 0 else 0,
                    "Correctly Predicted Samples": correct_group_preds,
                }

            # Compute overall class and spurious accuracy
            overall_class_accuracy = total_correct_class / total_samples if total_samples > 0 else 0
            overall_spurious_accuracy = total_correct_spurious / total_samples if total_samples > 0 else 0

            # Print overall accuracies
           
            self.logger.info(f"\nOverall Class Accuracy: {overall_class_accuracy:.2%}")
            self.logger.info(f"\nOverall Spurious Accuracy: {overall_spurious_accuracy:.2%}")

            # Print each group's accuracy
            for group, metrics in group_accuracies.items():
                self.logger.info(f"\nGroup {group}:")
                self.logger.info(f"  - Group Accuracy: {metrics['Group Accuracy']:.2%}")
                self.logger.info(f"  - Correctly Predicted Samples: {metrics['Correctly Predicted Samples']}/{n}")


    
    
