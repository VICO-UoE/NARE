# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from torch.autograd.function import Function
from torch.autograd import Variable
import os
import os.path as osp
import pickle


from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
# from maskrcnn_benchmark.layers.sparsemax_loss import SparsemaxLoss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        use_pme_softmax,
        use_bce_temp,
        num_classes,
        two_heads,
        self_train_loss,
        manifold_mixup,
        imp_manifold_mixup,
        soft_labels,
        cfg_device,
        predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.use_pme_softmax = use_pme_softmax
        self.use_bce_temp = use_bce_temp
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.num_classes = num_classes
        self.device = torch.device(cfg_device)
        self.two_heads = two_heads
        self.self_train_loss = self_train_loss
        self.manifold_mixup = manifold_mixup
        self.imp_manifold_mixup = imp_manifold_mixup
        self.soft_labels = soft_labels
        self.reverse_labelmatrix_array, _,_, _,_ =  self.generate_predicate_matrix_annotations(self.num_classes)
        
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            if not self.two_heads:
                if self.use_pme_softmax and not self.use_bce_temp:
                    self.rel_criterion_loss = PMESoftmax.apply
                elif self.use_bce_temp and not self.use_pme_softmax:
                    # self.bce_loss = nn.BCEWithLogitsLoss()
                    # self.rel_criterion_loss = BCELosswithTemp(self.device,self.bce_loss)

                    self.rel_criterion_loss = SparsemaxLoss()

                elif self.use_bce_temp and self.use_pme_softmax:
                    self.bce_loss = nn.BCEWithLogitsLoss()
                    self.rel_pmes_criterion_loss = PMESoftmax.apply
                    self.rel_bce_criterion_loss = SparsemaxLoss()
                    # self.rel_bce_criterion_loss = BCELosswithTemp(self.device,self.bce_loss)
                else:
                    if self.self_train_loss == 'multi_ce' or 'kl' or 'jsd':

                        self.rel_criterion_loss = nn.CrossEntropyLoss(reduction='none') #PMESoftmax.apply #nn.CrossEntropyLoss(reduction='none')
                    else:
                        self.rel_criterion_loss = nn.BCEWithLogitsLoss(reduction='none')
                    if self.self_train_loss == 'kl':
                        self.rel_bce_criterion_loss = torch.nn.KLDivLoss().to(self.device)
                    
            else:
                if self.use_bce_temp and self.use_pme_softmax:
                    self.bce_loss = nn.BCEWithLogitsLoss()
                    self.rel_pmes_criterion_loss = PMESoftmax.apply
                    # self.rel_bce_criterion_loss = SparsemaxLoss()
                    self.rel_bce_criterion_loss = torch.nn.KLDivLoss().to(self.device)

        self.criterion_loss = nn.CrossEntropyLoss()
    def get_one_hot_labels(self, pred_labels):
        batch = pred_labels.size()[0]
        vocab_size = 51

        one_hot_vector = torch.zeros(batch,vocab_size).to(self.device)
        for j, lbl in enumerate(pred_labels):
            one_hot_vector[j,lbl] = 1.

        one_hot_vector = Variable(one_hot_vector.float(), requires_grad=False)
        return one_hot_vector


    def __call__(self, proposals, rel_labels, relation_logits, mixed_relation_logits, lam, idx, loss_samples,refine_logits,multi_relation_logits=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        mixed_relation_logits = cat(mixed_relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        # print(multi_relation_logits)
        # input('enter')
        if not self.two_heads:
            
            if self.use_pme_softmax and not self.use_bce_temp:
                one_hot_rel_labels = self.get_one_hot_labels(rel_labels.long())
                reverse_labelmatrix_array, _,_, _,_ =  self.generate_predicate_matrix_annotations(self.num_classes)
                loss_relation = self.rel_criterion_loss(relation_logits, one_hot_rel_labels,reverse_labelmatrix_array.clone(),rel_labels.long())
            
            elif self.use_bce_temp and not self.use_pme_softmax:
                reverse_labelmatrix_array, _,implied_labelmatrix_array,_, prior_temp =  self.generate_predicate_matrix_annotations(self.num_classes)
                loss_relation = self.rel_criterion_loss(relation_logits, implied_labelmatrix_array[rel_labels.long()].float())
                # loss_relation = self.rel_criterion_loss(relation_logits,implied_labelmatrix_array, rel_labels.long(),prior_temp)
            elif self.use_bce_temp and self.use_pme_softmax:
                reverse_labelmatrix_array, _,implied_labelmatrix_array, _,prior_temp =  self.generate_predicate_matrix_annotations(self.num_classes)
                one_hot_rel_labels = self.get_one_hot_labels(rel_labels.long())
                # Translate input by max for numerical stability (otheriwse NAN losses)
                relation_logits = relation_logits - torch.max(relation_logits, dim=1, keepdim=True)[0].expand_as(relation_logits)
                # loss_relation = self.rel_pmes_criterion_loss(relation_logits, one_hot_rel_labels,reverse_labelmatrix_array.clone(),rel_labels.long()) + self.rel_bce_criterion_loss(relation_logits, implied_labelmatrix_array,rel_labels.long(),prior_temp)
                loss_relation = self.rel_pmes_criterion_loss(relation_logits, one_hot_rel_labels,reverse_labelmatrix_array.clone(),rel_labels.long()) + 0.5*self.rel_bce_criterion_loss(relation_logits, implied_labelmatrix_array[rel_labels.long()].float())
                # print(loss_relation)
            else:
                gt_implicit = [0,5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
                gt_explicit = [1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
                mask = torch.tensor([1. if i in gt_implicit else 0. for i in rel_labels.long()]).to(self.device)


           
                if self.imp_manifold_mixup:
                        one_hot_rel_labels = self.get_one_hot_labels(rel_labels.long())  
                        one_hot_refined_labels_a, one_hot_refined_labels_b = one_hot_rel_labels, one_hot_rel_labels[idx]
                        mixed_one_hot_rel_labels = lam * one_hot_refined_labels_a + (1 - lam) * one_hot_refined_labels_b 
                        loss_relation = -torch.mean(torch.sum(F.log_softmax(mixed_relation_logits, dim=1) * mixed_one_hot_rel_labels, dim=1)*mask)

                else:
                    loss_relation = (self.rel_criterion_loss(relation_logits, rel_labels.long())*mask).mean()
                    # loss_relation = (self.rel_criterion_loss(relation_logits, rel_labels.long())*(1. - mask)).mean()

                if self.self_train_loss == 'kl':
                    
                    if self.manifold_mixup:
                        if self.soft_labels:
                            relation_logits_exp, one_hot_exp_labels,pseudo_targets_exp = self.soft_pseudo_labels_for_train(lam, idx, relation_logits, mixed_relation_logits, rel_labels.long())
                        else:

                            relation_logits_exp, relation_logits_exp_raw, one_hot_exp_labels,one_hot_gt_labels_exp,pseudo_targets_exp , mask_loss_samples_pseudo= self.pseudo_labels_for_train_proposals(mixed_relation_logits,relation_logits,lam,idx,loss_samples, rel_labels.long())
            

                            one_hot_exp_labels = one_hot_exp_labels / one_hot_exp_labels.sum(dim=1, keepdim=True)
                            one_hot_exp_labels = torch.clamp(one_hot_exp_labels, min=1e-4, max=1.0)

                    
                    else:
                        relation_logits_exp, one_hot_exp_labels,pseudo_targets_exp = self.pseudo_labels_for_train_proposals(relation_logits,relation_logits,lam,idx, rel_labels.long())
                     
                        
                    if len(relation_logits_exp) > 0:
                      
                        relation_logits_exp = relation_logits_exp - torch.max(relation_logits_exp, dim=1, keepdim=True)[0].expand_as(relation_logits_exp)
                        # print(self.rel_bce_criterion_loss(F.log_softmax(relation_logits_exp), (one_hot_exp_labels).float()))
                        loss_relation+= (self.rel_bce_criterion_loss(F.log_softmax(relation_logits_exp), (one_hot_exp_labels).float()))
                        # loss_relation+= (self.rel_bce_criterion_loss(F.log_softmax(relation_logits_exp), (one_hot_exp_labels).float()).mean(1)).mean()

                        
                elif self.self_train_loss == 'multi_ce':
                    if self.manifold_mixup:
                        relation_logits_exp, one_hot_exp_labels,pseudo_targets_exp = self.pseudo_labels_for_train_proposals(mixed_relation_logits, rel_labels.long())
                    else:
                        relation_logits_exp, one_hot_exp_labels,pseudo_targets_exp = self.pseudo_labels_for_train_proposals(relation_logits, rel_labels.long())
                 
                    if len(relation_logits_exp) > 0:
                        
                        loss_relation+= self.rel_bce_criterion_loss(relation_logits_exp, (pseudo_targets_exp))
                elif self.self_train_loss == 'jsd':
                    relation_logits_exp, one_hot_exp_labels = self.pseudo_labels_for_train_proposals(relation_logits, rel_labels.long())
                    one_hot_exp_labels = one_hot_exp_labels/2.
                    one_hot_exp_labels = torch.clamp(one_hot_exp_labels, min=1e-4, max=1.0)
                    if len(relation_logits_exp) > 0:
                       
                        relation_logits_exp = relation_logits_exp - torch.max(relation_logits_exp, dim=1, keepdim=True)[0].expand_as(relation_logits_exp)
                        loss_relation+= self.rel_bce_criterion_loss(F.softmax(relation_logits_exp,-1), (one_hot_exp_labels).float())
                elif self.self_train_loss == 'lsep':
                    
                    if self.manifold_mixup:
                        relation_logits_exp, one_hot_exp_labels,_ = self.pseudo_labels_for_train_proposals(mixed_relation_logits, rel_labels.long())
                    else:
                        relation_logits_exp, one_hot_exp_labels,_ = self.pseudo_labels_for_train_proposals(relation_logits, rel_labels.long())
                    if len(relation_logits_exp) > 0:
                        # print(one_hot_exp_labels[10:30])
                        # input('enter')
                        relation_logits_exp = relation_logits_exp - torch.max(relation_logits_exp, dim=1, keepdim=True)[0].expand_as(relation_logits_exp)
                        loss_relation+= self.rel_bce_criterion_loss(torch.sigmoid(relation_logits_exp), (one_hot_exp_labels).float())
                        print(self.rel_bce_criterion_loss(torch.sigmoid(relation_logits_exp), (one_hot_exp_labels).float()))

                elif self.self_train_loss == 'bce':
                    if self.manifold_mixup:
                        relation_logits_exp, one_hot_exp_labels,pseudo_targets_exp = self.pseudo_labels_for_train_proposals(mixed_relation_logits, rel_labels.long())
                    else:
                        relation_logits_exp, one_hot_exp_labels,_ = self.pseudo_labels_for_train_proposals(relation_logits, rel_labels.long())
                    if len(relation_logits_exp) > 0:
                      
                        relation_logits_exp = relation_logits_exp - torch.max(relation_logits_exp, dim=1, keepdim=True)[0].expand_as(relation_logits_exp)
                        loss_relation+= self.rel_bce_criterion_loss(relation_logits_exp, (one_hot_exp_labels).float())
                elif self.self_train_loss == 'none':
                    loss_relation = loss_relation

             
        else:
            multi_relation_logits = cat(multi_relation_logits, dim=0)
            if self.use_bce_temp and self.use_pme_softmax:
                reverse_labelmatrix_array, _,implied_labelmatrix_array,implied_prob_labelmatrix_array, prior_temp =  self.generate_predicate_matrix_annotations(self.num_classes)
                one_hot_rel_labels = self.get_one_hot_labels(rel_labels.long())
              
                loss_relation = self.rel_pmes_criterion_loss(relation_logits, one_hot_rel_labels,reverse_labelmatrix_array.clone(),rel_labels.long()) +\
                    0.5* self.rel_bce_criterion_loss(F.log_softmax(multi_relation_logits), implied_prob_labelmatrix_array[rel_labels.long()].float())


        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj


    def soft_pseudo_labels_for_train(self, lam, idx, relation_logits, mixed_relation_logits, rel_labels):

        rel_labels_nobkg_idx = rel_labels.nonzero().squeeze(1)
        rel_labels_nobkg = rel_labels[rel_labels_nobkg_idx]
        # print(rel_labels_nobkg)
        # print(rel_labels)
        gt_explicit = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        mask_explicit = torch.tensor([1. if i in gt_explicit else 0. for i in rel_labels_nobkg.long()]).to(self.device)

    
        if self.self_train_loss == 'bce':
            relation_logits = torch.sigmoid(relation_logits[:,1:])
        else:
            relation_logits = F.softmax(relation_logits[:,1:], -1)

        bkg_logits = torch.zeros(relation_logits.size()[0]).unsqueeze(1).to(self.device)
        soft_relation_logits = torch.cat((bkg_logits, relation_logits), dim=1)
        soft_relation_logits[:,0] = 10e-4

        one_hot_refined_labels_a, one_hot_refined_labels_b = soft_relation_logits, soft_relation_logits[idx]
           
        mixed_one_hot_refined_labels = lam * one_hot_refined_labels_a + (1 - lam) * one_hot_refined_labels_b  
        mixed_one_hot_refined_labels = mixed_one_hot_refined_labels[rel_labels_nobkg_idx][mask_explicit.nonzero().squeeze(1)]
        relation_logits_exp = mixed_relation_logits[rel_labels_nobkg_idx][mask_explicit.nonzero().squeeze(1)]
       
        return relation_logits_exp, mixed_one_hot_refined_labels,soft_relation_logits

    

    def mix_up_labels_explicit_impute(self, lam, idx,relation_logits, gt_rel_labels, gt_explicit=None):
        
        implied_label_list = {0: [0], 1: [1, 13, 18, 19, 24, 26, 28, 31, 33, 35, 38, 40, 41, 46], 2: [2, 23], 3: [3, 4, 6, 15, 17, 28], 4: [3, 4], 5: [5, 10, 36, 50], 6: [3, 6, 25, 28, 29], 7: [7, 23, 28, 29, 32, 36, 42, 50], 8: [8, 25, 29, 32, 47], 9: [9, 30, 36, 42], 10: [5, 10, 50], 11: [11, 20, 21, 29, 50], 12: [12, 22, 27, 43], 13: [1, 13], 14: [14, 23, 25, 29], 15: [3, 15, 22], 16: [6, 16, 42], 17: [3, 6, 17, 19], 18: [1, 18, 31, 33], 19: [1, 7, 17, 19], 20: [4, 11, 20, 21, 27, 30, 34, 39, 44, 48, 49, 50], 21: [11, 20, 21, 37, 44, 50], 22: [12, 15, 22, 45, 48, 49], 23: [2, 7, 23, 29], 24: [1, 24, 26, 31, 33], 25: [2, 8, 25, 47], 26: [1, 24, 26, 31, 33], 27: [12, 20, 27, 30], 28: [3, 6, 7, 28, 31], 29: [4, 6, 7, 11, 14, 23, 25, 29, 44, 47], 30: [9, 27, 30], 31: [1, 18, 24, 26, 28, 31, 32, 33, 34, 35, 38, 40, 41, 46], 32: [7, 8, 31, 32], 33: [1, 18, 24, 26, 31, 33, 35, 38, 40, 41, 46], 34: [31, 34], 35: [1, 31, 33, 35], 36: [5, 7, 9, 30, 36, 42], 37: [21, 37, 44, 50], 38: [1, 31, 33, 38, 40], 39: [20, 39, 50], 40: [1, 31, 33, 38, 40], 41: [1, 31, 33, 41, 46], 42: [7, 9, 16, 36, 42, 50], 43: [12, 43], 44: [20, 21, 29, 37, 38, 40, 44, 50], 45: [22, 41, 45, 46], 46: [1, 31, 33, 41, 45, 46], 47: [8, 25, 29, 47], 48: [20, 22, 48, 49, 50], 49: [20, 22, 48, 49, 50], 50: [4, 5, 7, 10, 11, 19, 20, 21, 30, 36, 37, 42, 44, 48, 49, 50]}


        # base probs for logit adjustment
        gt_implicit = [5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
        # base probs for logit adjustment
        count = [0, 47341, 1996, 3092, 3624, 3477, 9903, 10190, 41356, 3288, 3411, 5213, 2312, 3806, 4688, 1973, 9145, 2945, 1853, 9894, 277936, 42722, 251756, 13715, 3739, 3083, 1869, 2380, 2253, 96589, 146339, 712409, 1914, 9317, 3095, 2721, 2065, 3810, 8856, 2241, 18643, 14185, 2517, 22596, 1925, 1740, 4613, 3490, 136099, 15457, 66425]
        count = np.array(count)
        middle_tail = [34, 9, 10, 5, 47, 4, 24, 13, 37, 46, 14, 11, 38, 16, 33, 19, 6,45, 18, 26, 32, 44, 15, 2, 36, 39, 28, 12, 27, 42, 35, 17, 25, 3]
        head=[7, 23, 41, 49, 40, 43, 8, 21, 1, 50, 29, 48, 30, 22, 20, 31]
        
        base_implicit_probs = torch.tensor(count / count[middle_tail].sum()).to(self.device)
        base_implicit_probs[head] = 1.
        log_base_implicit_probs = (base_implicit_probs + 1e-12).log().to(self.device)

        base_probs = torch.tensor([0.000, 0.0231, 0.0010, 0.0015, 0.0018, 0.0017, 0.0048, 0.0050, 0.0202, 0.0016,\
        0.0017, 0.0025, 0.0011, 0.0019, 0.0023, 0.0010, 0.0045, 0.0014, 0.0009,\
        0.0048, 0.1358, 0.0209, 0.1230, 0.0067, 0.0018, 0.0015, 0.0009, 0.0012,\
        0.0011, 0.0472, 0.0715, 0.3482, 0.0009, 0.0046, 0.0015, 0.0013, 0.0010,\
        0.0019, 0.0043, 0.0011, 0.0091, 0.0069, 0.0012, 0.0110, 0.0009, 0.0009,\
        0.0023, 0.0017, 0.0665, 0.0076, 0.0325]).to(self.device)
        log_base_probs = (base_probs + 1e-12).log().to(self.device)

        if gt_explicit is None:
            gt_explicit = [1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_explicit_mask = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_implicit_mask = [0, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]

       
        relation_logits_nobkg = relation_logits 
        if self.self_train_loss == 'bce':
            relation_logits_nobkg = torch.sigmoid(relation_logits_nobkg)
        else:
            relation_logits_nobkg = F.softmax(relation_logits_nobkg, -1)
        
        relation_logits_nobkg[:,gt_implicit_mask] = 0. 
        
        probs, pseudo_labels = torch.max(relation_logits_nobkg, 1)

        one_hot_gt_labels =  self.get_one_hot_labels(gt_rel_labels)
        one_hot_pseudo_labels =  self.get_one_hot_labels(pseudo_labels)

        one_hot_refined_labels = []
        mask_explicit = torch.tensor([1. if i in gt_implicit else 0. for i in gt_rel_labels.long()]).to(self.device)
        for i in range(len(gt_rel_labels)):
            if mask_explicit[i]:
              
                one_hot_refined_labels.append(((one_hot_gt_labels[i]+one_hot_pseudo_labels[i])>0.).float())
           
            else:
                one_hot_refined_labels.append(one_hot_gt_labels[i])
        
        one_hot_refined_labels = torch.stack(one_hot_refined_labels)

        one_hot_refined_labels_a, one_hot_refined_labels_b = one_hot_refined_labels, one_hot_refined_labels[idx]
           
        mixed_one_hot_refined_labels = lam * one_hot_refined_labels_a + (1 - lam) * one_hot_refined_labels_b  

        return mixed_one_hot_refined_labels




    def mix_up_labels(self, lam, idx,relation_logits, gt_rel_labels, gt_explicit=None):
        
        implied_label_list = {0: [0], 1: [1, 13, 18, 19, 24, 26, 28, 31, 33, 35, 38, 40, 41, 46], 2: [2, 23], 3: [3, 4, 6, 15, 17, 28], 4: [3, 4], 5: [5, 10, 36, 50], 6: [3, 6, 25, 28, 29], 7: [7, 23, 28, 29, 32, 36, 42, 50], 8: [8, 25, 29, 32, 47], 9: [9, 30, 36, 42], 10: [5, 10, 50], 11: [11, 20, 21, 29, 50], 12: [12, 22, 27, 43], 13: [1, 13], 14: [14, 23, 25, 29], 15: [3, 15, 22], 16: [6, 16, 42], 17: [3, 6, 17, 19], 18: [1, 18, 31, 33], 19: [1, 7, 17, 19], 20: [4, 11, 20, 21, 27, 30, 34, 39, 44, 48, 49, 50], 21: [11, 20, 21, 37, 44, 50], 22: [12, 15, 22, 45, 48, 49], 23: [2, 7, 23, 29], 24: [1, 24, 26, 31, 33], 25: [2, 8, 25, 47], 26: [1, 24, 26, 31, 33], 27: [12, 20, 27, 30], 28: [3, 6, 7, 28, 31], 29: [4, 6, 7, 11, 14, 23, 25, 29, 44, 47], 30: [9, 27, 30], 31: [1, 18, 24, 26, 28, 31, 32, 33, 34, 35, 38, 40, 41, 46], 32: [7, 8, 31, 32], 33: [1, 18, 24, 26, 31, 33, 35, 38, 40, 41, 46], 34: [31, 34], 35: [1, 31, 33, 35], 36: [5, 7, 9, 30, 36, 42], 37: [21, 37, 44, 50], 38: [1, 31, 33, 38, 40], 39: [20, 39, 50], 40: [1, 31, 33, 38, 40], 41: [1, 31, 33, 41, 46], 42: [7, 9, 16, 36, 42, 50], 43: [12, 43], 44: [20, 21, 29, 37, 38, 40, 44, 50], 45: [22, 41, 45, 46], 46: [1, 31, 33, 41, 45, 46], 47: [8, 25, 29, 47], 48: [20, 22, 48, 49, 50], 49: [20, 22, 48, 49, 50], 50: [4, 5, 7, 10, 11, 19, 20, 21, 30, 36, 37, 42, 44, 48, 49, 50]}


        # base probs for logit adjustment
        gt_implicit = [5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
        # base probs for logit adjustment
        count = [0, 47341, 1996, 3092, 3624, 3477, 9903, 10190, 41356, 3288, 3411, 5213, 2312, 3806, 4688, 1973, 9145, 2945, 1853, 9894, 277936, 42722, 251756, 13715, 3739, 3083, 1869, 2380, 2253, 96589, 146339, 712409, 1914, 9317, 3095, 2721, 2065, 3810, 8856, 2241, 18643, 14185, 2517, 22596, 1925, 1740, 4613, 3490, 136099, 15457, 66425]
        count = np.array(count)
        middle_tail = [34, 9, 10, 5, 47, 4, 24, 13, 37, 46, 14, 11, 38, 16, 33, 19, 6,45, 18, 26, 32, 44, 15, 2, 36, 39, 28, 12, 27, 42, 35, 17, 25, 3]
        head=[7, 23, 41, 49, 40, 43, 8, 21, 1, 50, 29, 48, 30, 22, 20, 31]
        
        base_implicit_probs = torch.tensor(count / count[middle_tail].sum()).to(self.device)
        base_implicit_probs[head] = 1.
        log_base_implicit_probs = (base_implicit_probs + 1e-12).log().to(self.device)

        base_probs = torch.tensor([0.000, 0.0231, 0.0010, 0.0015, 0.0018, 0.0017, 0.0048, 0.0050, 0.0202, 0.0016,\
        0.0017, 0.0025, 0.0011, 0.0019, 0.0023, 0.0010, 0.0045, 0.0014, 0.0009,\
        0.0048, 0.1358, 0.0209, 0.1230, 0.0067, 0.0018, 0.0015, 0.0009, 0.0012,\
        0.0011, 0.0472, 0.0715, 0.3482, 0.0009, 0.0046, 0.0015, 0.0013, 0.0010,\
        0.0019, 0.0043, 0.0011, 0.0091, 0.0069, 0.0012, 0.0110, 0.0009, 0.0009,\
        0.0023, 0.0017, 0.0665, 0.0076, 0.0325]).to(self.device)
        log_base_probs = (base_probs + 1e-12).log().to(self.device)

        if gt_explicit is None:
            gt_explicit = [1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_explicit_mask = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]

        relation_logits_nobkg = relation_logits 
        if self.self_train_loss == 'bce':
            relation_logits_nobkg = torch.sigmoid(relation_logits_nobkg)
        else:
            relation_logits_nobkg = F.softmax(relation_logits_nobkg, -1)
        
        relation_logits_nobkg[:,gt_explicit_mask] = 0. 
        
        probs, pseudo_labels = torch.max(relation_logits_nobkg, 1)

        one_hot_gt_labels =  self.get_one_hot_labels(gt_rel_labels)
        one_hot_pseudo_labels =  self.get_one_hot_labels(pseudo_labels)

        one_hot_refined_labels = []
        mask_explicit = torch.tensor([1. if i in gt_explicit else 0. for i in gt_rel_labels.long()]).to(self.device)
        for i in range(len(gt_rel_labels)):
            if mask_explicit[i]:
              
                one_hot_refined_labels.append(((one_hot_gt_labels[i]+one_hot_pseudo_labels[i])>0.).float())
            
            else:
                one_hot_refined_labels.append(one_hot_gt_labels[i])
        
        one_hot_refined_labels = torch.stack(one_hot_refined_labels)

        one_hot_refined_labels_a, one_hot_refined_labels_b = one_hot_refined_labels, one_hot_refined_labels[idx]
           
        mixed_one_hot_refined_labels = lam * one_hot_refined_labels_a + (1 - lam) * one_hot_refined_labels_b  

        return mixed_one_hot_refined_labels

    def pseudo_labels_for_train_proposals(self, relation_logits,relation_logits_raw,lam, idx, loss_samples, rel_labels):
        rel_labels_nobkg_idx = rel_labels.nonzero().squeeze(1)
        rel_labels_nobkg = rel_labels[rel_labels_nobkg_idx]

        idx_to_pred = {'1': 'above', '2': 'across', '3': 'against', '4': 'along', '5': 'and', '6': 'at', '7': 'attached to', '8': 'behind', '9': 'belonging to', '10': 'between', '11': 'carrying', '12': 'covered in', '13': 'covering', '14': 'eating', '15': 'flying in', '16': 'for', '17': 'from', '18': 'growing on', '19': 'hanging from', '20': 'has', '21': 'holding', '22': 'in', '23': 'in front of', '24': 'laying on', '25': 'looking at', '26': 'lying on', '27': 'made of', '28': 'mounted on', '29': 'near', '30': 'of', '31': 'on', '32': 'on back of', '33': 'over', '34': 'painted on', '35': 'parked on', '36': 'part of', '37': 'playing', '38': 'riding', '39': 'says', '40': 'sitting on', '41': 'standing on', '42': 'to', '43': 'under', '44': 'using', '45': 'walking in', '46': 'walking on', '47': 'watching', '48': 'wearing', '49': 'wears', '50': 'with'}
        # print(rel_labels_nobkg)
        # print(rel_labels)
        gt_explicit = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_explicit_new = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_implicit = [5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
        middle_tail = [34, 9, 10, 5, 47, 4, 24, 13, 37, 46, 14, 11, 38, 16, 33, 19, 6,45, 18, 26, 32, 44, 15, 2, 36, 39, 28, 12, 27, 42, 35, 17, 25, 3]
        head=[7, 23, 41, 49, 40, 43, 8, 21, 1, 50, 29, 48, 30, 22, 20, 31]
        mask_explicit = torch.tensor([1. if i in gt_explicit else 0. for i in rel_labels_nobkg.long()]).to(self.device)
        mask_implicit = torch.tensor([1. if i in gt_implicit else 0. for i in rel_labels_nobkg.long()]).to(self.device)
        # base probs for logit adjustment
        count = [0, 47341, 1996, 3092, 3624, 3477, 9903, 10190, 41356, 3288, 3411, 5213, 2312, 3806, 4688, 1973, 9145, 2945, 1853, 9894, 277936, 42722, 251756, 13715, 3739, 3083, 1869, 2380, 2253, 96589, 146339, 712409, 1914, 9317, 3095, 2721, 2065, 3810, 8856, 2241, 18643, 14185, 2517, 22596, 1925, 1740, 4613, 3490, 136099, 15457, 66425]
        count = np.array(count)
        base_implicit_probs = torch.tensor(count / count[middle_tail].sum()).to(self.device)
        base_implicit_probs[head] = 1.
        log_base_implicit_probs = (base_implicit_probs + 1e-12).log().to(self.device)

        base_probs = torch.tensor([0.000, 0.0231, 0.0010, 0.0015, 0.0018, 0.0017, 0.0048, 0.0050, 0.0202, 0.0016,\
        0.0017, 0.0025, 0.0011, 0.0019, 0.0023, 0.0010, 0.0045, 0.0014, 0.0009,\
        0.0048, 0.1358, 0.0209, 0.1230, 0.0067, 0.0018, 0.0015, 0.0009, 0.0012,\
        0.0011, 0.0472, 0.0715, 0.3482, 0.0009, 0.0046, 0.0015, 0.0013, 0.0010,\
        0.0019, 0.0043, 0.0011, 0.0091, 0.0069, 0.0012, 0.0110, 0.0009, 0.0009,\
        0.0023, 0.0017, 0.0665, 0.0076, 0.0325]).to(self.device)

        log_base_probs = (base_probs + 1e-12).log().to(self.device)

        mask_tensor = torch.ones(51)
        mask_tensor[0] = 0.
       

        ###### base probs over implicit labels and softmax over implicit labels
        relation_logits_nobkg = relation_logits_raw[rel_labels_nobkg_idx]
        relation_logits_nobkg = relation_logits_nobkg - log_base_implicit_probs
        mask_tensor = torch.zeros(51)
        mask_tensor[gt_implicit] = 1.
        mask_tensor = mask_tensor.repeat(len(relation_logits_nobkg), 1).to(self.device)
        relation_logits_nobkg = torch.exp(relation_logits_nobkg) / (torch.exp(relation_logits_nobkg) * mask_tensor).sum(1).unsqueeze(1).expand_as(relation_logits_nobkg)
        relation_logits_nobkg[:,gt_explicit] = 0.
       
        # pseudo_targets_exp = pseudo_targets[mask_explicit.nonzero().squeeze(1)]
        pseudo_targets_exp = rel_labels_nobkg[mask_explicit.nonzero().squeeze(1)]
        rel_labels_nobkg_exp = rel_labels_nobkg[mask_explicit.nonzero().squeeze(1)]
        one_hot_gt_labels_exp = self.get_one_hot_labels(rel_labels_nobkg_exp)
        one_hot_pseudo_labels_exp = self.get_one_hot_labels(pseudo_targets_exp)
        # one_hot_exp_labels =  one_hot_pseudo_labels_exp
        one_hot_exp_labels = one_hot_gt_labels_exp + one_hot_pseudo_labels_exp

        relation_logits_exp = relation_logits[rel_labels_nobkg_idx][mask_explicit.nonzero().squeeze(1)]
        relation_logits_exp_raw = relation_logits_raw[rel_labels_nobkg_idx][mask_explicit.nonzero().squeeze(1)]
        mixed_one_hot_refined_labels = self.mix_up_labels(lam, idx,relation_logits_raw, rel_labels)
        mixed_one_hot_refined_labels = mixed_one_hot_refined_labels[rel_labels_nobkg_idx][mask_explicit.nonzero().squeeze(1)]

        
        mask_loss_samples_pseudo = None
       

        return relation_logits_exp, relation_logits_exp_raw, mixed_one_hot_refined_labels, one_hot_gt_labels_exp,pseudo_targets_exp, mask_loss_samples_pseudo

    def pseudo_labels_for_all_train_proposals(self, relation_logits,relation_logits_raw, lam, idx, loss_samples, rel_labels):
        
      

        idx_to_pred = {'1': 'above', '2': 'across', '3': 'against', '4': 'along', '5': 'and', '6': 'at', '7': 'attached to', '8': 'behind', '9': 'belonging to', '10': 'between', '11': 'carrying', '12': 'covered in', '13': 'covering', '14': 'eating', '15': 'flying in', '16': 'for', '17': 'from', '18': 'growing on', '19': 'hanging from', '20': 'has', '21': 'holding', '22': 'in', '23': 'in front of', '24': 'laying on', '25': 'looking at', '26': 'lying on', '27': 'made of', '28': 'mounted on', '29': 'near', '30': 'of', '31': 'on', '32': 'on back of', '33': 'over', '34': 'painted on', '35': 'parked on', '36': 'part of', '37': 'playing', '38': 'riding', '39': 'says', '40': 'sitting on', '41': 'standing on', '42': 'to', '43': 'under', '44': 'using', '45': 'walking in', '46': 'walking on', '47': 'watching', '48': 'wearing', '49': 'wears', '50': 'with'}
        # print(rel_labels_nobkg)
        # print(rel_labels)
        gt_explicit = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        mask_explicit = torch.tensor([1. if i in gt_explicit else 0. for i in rel_labels.long()]).to(self.device)

        relation_logits_nobkg = relation_logits_raw
        
        
        if self.self_train_loss == 'bce':
            relation_logits_nobkg = torch.sigmoid(relation_logits_nobkg)
        else:
            relation_logits_nobkg = F.softmax(relation_logits_nobkg, -1)
        
        relation_logits_nobkg[:,gt_explicit] = 0. 
        probs, pseudo_targets = torch.max(relation_logits_nobkg, 1)
    
        
        pseudo_targets_exp = pseudo_targets[mask_explicit.nonzero().squeeze(1)]
        rel_labels_nobkg_exp = rel_labels[mask_explicit.nonzero().squeeze(1)]
        one_hot_gt_labels_exp = self.get_one_hot_labels(rel_labels_nobkg_exp)
        one_hot_pseudo_labels_exp = self.get_one_hot_labels(pseudo_targets_exp)
        # one_hot_exp_labels =  one_hot_pseudo_labels_exp
        one_hot_exp_labels = one_hot_gt_labels_exp + one_hot_pseudo_labels_exp

        relation_logits_exp = relation_logits[mask_explicit.nonzero().squeeze(1)]

        mixed_one_hot_refined_labels = self.mix_up_labels(lam, idx,relation_logits_raw, rel_labels,gt_explicit)
        mixed_one_hot_refined_labels = mixed_one_hot_refined_labels[mask_explicit.nonzero().squeeze(1)]

       
        loss_samples_nobkg = loss_samples[mask_explicit.nonzero().squeeze(1)]
        mask_loss_samples_pseudo = (loss_samples_nobkg < 0.5).float()

        return relation_logits_exp, mixed_one_hot_refined_labels, pseudo_targets_exp, mask_loss_samples_pseudo


    def multi_pseudo_labels_for_train_proposals(self, relation_logits, rel_labels):
        rel_labels_nobkg_idx = rel_labels.nonzero().squeeze(1)
        rel_labels_nobkg = rel_labels[rel_labels_nobkg_idx]
        gt_explicit = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        mask_explicit = torch.tensor([1. if i in gt_explicit else 0. for i in rel_labels_nobkg.long()]).to(self.device)

        relation_logits_nobkg = relation_logits[rel_labels_nobkg_idx]
        if self.self_train_loss == 'lsep':
            # relation_logits_nobkg = F.softmax(relation_logits_nobkg, -1)
            relation_logits_nobkg = torch.sigmoid(relation_logits_nobkg)
        elif self.self_train_loss == 'bce':
            relation_logits_nobkg = torch.sigmoid(relation_logits_nobkg)
        relation_logits_nobkg[:,gt_explicit] = 0. 
        
        _, pseudo_targets = torch.topk(relation_logits_nobkg, 3, 1)

        pseudo_targets_exp = (pseudo_targets[mask_explicit.nonzero().squeeze(1)]).squeeze(1)
        rel_labels_nobkg_exp = rel_labels_nobkg[mask_explicit.nonzero().squeeze(1)]
        one_hot_gt_labels_exp = self.get_one_hot_labels(rel_labels_nobkg_exp)
        one_hot_pseudo_labels_exp = F.one_hot(pseudo_targets_exp, num_classes=self.num_classes).sum(1)
        one_hot_exp_labels = one_hot_gt_labels_exp + one_hot_pseudo_labels_exp

        relation_logits_exp = (relation_logits[rel_labels_nobkg_idx])[mask_explicit.nonzero().squeeze(1)]
    
        return relation_logits_exp, one_hot_exp_labels




    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


    def generate_predicate_matrix_annotations(self, n_classes):

        self.labelmatrix_file = osp.join('meta_files/'+'vgimp'+'_onehot_highlevellabels_bkgd.pkl') #vrd_onehot_highlevellabels.pkl')
        self.labelmatrix_onlyHI_file = osp.join('meta_files/'+'vgimp'+'_onehot_highlevellabels_onlyHI_bkgd.pkl')

        self.labelmatrix_onlHI = pickle.load(open(self.labelmatrix_onlyHI_file, 'rb')) 

        self.labelmatrix = pickle.load(open(self.labelmatrix_file, 'rb')) 
        data = list(self.labelmatrix.items())
        labelmatrix_array = np.array(data)

        data_onlyHI = list(self.labelmatrix_onlHI.items())
        labelmatrix_onlyHI_array = np.array(data_onlyHI)

        self.labelmatrix_array = []
        self.labelmatrix_onlHI_array = []
        self.reverse_labelmatrix_array = []
        for i in range(len(labelmatrix_array)):
            self.labelmatrix_array.append(labelmatrix_array[i][1])
            self.labelmatrix_onlHI_array.append(labelmatrix_onlyHI_array[i][1])
            self.reverse_onehot = 1. - labelmatrix_array[i][1]
            self.reverse_onehot[i] = 1.
            self.reverse_labelmatrix_array.append(self.reverse_onehot)

        self.mutex_labelmatrix_array = []
        for i in range(len(labelmatrix_array)):
            self.labelmatrix_array.append(labelmatrix_array[i][1])
            self.reverse_onehot = labelmatrix_array[i][1]
            self.mutex_labelmatrix_array.append(self.reverse_onehot)
            
        classes = np.arange(n_classes)
        self.labelmatrix_classes_list_array = []
        for lbl_idx in classes:
            self.labelmatrix_classes_list_array.append([classes[i] for i in range(n_classes) if int(self.labelmatrix_array[lbl_idx][i])==1.])
        self.labelmatrix_classes_list_array = np.array(self.labelmatrix_classes_list_array)

        self.reverse_labelmatrix_classes_list_array = []
        for lbl_idx in classes:
            self.reverse_labelmatrix_classes_list_array.append([classes[i] for i in range(n_classes) if int(self.reverse_labelmatrix_array[lbl_idx][i])==0.])
        self.reverse_labelmatrix_classes_list_array = np.array(self.reverse_labelmatrix_classes_list_array)

        self.labelmatrix_array = Variable(torch.tensor(np.stack(self.labelmatrix_array)).to(self.device))
        
        self.labelmatrix_onlHI_array = Variable(torch.tensor(np.stack(self.labelmatrix_onlHI_array)).to(self.device)).float()
        
        gt_explicit = [0, 1, 2, 3, 4, 6, 8, 10, 22, 23, 29, 31, 33, 43]
        gt_implicit = [5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
        self.prior_temp = []
        for i, lbl_idx in enumerate(classes):
            class_prior_temp = torch.ones(len(classes)).to(self.device)
            for k in range(len(classes)):
                if classes[k] in gt_implicit and self.labelmatrix_onlHI_array[i][k]==1:
                    class_prior_temp[k] = 0.5
                elif classes[k] in gt_explicit  and self.labelmatrix_onlHI_array[i][k]==1:
                    class_prior_temp[k] = 2.0
                
                self.prior_temp.append(class_prior_temp)
    
        self.prior_temp = Variable(torch.tensor(torch.stack(self.prior_temp)).to(self.device))

        self.labelmatrix_onlHI_array_probs = []
        for lbl_idx in classes:
            if lbl_idx in  gt_explicit:
                imp_one_hot_probs = self.labelmatrix_onlHI_array[lbl_idx] / self.labelmatrix_onlHI_array[lbl_idx].sum()
                imp_one_hot_probs[imp_one_hot_probs==0.] = 10e-5
                self.labelmatrix_onlHI_array_probs.append(imp_one_hot_probs)
                
            else:
                imp_probs = []
                for k in range(len(classes)):
                    if classes[k] in gt_implicit:
                        imp_probs.append((self.labelmatrix_onlHI_array[lbl_idx][k] / self.labelmatrix_onlHI_array[lbl_idx].sum()) *2.)
                    else:
                        imp_probs.append((self.labelmatrix_onlHI_array[lbl_idx][k] / self.labelmatrix_onlHI_array[lbl_idx].sum()) /2.)
                # print(imp_probs[0:2])
                imp_probs = torch.stack(imp_probs)
                softmax_imp_probs = torch.exp(imp_probs) / torch.exp(imp_probs)[self.labelmatrix_onlHI_array[lbl_idx]!=0.].sum()
                softmax_imp_probs[self.labelmatrix_onlHI_array[lbl_idx]==0.] = 10e-5
                self.labelmatrix_onlHI_array_probs.append(softmax_imp_probs)
            # print(self.labelmatrix_onlHI_array_probs[0:2])
           
                # self.labelmatrix_onlHI_array_probs.append(torch.stack(imp_probs))
        self.labelmatrix_onlHI_array_probs = Variable(torch.tensor(torch.stack(self.labelmatrix_onlHI_array_probs)).cuda())
        
        self.reverse_labelmatrix_array = Variable(torch.tensor(np.stack(self.reverse_labelmatrix_array)).to(self.device))
        self.mutex_labelmatrix_array = Variable(torch.tensor(np.stack(self.mutex_labelmatrix_array)).to(self.device))
       
        return self.reverse_labelmatrix_array, self.mutex_labelmatrix_array, self.labelmatrix_onlHI_array, self.labelmatrix_onlHI_array_probs, self.prior_temp



def _to_one_hot(y, n_dims,device,dtype=torch.cuda.FloatTensor):
    """ 
    Take integer y (tensor or variable) with n dims and 
    convert it to 1-hot representation with n+1 dims
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.to(device).float().view(-1, 1)
    
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    # print(y_tensor)
    # print(y_tensor.shape)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).to(device).long().scatter(1, y_tensor.long(), 1)
    
    y_one_hot = y_one_hot.view(y.size()[0], -1).float()

    return y_one_hot



class LSEP(Function): 
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
    """
    
  
    @staticmethod
    def forward(ctx, input, target, device):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        
        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]):
            pos = positive_indices[i].nonzero()
            neg = negative_indices[i].nonzero()
            pos_examples = input[i, pos]
            neg_examples = torch.transpose(input[i, neg], 0, 1)
            loss += torch.sum(torch.exp(neg_examples - pos_examples))
        
        loss = torch.log(1 + loss)
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.device = device
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        # print(loss)
        return loss


    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        dtype = torch.cuda.FloatTensor
        device = ctx.device
        input, target = ctx.saved_variables
        N = input.size()[1]
        loss = Variable(ctx.loss, requires_grad = False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices


        fac  = -1 / loss
        grad_input = torch.zeros(input.size()).to(device).float()#type(dtype)
        
        scale = grad_input.size(0), -1
        one_hot_pos = _to_one_hot(positive_indices.nonzero()[:, 1].view(*scale), N, device)
        one_hot_neg = _to_one_hot(negative_indices.nonzero()[:, 1].view(*scale), N, device)

        ## grad
        for i in range(grad_input.size()[0]):
            for dum_j,phot in enumerate(one_hot_pos[i]):
                for dum_k,nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot-nhot)*torch.exp(-input[i].data*(phot-nhot))
        ## 
        # grad_input = Variable(grad_input) * (grad_output * fac)
        grad_input = (grad_input) * (grad_output * fac)
 
        return grad_input, None, None
    
    
#--- main class

def Log_Sum_Exp_Pairwise_Loss(predictions, labels, size = 1):
    loss_op = 0.0
    size = predictions.size()[0]
    for i in range(size):
        positive = predictions[i][labels[i].nonzero().squeeze(1)]
        negative = predictions[i][(1.-labels[i]).nonzero().squeeze(1)]
   

        exp_sub = torch.exp(negative.unsqueeze(1) - positive.unsqueeze(0))
        exp_sum = exp_sub.sum()
        
        loss_op += torch.log(1 + exp_sum)
    
    return loss_op / size


class LSEPLoss(nn.Module): 
    def __init__(self, device, num_classes): 
        super(LSEPLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, predictions, labels): 

 
        loss_op = 0.0
        size = predictions.size()[0]
        for i in range(size):
            positive = predictions[i][labels[i].nonzero().squeeze(1)]
            negative = predictions[i][(1.-labels[i]).nonzero().squeeze(1)]
    

            exp_sub = torch.exp(negative.unsqueeze(1) - positive.unsqueeze(0))
            exp_sum = exp_sub.sum()
            
            loss_op += torch.log(1 + exp_sum)
    
        return loss_op / size
     

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class JSDiv(nn.Module):
    def __init__(self, device, size_average=True):
        super(JSDiv, self).__init__()
        self.device = device
        self.size_average = size_average

    def KLDivLoss(self,p,q):

        kl_loss = (((p*(p.log()-q.log())).sum(1))).mean()
        return kl_loss

    def forward(self,p_output, q_output):
        """
        Function that measures JS divergence between target and output logits:
        """

        log_mean_output = ((p_output + q_output )/2)
        return (self.KLDivLoss(log_mean_output, p_output) + self.KLDivLoss(log_mean_output, q_output))/2

class BCELosswithTemp(nn.Module):
    def __init__(self,device,bce_loss,size_average=True):

        super().__init__()
        self.size_average = size_average
        self.device = device
        self.bce_loss = bce_loss
        
    def forward(self, input,implied_labelmatrix_array, target, T):
        self.T = T[target]
        one_hot_targets = implied_labelmatrix_array[target]
        # print(self.T[0:2])
        input = input / self.T

        loss = self.bce_loss(input, one_hot_targets)

        return loss
       

class CrossEntropyMultiLabel(nn.Module):
    def __init__(self,device,size_average=True):

        super().__init__()
        self.size_average = size_average
        self.device = device
    
        
    def forward(self, input, target):
        
        
        pred = F.log_softmax(input, dim=1)
        # labels are probability targets 
        loss = -1 * torch.sum(target * pred, dim=1)
    
        return loss.mean()
       

class PMESoftmax(Function):
    """
    CrossEntropy function with custom forward and backward propagation function
    """
    @staticmethod
    def forward(ctx, x, target, mch_labels, gt_labels,mask):
        """
        forward propagation
        function:
        E(t_i, x) = - \sum_i t_i log x, where t_i is one-hot label
        """
       
        ctx.save_for_backward(x, target, mch_labels, gt_labels,mask)
        mch_batch_labels = mch_labels[gt_labels].bool()
        softmax_x = torch.exp(x) / (torch.exp(x) * mch_batch_labels).sum(1).unsqueeze(1).expand_as(x)
        log_softmax_x = torch.log(torch.clamp(softmax_x, min=1e-5, max=1.))
        

        mch_batch_labels = mch_labels[gt_labels]
        # log_softmax_x[log_softmax_x!=log_softmax_x] = -50.0
        log_softmax_x[mch_batch_labels == 0.] = 0.
        softmax_x[mch_batch_labels == 0.] = 0.
        
        
        # output = (- target * torch.log(softmax_x)).sum() / softmax_x.size(0)
        output_mul = (- target * log_softmax_x)
        output = (((output_mul).sum(1))*mask).mean()

        return output


    @staticmethod
    def backward(ctx, grad_output):
        """
        backward propagation
        gradient:
        \frac{\partial E}{\partial x} = - t_i / x
        """
        # print(grad_output)
        x, target, mch_labels, gt_labels, mask = ctx.saved_variables
        mch_batch_labels_bool = mch_labels[gt_labels].bool()
        softmax_x = torch.exp(x) / (torch.exp(x) * mch_batch_labels_bool).sum(1).unsqueeze(1).expand_as(x)
        grad_input = grad_output * (softmax_x - target) / x.size(0)
        mask = mask.unsqueeze(1).expand_as(x)
        # grad_input = - grad_output * target / x / x.size(0)
        mch_batch_labels = mch_labels[gt_labels] ### [batch_size, 70]
        grad_input[mch_batch_labels == 0.] = 0.
        grad_input = grad_input * mask
        # print('backward')
        # print(grad_input[0:3])
        # input('enter')
        return grad_input, None, None, None, None


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.PME_SOFTMAX_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.BCE_TEMP_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
        cfg.MODEL.ROI_RELATION_HEAD.TWO_HEADS,
        cfg.MODEL.ROI_RELATION_HEAD.SELF_TRAIN_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.MANIFOLD_MIXUP,
        cfg.MODEL.ROI_RELATION_HEAD.IMP_MANIFOLD_MIXUP,
        cfg.MODEL.ROI_RELATION_HEAD.SOFT_LABELS,
        cfg.MODEL.DEVICE,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator



   
