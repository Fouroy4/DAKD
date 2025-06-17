from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

def js_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    mean_pred = 0.5 * (pred_student + pred_teacher)
    
    log_mean = torch.log(mean_pred + 1e-10) 
    kl_student_mean = F.kl_div(
        F.log_softmax(logits_student / temperature, dim=1),
        mean_pred,
        reduction="none"
    ).sum(1)
    
    kl_teacher_mean = F.kl_div(
        F.log_softmax(logits_teacher / temperature, dim=1),
        mean_pred,
        reduction="none"
    ).sum(1)
    
    js_div = 0.5 * (kl_student_mean + kl_teacher_mean)
    
    if reduce:
        loss_js = js_div.mean()
    else:
        loss_js = js_div
    
    loss_js *= temperature**2
    
    return loss_js

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MLKD(nn.Module):
    def __init__(self, cfg):
        super(MLKD, self).__init__()
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 

        self.kd_weight = 1.0
        self.instance_weight = 0.5
        self.class_weight =1 - self.instance_weight


        self.use_kd_loss = False
        self.use_instance_loss = False
        self.use_class_loss = False

    def forward(self, logits_teacher_strong,logits_teacher_weak, logits_student_strong, logits_student_weak, target, rgb_teacher_center, ir_teacher_center, **kwargs):
        temp = 0.05
        logits_student_strong_c = (logits_student_strong.mm(rgb_teacher_center.t()))/temp
        logits_student_weak_c = (logits_student_weak.mm(ir_teacher_center.t()))/temp

        logits_teacher_strong_c = (logits_teacher_strong.mm(rgb_teacher_center.t()))/temp
        logits_teacher_weak_c = (logits_teacher_weak.mm(ir_teacher_center.t()))/temp

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        # conf_thresh = confidence.cpu().numpy().flatten().max() + 1e-6

        mask = confidence.le(conf_thresh).bool()
        # mask = confidence.gt(conf_thresh).bool()


        pred_teacher_weak_c = F.softmax(logits_teacher_weak_c.detach(), dim=1)
        confidence_c, pseudo_labels = pred_teacher_weak_c.max(dim=1)
        confidence_c = confidence_c.detach()
        conf_thresh_c = np.percentile(
            confidence_c.cpu().numpy().flatten(), 50
        )
        # conf_thresh_c = confidence_c.cpu().numpy().flatten().max() + 1e-6

        mask_c = confidence_c.le(conf_thresh_c).bool()
        # mask_c = confidence_c.gt(conf_thresh_c).bool()

        loss_kd_strong = self.kd_loss_weight * ((kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        )).mean() ) 
        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        )).mean() ) 

        loss_instance_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean())

        loss_instance_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
        )

        loss_class_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask_c).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask_c).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask_c).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask_c).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask_c).mean())

        loss_class_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong_c,
            logits_teacher_strong_c,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong_c,
            logits_teacher_strong_c,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong_c,
            logits_teacher_strong_c,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak_c,
            logits_teacher_weak_c,
            6.0,
            logit_stand=self.logit_stand,
        )
        
        loss_kd = 0
        loss_instance = 0
        loss_class = 0

        loss_kd = torch.tensor(0.0, device=logits_student_strong.device) if not self.use_kd_loss else loss_kd
        loss_instance = torch.tensor(0.0, device=logits_student_strong.device) if not self.use_instance_loss else loss_instance
        loss_class = torch.tensor(0.0, device=logits_student_strong.device) if not self.use_class_loss else loss_class

        if self.use_kd_loss:
            loss_kd = self.kd_weight * (loss_kd_strong + loss_kd_weak)

        if self.use_instance_loss:
            loss_instance = self.instance_weight * (loss_instance_weak + loss_instance_strong)
        
        if self.use_class_loss:
            loss_class = self.class_weight * (loss_class_weak + loss_class_strong)
        
        loss = loss_kd + loss_instance  + loss_class 
        
        losses_dict = {
            "total_loss": loss,
            "loss_kd": loss_kd,
            "loss_instance": loss_instance,
            "loss_class": loss_class,
            "loss_kd_strong": loss_kd_strong,
            "loss_kd_weak": loss_kd_weak,
            "loss_instance_strong": loss_instance_strong,
            "loss_instance_weak": loss_instance_weak,
            "loss_class_strong": loss_class_strong,
            "loss_class_weak": loss_class_weak
        }
    
        return losses_dict

        


