import torch
import torch.nn as nn
from MLKD import MLKD
from yacs.config import CfgNode as CN
from hard_mine_triplet_loss import TripletLoss

class DistillModel(nn.Module):
    def __init__(self, teacher, student):
        super(DistillModel, self).__init__()
        self.teacher = teacher      
        self.student = student      

        # Configuration for MLKD
        CFG = CN()
        CFG.KD = CN()
        CFG.KD.TEMPERATURE = 4
        CFG.KD.LOSS = CN()
        CFG.KD.LOSS.CE_WEIGHT = 0.1
        CFG.KD.LOSS.KD_WEIGHT = 0.9
        CFG.EXPERIMENT = CN()
        CFG.EXPERIMENT.LOGIT_STAND = True

        self.mlkd_loss = MLKD(CFG)
        self.tri_loss = TripletLoss()

        # Add explicit flags for ablation
        self.mlkd = False
        self.use_kd = False
        self.use_instance = False
        self.use_class = False

    def forward(self, rgb_inputs, ir_inputs, labels):
        losses = {
            "total_loss": 0,
            "loss_kd": 0,
            "loss_instance": 0,
            "loss_class": 0,
            "loss_kd_strong": 0,
            "loss_kd_weak": 0,
            "loss_instance_strong": 0,
            "loss_instance_weak": 0,
            "loss_class_strong": 0,
            "loss_class_weak": 0,
            "id_loss": 0,
            "triplet_loss": 0,
            "infonce_loss": 0,
        }

        # Get student features
        rgb_student_logits = self.student(rgb_inputs)
        ir_student_logits = self.student(ir_inputs)

        if self.mlkd:
            # Get teacher outputs (no gradient)
            with torch.no_grad():
                self.teacher.eval()
                rgb_teacher_logits = self.teacher(rgb_inputs)
                ir_teacher_logits = self.teacher(ir_inputs)
                rgb_teacher_center = self.teacher.memory_rgb.centers
                ir_teacher_center = self.teacher.memory_ir.centers

            # Configure MLKD for ablation
            self.mlkd_loss.use_kd_loss = self.use_kd
            self.mlkd_loss.use_instance_loss = self.use_instance
            self.mlkd_loss.use_class_loss = self.use_class

            # Get MLKD losses
            losses_dict = self.mlkd_loss(
                rgb_teacher_logits, 
                ir_teacher_logits,
                rgb_student_logits, 
                ir_student_logits,
                labels,
                rgb_teacher_center,
                ir_teacher_center
            )

            # Accumulate losses based on enabled components
            if self.use_kd:
                losses["loss_kd"] = losses_dict["loss_kd"]
                losses["loss_kd_strong"] = losses_dict["loss_kd_strong"]
                losses["loss_kd_weak"] = losses_dict["loss_kd_weak"]

            if self.use_instance:
                losses["loss_instance"] = losses_dict["loss_instance"]
                losses["loss_instance_strong"] = losses_dict["loss_instance_strong"]
                losses["loss_instance_weak"] = losses_dict["loss_instance_weak"]

            if self.use_class:
                losses["loss_class"] = losses_dict["loss_class"]
                losses["loss_class_strong"] = losses_dict["loss_class_strong"]
                losses["loss_class_weak"] = losses_dict["loss_class_weak"]

            losses["total_loss"] += losses_dict["total_loss"]

        return losses