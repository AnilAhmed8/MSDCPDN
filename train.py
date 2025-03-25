import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, T):
    """
    Computes the KL divergence loss between softened outputs of student and teacher.
    """
    student_probs = torch.sigmoid(student_logits / T)
    teacher_probs = torch.sigmoid(teacher_logits / T)
    # For binary classification, form 2-class distributions.
    student_dist = torch.cat([student_probs, 1 - student_probs], dim=1)
    teacher_dist = torch.cat([teacher_probs, 1 - teacher_probs], dim=1)
    log_student_dist = torch.log(student_dist + 1e-8)
    loss = F.kl_div(log_student_dist, teacher_dist, reduction='batchmean') * (T * T)
    return loss

def train_student(teacher, student, dataloader, epochs, optimizer, T, alpha, device, use_ground_truth=True):
    """
    Trains the student model with knowledge distillation.
    If use_ground_truth is True, the BCE loss uses the provided labels.
    Otherwise, teacher pseudo-labels (soft targets) are used.
    """
    bce_criterion = nn.BCEWithLogitsLoss()
    teacher.eval()
    student.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for S1, S2, labels in dataloader:
            S1, S2 = S1.to(device), S2.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(S1, S2)
            
            student_logits = student(S1, S2)
            loss_kl = distillation_loss(student_logits, teacher_logits, T)
            
            if use_ground_truth:
                # Only compute BCE if labels are valid (>=0)
                loss_bce = bce_criterion(student_logits, labels)
            else:
                # Use teacher's pseudo-labels as soft targets
                pseudo_labels = torch.sigmoid(teacher_logits).detach()
                loss_bce = bce_criterion(student_logits, pseudo_labels)
            
            loss_total = alpha * loss_kl + (1 - alpha) * loss_bce
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return student

def evaluate_student(model, dataloader, device):
    """
    Evaluates the student model on the given dataloader.
    Returns lists: y_pred, y_true.
    Samples with label -1 (i.e. unlabeled) are skipped.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for S1, S2, labels in dataloader:
            S1, S2, labels = S1.to(device), S2.to(device), labels.to(device)
            # Skip unlabeled examples (label == -1)
            mask = (labels != -1).view(-1)
            if mask.sum().item() == 0:
                continue
            logits = model(S1, S2)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).int()  # Binary thresholding
            valid_preds = preds[mask].cpu().numpy()
            valid_labels = labels[mask].cpu().numpy()
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
    return all_preds, all_labels