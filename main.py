import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from preprocessing import preprocess_data, text_to_sequence
from dataset import PreprocessedSiameseDataset
from model import SiameseModel
from train import train_student, evaluate_student
from utils import compute_metrics, print_metrics
from visualization import plot_confusion_matrix, plot_metrics

def main():
    # Settings
    supervised = True         # Set to False for semi-supervised training (use teacher pseudo-labels)
    max_len = 50              # Maximum sequence length for padding
    batch_size = 32
    epochs = 30
    dropout=0.3
    learning_rate = 1e-3
    temperature = 2.0         # Temperature for distillation
    alpha = 0.7               # Mixing coefficient (KL vs BCE)
    embed_dim = 768           # Embedding dimension
    hidden_dim = 512          # Hidden dimension for RNNs
    attention_dim = 768        # Attention projection dimension
    num_heads = 12             # Number of attention heads for multi-head attention
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Preprocess Training Data (Arabic)
    # -----------------------------
    # Assume "arabic_train.csv" exists with columns: text1, text2, label
    train_data, vocab = preprocess_data("arabic_train.csv", lang="arabic", build_vocab_flag=True)
    print(f"Built vocabulary of size: {len(vocab)}")
    
    # Convert token lists to index sequences is done in the dataset class
    train_dataset = PreprocessedSiameseDataset(train_data, vocab, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # -----------------------------
    # Preprocess Test Data (Urdu)
    # -----------------------------
    # Assume "urdu_test.csv" exists with columns: text1, text2, label
    test_data, _ = preprocess_data("urdu_test.csv", lang="urdu", build_vocab_flag=False, vocab=vocab)
    test_dataset = PreprocessedSiameseDataset(test_data, vocab, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # -----------------------------
    # Initialize Teacher and Student Models
    # -----------------------------
    vocab_size = len(vocab)
    teacher_model = SiameseModel(vocab_size, embed_dim, hidden_dim, attention_dim, num_heads=num_heads).to(device)
    student_model = SiameseModel(vocab_size, embed_dim, hidden_dim, attention_dim, num_heads=num_heads).to(device)
    
    # Simulate a pretrained teacher by freezing its parameters.
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # -----------------------------
    # Train Student with Knowledge Distillation
    # -----------------------------
    print("Starting training with knowledge distillation...")
    trained_student = train_student(teacher_model, student_model, train_loader,
                                    epochs, optimizer, temperature, alpha, device,
                                    use_ground_truth=supervised)
    
    # -----------------------------
    # Evaluate on Test Data
    # -----------------------------
    print("Evaluating on test set...")
    y_pred, y_true = evaluate_student(trained_student, test_loader, device)
    if len(y_true) == 0:
        print("No labeled samples in test set for evaluation.")
        return
    cm, precision, recall, f1, accuracy = compute_metrics(y_true, y_pred)
    print_metrics(cm, precision, recall, f1, accuracy)
    
    # -----------------------------
    # Visualize Results
    # -----------------------------
    plot_confusion_matrix(cm)
    plot_metrics(precision, recall, f1, accuracy)

if __name__ == "__