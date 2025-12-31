#######################
## Dataset splitting ##
#######################

train_len = int(0.8 * len(X_scaled)) # MUST be INTEGER
val_len = int(0.1 * len(X_scaled))
test_len = len(X_scaled) - (train_len + val_len)

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X_scaled, y_scaled)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=2**11, shuffle=True)
val_set = DataLoader(val_split, batch_size=2**11, shuffle=False)
test_set = DataLoader(test_split, batch_size=2**11, shuffle=False)

###################
## Class weights ##
###################

from collections import Counter

# Extract labels from training split only
train_labels = [y_scaled[idx].item() for idx in train_split.indices]
class_counts = Counter(train_labels)

# Calculate class weights (inverse frequency)
class_count_tensor = torch.tensor([class_counts[i] for i in range(7)], dtype=torch.float32)
class_weights = 1.0 / class_count_tensor
class_weights = class_weights / class_weights.sum() * 7  # Normalize to sum = 7

print("\nClass distribution in training set:")
for i in range(7):
    orig_class = i + 1  # Show original class labels (1-7)
    print(f"  Class {orig_class}: {class_counts[i]:>6} samples, weight: {class_weights[i]:.4f}")
# Class distribution in training set:
#   Class 1: 169117 samples, weight: 0.0541
#   Class 2: 226849 samples, weight: 0.0403
#   Class 3:  28730 samples, weight: 0.3185
#   Class 4:   2195 samples, weight: 4.1691
#   Class 5:   7614 samples, weight: 1.2019
#   Class 6:  13915 samples, weight: 0.6576
#   Class 7:  16389 samples, weight: 0.5584

# Move weights to device
class_weights = class_weights.to(device)

#############
## Testing ##
#############

test_loss = 0
test_preds_list = []
test_true_list = []

model.eval()

with torch.inference_mode():
    for X_test, y_test in test_set:
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_test = y_test.squeeze()  # Remove extra dimension
        
        test_preds = model(X_test)  # Raw logits (batch_size, 7)
        
        # Accumulate loss
        test_loss += loss_fn(test_preds, y_test).item()
        
        # Get predicted class (argmax of logits)
        _, predicted_classes = torch.max(test_preds, 1)
        
        # Collect predictions and true labels
        test_preds_list.append(predicted_classes.cpu())
        test_true_list.append(y_test.cpu())

# Calculate average test loss
avg_test_loss = test_loss / len(test_set)
print(f"Average Test Loss: {avg_test_loss:.4f}\n")

# Concatenate all batches
test_preds_class = torch.cat(test_preds_list, dim=0).numpy()  # Predicted classes (0-6)
test_true = torch.cat(test_true_list, dim=0).numpy()  # True classes (0-6)

# Calculate metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

accuracy = accuracy_score(test_true, test_preds_class)
print(f'Accuracy on test set: {accuracy:.4f}\n')

# Confusion Matrix
labels = [f'Class {i+1}' for i in range(7)]  # Class 1-7 for display
cm = confusion_matrix(test_true, test_preds_class)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print(f'Confusion matrix:\n{cm_df}\n')

# Classification Report
print(f'Classification report:\n{classification_report(test_true, test_preds_class, target_names=labels, digits=4)}\n')

# Per-class accuracy (useful for imbalanced datasets)
print("Per-class accuracy:")
for i in range(7):
    class_mask = (test_true == i)
    if class_mask.sum() > 0:
        class_acc = (test_preds_class[class_mask] == i).sum() / class_mask.sum()
        print(f"  Class {i+1}: {class_acc:.4f} ({class_mask.sum()} samples)")
    else:
        print(f"  Class {i+1}: No samples in test set")

# Check minority class performance (Class 4 and 5)
print("\n⚠️ Minority class performance:")
for i in [3, 4]:  # Class 4 and 5 (indices 3 and 4)
    class_mask = (test_true == i)
    if class_mask.sum() > 0:
        recall = (test_preds_class[class_mask] == i).sum() / class_mask.sum()
        precision = (test_preds_class == i).sum()
        if precision > 0:
            precision = (test_preds_class[class_mask] == i).sum() / (test_preds_class == i).sum()
        else:
            precision = 0
        print(f"  Class {i+1}: Recall={recall:.4f}, Precision={precision:.4f}")

# Average Test Loss: 0.0047

# Accuracy on test set: 0.2988

# Confusion matrix:
#          Class 1  Class 2  Class 3  Class 4  Class 5  Class 6  Class 7
# Class 1     9316      277        1        0     2027      124     9478
# Class 2    11215     1687      371        6    10558     1652     2854
# Class 3        0        0     1463      243      151     1713        0
# Class 4        0        0        0      264        0        3        0
# Class 5        0        0        0        0      967        3        0
# Class 6        0        0        8       50       10     1637        0
# Class 7        0        0        0        0        0        0     2024

# Classification report:
#               precision    recall  f1-score   support

#      Class 1     0.4538    0.4390    0.4462     21223
#      Class 2     0.8590    0.0595    0.1113     28343
#      Class 3     0.7938    0.4098    0.5406      3570
#      Class 4     0.4689    0.9888    0.6361       267
#      Class 5     0.0705    0.9969    0.1317       970
#      Class 6     0.3190    0.9601    0.4789      1705
#      Class 7     0.1410    1.0000    0.2471      2024

#     accuracy                         0.2988     58102
#    macro avg     0.4437    0.6934    0.3703     58102
# weighted avg     0.6511    0.2988    0.2783     58102


# Per-class accuracy:
#   Class 1: 0.4390 (21223 samples)
#   Class 2: 0.0595 (28343 samples)
#   Class 3: 0.4098 (3570 samples)
#   Class 4: 0.9888 (267 samples)
#   Class 5: 0.9969 (970 samples)
#   Class 6: 0.9601 (1705 samples)
#   Class 7: 1.0000 (2024 samples)

# ⚠️ Minority class performance:
#   Class 4: Recall=0.9888, Precision=0.4689
#   Class 5: Recall=0.9969, Precision=0.0705