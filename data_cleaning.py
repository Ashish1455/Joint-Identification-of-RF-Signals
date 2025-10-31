import numpy as np

# List of input .npz files
input_files = ['conv2.npz', 'conv8.npz', 'conv16.npz', 'conv64.npz', 'turbo2.npz', 'turbo8.npz', 'turbo16.npz', 'turbo64.npz', 'polar2.npz', 'polar8.npz', 'polar16.npz', 'polar64.npz']

all_train_features = []
all_train_labels = []

all_test_features = []
all_test_labels = []

for label, filename in enumerate(input_files):
    # Load the file
    data = np.load(filename)
    
    # Extract features
    features = data['features']
    print(f"{filename} - features shape: {features.shape}")
    
    # Slice train and test data
    train = features[3000:13000]   # 10,000 samples
    test = features[13000:14000]   # 1,000 samples
    
    print(f"  Train: {train.shape}, Test: {test.shape}, Label: {label}")
    
    # Create label arrays
    train_labels = np.full((train.shape[0],), label, dtype=np.int64)
    test_labels = np.full((test.shape[0],), label, dtype=np.int64)
    
    # Append to master lists
    all_train_features.append(train)
    all_train_labels.append(train_labels)
    
    all_test_features.append(test)
    all_test_labels.append(test_labels)

# Combine all parts
combined_train_features = np.concatenate(all_train_features, axis=0)
combined_train_labels = np.concatenate(all_train_labels, axis=0)

combined_test_features = np.concatenate(all_test_features, axis=0)
combined_test_labels = np.concatenate(all_test_labels, axis=0)

print(f"\nFinal Train Shape: {combined_train_features.shape}, Labels: {combined_train_labels.shape}")
print(f"Final Test Shape: {combined_test_features.shape}, Labels: {combined_test_labels.shape}")

# Save into single .npz files
np.savez('train.npz', features=combined_train_features, labels=combined_train_labels)
np.savez('test.npz', features=combined_test_features, labels=combined_test_labels)

print("\nSaved: train.npz and test.npz (with features and labels)")