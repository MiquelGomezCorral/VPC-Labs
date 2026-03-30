from torch import nn

# Architecture SpecificationsInput Layer: 
# $32\times32$ pixel RGB images.
# Convolutional Block 1: 
#   Two consecutive convolutional layers with 32 filters each, using a $3\times3$ spatial dimension.
#   Followed by a $2\times2$ Max-Pooling layer.
# 
# Convolutional Block 2: 
#   Two consecutive convolutional layers with 64 filters each, using a $3\times3$ spatial dimension.
#   Followed by a $2\times2$ Max-Pooling layer.
# 
# Fully-Connected (FC) Layer: A single layer containing 16 neurons.
# 
# Output Layer: A Softmax layer with 2 neurons for binary gender classification.
# 
# Technical Parameters
#   Activation Function: Rectified Linear Units (ReLU) are used in all layers.
#   Regularization: Dropout is applied to both convolutional and fully-connected activations to prevent overfitting.
#   Optimization Gain: This specific architecture is 16 times more memory efficient than the initial Starting CNN while maintaining comparable accuracy.

class GENDER_CNN(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=2):
        super(GENDER_CNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Output logits for binary classification
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


    def print_number_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        model_mb = model_bytes / (1024 ** 2)

        print(f"Model parameters (trainable/total): {trainable_params:,}/{total_params:,}")
        print(f"Approx. model size (parameters only): {model_mb:.3f} MB")