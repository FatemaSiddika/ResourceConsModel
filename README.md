```bash
pip install timm
pip install tqdm

# Training Configurations for CIFAR-10

| **Model**       | **Train Size** | **Test Size** | **Pre-trained** | **Batch** | **LR** | **Momentum** | **Weight Decay** | **Epochs** | **Optimizer** |
|------------------|-------------------------|------------------------|------------------|----------------|--------------------|--------------|------------------|------------|---------------|
| **MobileNetV2**  | 50,000                 | 10,000                | No               | 128            | 0.05               | 0.9          | 4e-05            | 200        | SGD           |
| **ResNet34**     | 50,000                 | 10,000                | No               | 128            | 0.1                | 0.9          | 5e-4             | 150        | SGD           |
| **ViT-Base**     | 50,000                 | 10,000                | No               | 128            | 0.003              | 0.9          | 1e-2             | 50         | AdamW         |



# Training Configurations for CIFAR-10 (Pretrained Models)

| **Model**       | **Train Dataset Size** | **Test Dataset Size** | **Pre-trained** | **Batch Size** | **Learning Rate** | **Momentum** | **Weight Decay** | **Epochs** | **Optimizer** |
|------------------|-------------------------|------------------------|------------------|----------------|--------------------|--------------|------------------|------------|---------------|
| **MobileNetV2**  | 50,000                 | 10,000                | Yes              | 128            | 0.001              | 0.9          | 1e-5             | 100        | SGD           |
| **ResNet34**     | 50,000                 | 10,000                | Yes              | 128            | 0.001              | 0.9          | 1e-4             | 100        | SGD           |
| **ViT-Base**     | 50,000                 | 10,000                | Yes              | 128            | 5e-5               | -            | 1e-2             | 50         | AdamW         |