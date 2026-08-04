# Session 10 — Custom ResNet on CIFAR-10

This folder contains Assignment 10 from the ERA course by The School of AI.

## 1. Problem Statement

Build and train a custom residual network for CIFAR-10 with the following prescribed architecture:

- **Preparation layer:** `3×3 Conv (stride=1, padding=1) → BatchNorm → ReLU`, producing 64 channels.
- **Layer 1:**
  - `X = 3×3 Conv → MaxPool2d → BatchNorm → ReLU`, producing 128 channels.
  - `R1 = Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU`, retaining 128 channels.
  - Combine the paths using `X + R1`.
- **Layer 2:** `3×3 Conv → MaxPool2d → BatchNorm → ReLU`, producing 256 channels.
- **Layer 3:**
  - `X = 3×3 Conv → MaxPool2d → BatchNorm → ReLU`, producing 512 channels.
  - `R2 = Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU`, retaining 512 channels.
  - Combine the paths using `X + R2`.
- **Classifier:** apply max pooling with a kernel size of 4, followed by a fully connected layer and class prediction.

Use the **One Cycle learning-rate policy** with these constraints:

- Train for 24 epochs.
- Reach the maximum learning rate at epoch 5.
- Find suitable minimum and maximum learning rates.
- Do not use an annihilation phase.

Use this training transform:

```text
Pad by 4 → RandomCrop(32×32) → HorizontalFlip → CutOut(8×8)
```

Use a batch size of 512, the Adam optimizer, and cross-entropy loss. The target validation accuracy is **90%**.

## 2. Implementation

### Repository structure

- `custom_resnet.py` defines the custom residual network.
- `utils.py` contains device, training, evaluation, model-summary, and metric-plotting helpers.
- `S10.ipynb` prepares CIFAR-10, finds a learning rate, configures the One Cycle policy, trains the model, and reports results.

### Quick revision notes

#### Residual connections

The two residual stages learn a correction to their input rather than an entirely new representation. The shortcut and residual tensors have identical shapes, so they can be added directly:

```python
x = self.convblockL1X1(x)
x = x + self.convblockL1R1(x)

x = self.convblockL3X1(x)
x = x + self.convblockL3R1(x)
```

This identity path improves gradient flow and allows the convolutional branch to focus on residual features.

#### Spatial and channel progression

```text
3×32×32 → 64×32×32 → 128×16×16 → 256×8×8
          → 512×4×4 → 512×1×1 → 10 logits
```

Each max-pooling operation halves the spatial dimensions while the convolutional stages increase channel capacity. The final `4×4` pool collapses each feature map to one value before classification.

#### Albumentations pipeline

The implemented training pipeline adds optional color jitter to the required crop, flip, and CutOut-style augmentation. Test images are only normalized:

```python
train_transforms = A.Compose([
    A.augmentations.transforms.ColorJitter(
        brightness=0.10, contrast=0.10, saturation=0.10,
        hue=0.10, always_apply=False, p=0.5),
    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
    A.RandomCrop(height=32, width=32, always_apply=True),
    A.HorizontalFlip(),
    A.Normalize(mean=means, std=stds, always_apply=True),
    A.CoarseDropout(max_holes=1, min_height=8, max_height=8,
                    min_width=8, max_width=8, fill_value=means,
                    always_apply=True),
    ToTensorV2(),
])
```

Padding to `40×40` followed by a `32×32` random crop is equivalent to allowing a four-pixel translation around the original image. `CoarseDropout` implements the required `8×8` CutOut region.

#### Learning-rate selection and scheduling

Before the full training run, an **LR range test** tries progressively larger learning rates over 200 mini-batches. `step_mode="exp"` means that the rate is multiplied by a constant factor between batches, rather than increased by a fixed amount. The test stops when the trial rate reaches 10 and plots loss against learning rate:

```python
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(
    train_loader,
    end_lr=10,
    num_iter=200,
    step_mode="exp",
)
lr_finder.plot()
lr_finder.reset()
```

The value `10` is only the upper boundary of this short search; it is **not** the learning rate used for the 24-epoch training run. The useful region is read from the plot—typically where loss is falling rapidly but before it becomes unstable. Based on that experiment, `4.51e-2` was selected as the maximum rate for `OneCycleLR`.

The full training run then uses Adam with weight decay and steps `OneCycleLR` after every batch:

```python
optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

scheduler = OneCycleLR(
    optimizer,
    max_lr=4.51e-2,
    steps_per_epoch=len(train_loader),
    epochs=24,
    pct_start=5/24,
    div_factor=80,
    three_phase=False,
    final_div_factor=550,
)
```

`pct_start=5/24` places the learning-rate peak at roughly epoch 5. Calling `scheduler.step()` inside the batch loop produces the intended per-iteration schedule.

### Model summary

```text
----------------------------------------------------------------
        Layer (type)      |       Output Shape      |   Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
------------------------------------------------------------------
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.07
Estimated Total Size (MB): 33.09
----------------------------------------------------------------
```

## 3. Results

Tuning the Albumentations pipeline and One Cycle parameters allowed the model to cross the 90% validation target at epoch 23 and improve further at epoch 24.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Final learning rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0.3193 | 88.81% | 0.0006 | **90.53%** | 0.00030234 |
| 24 | 0.3124 | 89.50% | 0.0006 | **90.63%** | 0.000001057 |

The final validation accuracy was **90.63%**, exceeding the assignment target by 0.63 percentage points.

## 5. API Reference

### `custom_resnet.py`

#### Get CNN model

```http
  getModel()
```


### `utils.py`

#### CUDA Availability

```http
  isCUDAAvailable()
```

#### Get PyTorch device

```http
  getDevice()
```

#### Plot Data

```http
  plotData(loader, count, cmap_code)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| loader | torch.utils.data.dataloader.DataLoader | Required: Loader consisting of train/test data to be plotted |
| count | integer | Required: The number of elements from loader to be plotted |
| cmap_code | string | Required: CMAP color code used for plotting |

#### Get Transform (Crop, Resize, Rotate) for train data
```http
  getTrainTransforms_CropRotate(centerCrop, resize, randomRotate, mean, std_dev)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| centerCrop | int | Required: Number of pixels to be cropped at center of the image |
| resize | integer | Required: Number of pixels to resize the cropped image |
| randomRotate | float | Required: Angle to rotate the image during training |
| mean | float | Required: Mean of the input data |
| std_dev | float | Required: Standard deviation of the input data |


#### Get Transform for test data
```http
  getTestTransforms(mean,std_dev)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| mean | float | Required: Mean of the input data |
| std_dev | float | Required: Standard deviation of the input data |


#### Train model
```http
  train(model, train_loader, optimizer, criterion, scheduler)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model. Use 'model.getModel()' to get an instance of our first PyTorch CNN model|
| train_loader | torch.utils.data.dataloader.DataLoader	 | Required: Loader consisting of train data |
| optimizer | torch.optim  | Required: Optimizer used in training data |
| criterion | function | Required: Fucntion used to calculate the loss during training |
| scheduler | torch.optim.lr_scheduler | Required: Learning-rate scheduler stepped after every training batch |


#### Test model
```http
  test(model, test_loader,  criterion)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model. Use 'model.getModel()' to get an instance of our first PyTorch CNN model|
| test_loader | torch.utils.data.dataloader.DataLoader	 | Required: Loader consisting of test data |
| criterion | function | Required: Fucntion used to calculate the loss during test |


#### Plot Accuracy
```http
  printModelTrainTestAccuracy(train_acc, train_losses, test_acc, test_losses)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| train_acc | list | Required: List of training accuracies obtained from 'utils.train()'|
| train_losses |list| Required:  List of train losses obtained from 'utils.train()' |
| test_acc | list | Required:  List of test accuracies obtained from 'utils.test()' |
| test_losses | list | Required: List of test losses obtained from 'utils.test()'|
