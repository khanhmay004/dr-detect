# Tài Liệu Kỹ Thuật Đầy Đủ: Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading

**Mục đích**: Tổng hợp toàn bộ kiến thức kỹ thuật (toán, deep learning, metrics) cần thiết cho dự án từ cơ bản đến nâng cao, bao gồm lý thuyết, công thức và nguồn tham khảo.

---

## MỤC LỤC

1. [Toán Cơ Bản & Xác Suất - Thống Kê](#1-toán-cơ-bản--xác-suất---thống-kê)
2. [Kiến Trúc Neural Network Cơ Bản](#2-kiến-trúc-neural-network-cơ-bản)
3. [Convolutional Neural Networks (CNN)](#3-convolutional-neural-networks-cnn)
4. [ResNet-50 Architecture](#4-resnet-50-architecture)
5. [Attention Mechanisms - CBAM](#5-attention-mechanisms---cbam)
6. [Uncertainty Quantification - Monte Carlo Dropout](#6-uncertainty-quantification---monte-carlo-dropout)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers & Learning Rate Schedulers](#8-optimizers--learning-rate-schedulers)
9. [Regularization Techniques](#9-regularization-techniques)
10. [Data Augmentation](#10-data-augmentation)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Training Techniques](#12-training-techniques)
13. [Model Calibration & Uncertainty Analysis](#13-model-calibration--uncertainty-analysis)
14. [Medical Imaging - Diabetic Retinopathy](#14-medical-imaging---diabetic-retinopathy)
15. [Implementation Framework](#15-implementation-framework)

---

## 1. TOÁN CƠ BẢN & XÁC SUẤT - THỐNG KÊ

### 1.1 Xác Suất Cơ Bản

**Khái niệm cần nắm**:
- **Biến ngẫu nhiên**: Rời rạc (discrete) và liên tục (continuous)
- **Phân phối xác suất**: Gaussian/Normal, Bernoulli, Categorical
- **Kỳ vọng (Expected Value)**:
  ```
  E[X] = Σ x·P(X=x)  (rời rạc)
  E[X] = ∫ x·f(x)dx   (liên tục)
  ```
- **Phương sai (Variance)**:
  ```
  Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
  ```
- **Độ lệch chuẩn (Standard Deviation)**: σ = √Var(X)

**Luật Bayes** (nền tảng cho Bayesian Deep Learning):
```
P(θ|D) = [P(D|θ) · P(θ)] / P(D)
```
Trong đó:
- P(θ|D): Posterior (xác suất tham số sau khi quan sát dữ liệu)
- P(D|θ): Likelihood (khả năng dữ liệu xuất hiện với tham số θ)
- P(θ): Prior (xác suất tham số trước khi quan sát)
- P(D): Evidence (xác suất dữ liệu)

**Ứng dụng trong dự án**[1][2]:
- Monte Carlo dropout xấp xỉ phân phối hậu nghiệm P(θ|D)
- Tính mean và variance của dự đoán qua T=20 forward passes
- Variance đại diện cho epistemic uncertainty

### 1.2 Thống Kê Mô Tả

**Correlation (Tương quan)**:

**Pearson Correlation Coefficient**[82][85][88]:
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² · Σ(yi - ȳ)²]
```
- Phạm vi: -1 ≤ r ≤ 1
- r = 1: Tương quan dương hoàn hảo
- r = 0: Không có tương quan tuyến tính
- r = -1: Tương quan âm hoàn hảo

**Ứng dụng**: Phân tích tương quan giữa prediction error và uncertainty score (mục tiêu: r > 0.6)[11]

### 1.3 Ma Trận & Đại Số Tuyến Tính

**Phép nhân ma trận** (cơ sở cho forward propagation):
```
C = A × B
C[i,j] = Σk A[i,k] · B[k,j]
```

**Gradient** (đạo hàm riêng):
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Chain Rule** (nền tảng cho backpropagation):
```
∂z/∂x = (∂z/∂y) · (∂y/∂x)
```

**Nguồn đọc thêm**:
- 📘 *Probability and Statistics for Machine Learning* - Patrick Hall
- 📘 *Mathematics for Machine Learning* - Marc Peter Deisenroth
- 🌐 Khan Academy: Probability & Statistics
- 🌐 3Blue1Brown: "Essence of Linear Algebra" (YouTube series)

---

## 2. KIẾN TRÚC NEURAL NETWORK CƠ BẢN

### 2.1 Perceptron & Multilayer Perceptron (MLP)

**Perceptron đơn giản**:
```
y = σ(w·x + b)
```
Trong đó:
- x: input vector
- w: weights
- b: bias
- σ: activation function

**Forward Propagation** (tính toán từ input → output)[121][127]:
```
Layer l: 
z⁽ˡ⁾ = W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

### 2.2 Activation Functions

#### ReLU (Rectified Linear Unit)[123][126][129]:
```
ReLU(x) = max(0, x)
Derivative: ReLU'(x) = 1 if x > 0, else 0
```

**Ưu điểm**:
- Giải quyết vanishing gradient problem
- Tính toán nhanh
- Tạo sparsity (một số neurons = 0)

**Nhược điểm**: "Dying ReLU" khi neurons luôn output 0

**Biến thể**:
- **Leaky ReLU**: f(x) = max(αx, x) với α ≈ 0.01
- **Parametric ReLU (PReLU)**: α là learnable parameter
- **ELU**: f(x) = x if x > 0, else α(eˣ - 1)

#### Sigmoid:
```
σ(x) = 1 / (1 + e⁻ˣ)
Derivative: σ'(x) = σ(x)(1 - σ(x))
```
**Vấn đề**: Vanishing gradient khi |x| lớn

#### Tanh:
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```
**Output**: [-1, 1] (zero-centered, tốt hơn sigmoid)

#### Softmax (cho multi-class classification)[142][145][148]:
```
softmax(zi) = exp(zi) / Σⱼ exp(zⱼ)
```
**Tính chất**:
- Output là probability distribution: Σ softmax(zi) = 1
- 0 ≤ softmax(zi) ≤ 1
- Sử dụng ở output layer cho classification

### 2.3 Backpropagation[121][130][136]

**Thuật toán** (tính gradient của loss theo weights):

1. **Forward pass**: Tính output
2. **Compute loss**: L(ŷ, y)
3. **Backward pass**: Tính gradient sử dụng chain rule
   ```
   ∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
   ```
4. **Update weights**: W := W - η·∂L/∂W

**Vanishing/Exploding Gradient Problem**[123][127]:
- **Vanishing**: Gradient → 0 ở layers sâu (do sigmoid/tanh saturation)
- **Exploding**: Gradient → ∞ (do weights lớn)
- **Giải pháp**: ReLU, batch normalization, proper initialization, skip connections

### 2.4 Weight Initialization[182][184][187]

#### Xavier/Glorot Initialization (cho tanh/sigmoid)[182][184]:
```
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))
hoặc
W ~ Normal(0, √(2/(nᵢₙ + nₒᵤₜ)))
```

#### He Initialization (cho ReLU)[182][184]:
```
W ~ Normal(0, √(2/nᵢₙ))
```
Trong đó:
- nᵢₙ: số input neurons
- nₒᵤₜ: số output neurons

**Lý do**: Duy trì variance của activations qua các layers, tránh vanishing/exploding gradients

**Trong dự án**: Sử dụng He initialization với ResNet-50 (vì dùng ReLU)

**Nguồn đọc thêm**:
- 📄 *Understanding the difficulty of training deep feedforward neural networks* - Glorot & Bengio (2010)
- 📄 *Delving Deep into Rectifiers* - He et al. (2015)
- 🌐 CS231n Stanford: Neural Networks Part 2
- 🌐 DeepLearning.AI: Improving Deep Neural Networks

---

## 3. CONVOLUTIONAL NEURAL NETWORKS (CNN)

### 3.1 Convolution Operation

**Công thức tổng quát**[141][144]:
```
S(i,j) = (I * K)(i,j) = ΣₘΣₙ I(i+m, j+n)·K(m,n)
```
Trong đó:
- I: Input image/feature map
- K: Kernel/Filter
- S: Output feature map

**Các thành phần**:

#### Kernel/Filter[141][153]:
- Ma trận nhỏ (thường 3×3, 5×5, 7×7)
- Learnable parameters (học qua backpropagation)
- Mỗi kernel detect một feature cụ thể (edges, textures, patterns)

**Ví dụ Edge Detection Kernel**:
```
Vertical Edge:        Horizontal Edge:
[-1  0  1]           [-1 -1 -1]
[-1  0  1]           [ 0  0  0]
[-1  0  1]           [ 1  1  1]
```

#### Feature Map[144]:
- Output của convolution operation
- Mỗi filter tạo ra 1 feature map
- Số lượng feature maps = số filters
- Kích thước feature map:
  ```
  Output_size = (Input_size - Kernel_size + 2·Padding) / Stride + 1
  ```

### 3.2 Hyperparameters

**Stride**: Bước nhảy khi slide kernel
- Stride = 1: Kernel di chuyển 1 pixel mỗi lần
- Stride lớn → output size nhỏ hơn

**Padding**: Thêm pixels vào biên
- Valid (no padding): Output size giảm
- Same padding: Output size = Input size (khi stride=1)
- Padding = (K-1)/2 để maintain size

**Dilation**: Tăng receptive field không tăng params
```
Dilated convolution = convolution với "holes" trong kernel
```

### 3.3 Pooling Layers

**Max Pooling** (phổ biến nhất):
```
Output(i,j) = max(Input trong window)
```
- Thường 2×2 window, stride=2 → giảm kích thước 1/2

**Average Pooling**:
```
Output(i,j) = mean(Input trong window)
```

**Global Average Pooling (GAP)**:
- Average toàn bộ feature map thành 1 số
- Thay thế fully connected layers
- Giảm overfitting

**Lợi ích pooling**:
- Giảm spatial dimensions → giảm computation
- Translation invariance
- Tăng receptive field

### 3.4 Receptive Field

**Định nghĩa**: Vùng trong input image ảnh hưởng đến 1 neuron ở layer sau

**Tính receptive field**:
```
RFₗ = RFₗ₋₁ + (Kₗ - 1) × Πᵢ₌₁ˡ⁻¹ Sᵢ
```
Trong đó:
- RFₗ: Receptive field tại layer l
- Kₗ: Kernel size tại layer l
- Sᵢ: Stride tại layer i

**Trong dự án**: ResNet-50 có receptive field rất lớn (∼224×224) để capture global context

**Nguồn đọc thêm**:
- 📄 *ImageNet Classification with Deep CNNs* - Krizhevsky et al. (AlexNet, 2012)
- 📄 *Visualizing and Understanding CNNs* - Zeiler & Fergus (2014)
- 🌐 CS231n: Convolutional Neural Networks
- 🎥 Andrew Ng: CNNs (Coursera Deep Learning Specialization)

---

## 4. RESNET-50 ARCHITECTURE

### 4.1 Residual Learning

**Vấn đề degradation**: Mạng càng sâu, accuracy càng giảm (không phải do overfitting)

**Giải pháp - Skip Connections**[21][24][39]:
```
y = F(x, {Wᵢ}) + x
```
Trong đó:
- x: Input
- F(x): Learned residual mapping
- y: Output

**Lý thuyết**: Thay vì học H(x), mạng học residual F(x) = H(x) - x
- Nếu identity mapping là optimal, F(x) chỉ cần học → 0 (dễ hơn)
- Skip connection cho phép gradient flow trực tiếp

### 4.2 ResNet-50 Architecture

**Cấu trúc tổng thể**[21][27][30]:
```
Input (224×224×3)
    ↓
Conv1: 7×7, 64, stride 2
    ↓
MaxPool: 3×3, stride 2
    ↓
Conv2_x: [1×1,64 / 3×3,64 / 1×1,256] × 3 blocks
    ↓
Conv3_x: [1×1,128 / 3×3,128 / 1×1,512] × 4 blocks
    ↓
Conv4_x: [1×1,256 / 3×3,256 / 1×1,1024] × 6 blocks
    ↓
Conv5_x: [1×1,512 / 3×3,512 / 1×1,2048] × 3 blocks
    ↓
Global Average Pooling
    ↓
Fully Connected (1000 classes for ImageNet)
```

**Bottleneck Block** (sử dụng trong ResNet-50):
```
Input
  ↓
1×1 Conv (reduce dimensions) → BN → ReLU
  ↓
3×3 Conv (spatial convolution) → BN → ReLU
  ↓
1×1 Conv (restore dimensions) → BN
  ↓
Add with Input (skip connection)
  ↓
ReLU
```

**Tổng số layers**: 50 convolutional layers
**Tổng số parameters**: ~25.6 million

### 4.3 Transfer Learning với ImageNet

**Pre-trained weights**[101][107][110]:
- ResNet-50 được train trên ImageNet (1.2M images, 1000 classes)
- Học được general features: edges, textures, shapes, objects
- Lower layers: low-level features (edges, colors)
- Higher layers: high-level features (object parts, semantics)

**Fine-tuning strategy** trong dự án:
1. Load pretrained ResNet-50 weights
2. Freeze hoặc fine-tune backbone (tuỳ dataset size)
3. Replace final FC layer: 1000 classes → 5 classes (DR grades)
4. Train với learning rate nhỏ (1e-4) nếu fine-tune toàn bộ

**Lợi ích**:
- Faster convergence
- Better performance với limited data (APTOS chỉ 3662 images)
- Avoid overfitting

### 4.4 Batch Normalization trong ResNet

**Công thức**[122][128][131]:
```
Normalize: x̂ = (x - μ_B) / √(σ²_B + ε)
Scale & Shift: y = γ·x̂ + β
```
Trong đó:
- μ_B, σ²_B: Mean và variance của mini-batch
- γ, β: Learnable parameters
- ε: Small constant (1e-5) for numerical stability

**Lợi ích**:
- Giảm Internal Covariate Shift
- Cho phép learning rate cao hơn
- Regularization effect
- Faster convergence

**Nguồn đọc thêm**:
- 📄 *Deep Residual Learning for Image Recognition* - He et al. (2015) ⭐
- 📄 *Identity Mappings in Deep Residual Networks* - He et al. (2016)
- 🌐 PyTorch ResNet Documentation
- 🎥 Arxiv Insights: "ResNet Explained" (YouTube)

---

## 5. ATTENTION MECHANISMS - CBAM

### 5.1 Attention Concept

**Mục đích**: Mô hình tự học "chú ý" vào vùng/features quan trọng

**Hai dạng attention**:
1. **Channel Attention**: "What" to focus - features nào quan trọng
2. **Spatial Attention**: "Where" to focus - vùng nào trong image quan trọng

### 5.2 CBAM (Convolutional Block Attention Module)[22][25][28]

**Architecture tổng quát**:
```
Input Feature F
    ↓
Channel Attention Module
    ↓
F' = Mc(F) ⊗ F
    ↓
Spatial Attention Module
    ↓
F'' = Ms(F') ⊗ F'
    ↓
Output Feature F''
```

#### 5.2.1 Channel Attention Module

**Công thức**[22][25]:
```
Mc(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
```

**Chi tiết**:
1. **Aggregate spatial information**:
   - Average Pooling: F^c_avg ∈ R^(C×1×1)
   - Max Pooling: F^c_max ∈ R^(C×1×1)

2. **Shared MLP** (2 FC layers):
   ```
   FC1: C → C/r (reduction ratio r=16)
   ReLU
   FC2: C/r → C
   ```

3. **Combine**:
   ```
   Mc = Sigmoid(MLP(F^c_avg) + MLP(F^c_max))
   ```

**Intuition**: Channels có activation cao (qua AvgPool và MaxPool) được emphasize

#### 5.2.2 Spatial Attention Module

**Công thức**[22][25]:
```
Ms(F) = σ(f^(7×7)([AvgPool(F); MaxPool(F)]))
```

**Chi tiết**:
1. **Aggregate channel information**:
   - AvgPool along channel axis: F^s_avg ∈ R^(1×H×W)
   - MaxPool along channel axis: F^s_max ∈ R^(1×H×W)

2. **Concatenate**:
   ```
   F^s = Concat(F^s_avg, F^s_max) ∈ R^(2×H×W)
   ```

3. **Convolution**:
   ```
   Ms = Sigmoid(Conv^(7×7)(F^s))
   ```

**Intuition**: Vị trí nào có response mạnh (qua pooling) được attend

### 5.3 CBAM Integration với ResNet

**Trong dự án**: Insert CBAM sau mỗi ResNet block
```
ResNet Block
    ↓
Residual Connection
    ↓
CBAM Module
    ↓
Next Block
```

**Vị trí insert**: Sau conv2_x, conv3_x, conv4_x, conv5_x (4 vị trí)

**Parameters overhead**: Minimal (~0.5% increase)
**Computation overhead**: Negligible

**Nguồn đọc thêm**:
- 📄 *CBAM: Convolutional Block Attention Module* - Woo et al. (ECCV 2018) ⭐
- 📄 *Squeeze-and-Excitation Networks* - Hu et al. (CVPR 2018)
- 🌐 CBAM GitHub: https://github.com/Jongchan/attention-module
- 🎥 CBAM Paper Explained (YouTube)

---

## 6. UNCERTAINTY QUANTIFICATION - MONTE CARLO DROPOUT

### 6.1 Types of Uncertainty[81][84][90]

#### Epistemic Uncertainty (Model Uncertainty):
- **Định nghĩa**: Uncertainty do thiếu knowledge/data
- **Có thể giảm**: Bằng cách thu thập thêm data hoặc cải thiện model
- **Nguồn gốc**: 
  - Limited training data
  - Model architecture không phù hợp
  - Insufficient model capacity

#### Aleatoric Uncertainty (Data Uncertainty):
- **Định nghĩa**: Uncertainty inherent trong data
- **Không thể giảm**: Là noise tự nhiên
- **Nguồn gốc**:
  - Sensor noise
  - Occlusion
  - Ambiguous cases (e.g., borderline DR grades)

### 6.2 Bayesian Neural Networks

**Bayesian approach**:
```
P(y|x, D) = ∫ P(y|x, W) · P(W|D) dW
```
Trong đó:
- P(W|D): Posterior distribution over weights
- P(y|x, W): Likelihood

**Vấn đề**: Intractable integral (không tính được chính xác)

**Giải pháp**: Variational inference - approximate P(W|D) với q_θ(W)

### 6.3 Monte Carlo Dropout[23][26][29]

**Ý tưởng chính**[23]:
- Dropout during training: Regularization
- **Dropout during inference**: Approximate Bayesian inference

**Algorithm**:
```
For t = 1 to T:
    1. Enable dropout at inference
    2. Forward pass: ŷₜ = f(x; Wₜ)  (Wₜ sampled via dropout)
    3. Store prediction ŷₜ

Predictive mean: ŷ_mean = (1/T) Σₜ ŷₜ
Predictive entropy: H(ŷ_mean) = -Σ_c ŷ_mean_c · log(ŷ_mean_c)
Confidence: max(ŷ_mean)
```

**Trong dự án**:
- T = 20 forward passes
- Dropout rate p = 0.5 trong classification head
- Final prediction: ŷ = argmax(ŷ_mean)
- Uncertainty score: H (Predictive entropy)

### 6.4 Implementation Details

**PyTorch code**:
```python
model.train()  # Enable dropout!
predictions = []

with torch.no_grad():
    for _ in range(T):
        output = torch.softmax(model(input), dim=1)
        predictions.append(output)

predictions = torch.stack(predictions)  # Shape: [T, batch, classes]
mean_pred = predictions.mean(dim=0)     # Predictive mean

# Calculate Predictive Entropy (Uncertainty)
epsilon = 1e-12
entropy = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=1)

# Calculate Confidence
confidence = mean_pred.max(dim=1)[0]
```

### 6.5 Uncertainty-Error Correlation Analysis[11]

**Mục tiêu**: Validate uncertainty quantification

**Phương pháp**:
1. Tính prediction error:
   ```
   error = |ŷ - y_true|  (cho ordinal classification)
   ```
2. Tính Pearson correlation: ρ(error, uncertainty)
3. Target: ρ > 0.6 (strong positive correlation)

**Ý nghĩa**:
- High correlation → model biết khi nào nó uncertain
- Low correlation → uncertainty estimation không reliable

**Nguồn đọc thêm**:
- 📄 *Dropout as a Bayesian Approximation* - Gal & Ghahramani (ICML 2016) ⭐
- 📄 *Uncertainty in Deep Learning* - Yarin Gal (PhD Thesis, 2016)
- 📄 *What Uncertainties Tell You in Bayesian Deep Learning* - Kendall & Gal (2017)
- 🌐 Yarin Gal's Blog: http://www.cs.ox.ac.uk/people/yarin.gal/website/blog.html

---

## 7. LOSS FUNCTIONS

### 7.1 Cross-Entropy Loss

#### Binary Cross-Entropy[181][183]:
```
BCE = -(1/N) Σᵢ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]
```
- yᵢ ∈ {0, 1}: True label
- pᵢ ∈ [0, 1]: Predicted probability

#### Categorical Cross-Entropy[181][186]:
```
CCE = -(1/N) Σᵢ Σⱼ yᵢⱼ·log(pᵢⱼ)
```
- yᵢⱼ: One-hot encoded true label
- pᵢⱼ: Predicted probability for class j

**Gradient**:
```
∂CCE/∂zᵢ = pᵢ - yᵢ  (với softmax output)
```
→ Clean gradient, dễ optimize

### 7.2 Focal Loss[41][47][50]

**Motivation**: Giải quyết class imbalance

**Công thức**[41]:
```
FL(pₜ) = -αₜ(1 - pₜ)^γ log(pₜ)
```
Trong đó:
```
pₜ = p    if y=1
pₜ = 1-p  if y=0
```

**Hyperparameters**:
- **γ (focusing parameter)**: Thường 2
  - γ = 0 → Focal Loss = Cross-Entropy
  - γ tăng → down-weight easy examples nhiều hơn
- **α (class weights)**: Balance class frequency
  ```
  αc = N / (K · Nc)  hoặc custom weights
  ```

**Intuition**:
- Easy examples (pₜ cao): (1-pₜ)^γ ≈ 0 → loss nhỏ
- Hard examples (pₜ thấp): (1-pₜ)^γ ≈ 1 → loss lớn
- Model focus vào hard/misclassified examples

**Trong dự án** (APTOS imbalance):
```python
Class distribution:
  Grade 0 (No DR): 1805 images
  Grade 1 (Mild): 370
  Grade 2 (Moderate): 999
  Grade 3 (Severe): 193
  Grade 4 (Proliferative): 295

Class weights α = [0.1, 0.2, 0.2, 0.25, 0.25]
γ = 2.0
```

**Nguồn đọc thêm**:
- 📄 *Focal Loss for Dense Object Detection* - Lin et al. (ICCV 2017) ⭐
- 🌐 Understanding Focal Loss: https://amaarora.github.io/2020/06/29/FocalLoss.html

---

## 8. OPTIMIZERS & LEARNING RATE SCHEDULERS

### 8.1 Gradient Descent Variants

#### Stochastic Gradient Descent (SGD):
```
θₜ₊₁ = θₜ - η·∇L(θₜ)
```

#### SGD with Momentum:
```
vₜ = β·vₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ - η·vₜ
```
- β ≈ 0.9: Momentum coefficient

#### Adam (Adaptive Moment Estimation):
```
mₜ = β₁·mₜ₋₁ + (1-β₁)·∇L(θₜ)  (1st moment)
vₜ = β₂·vₜ₋₁ + (1-β₂)·∇L(θₜ)²  (2nd moment)

m̂ₜ = mₜ / (1 - β₁ᵗ)  (bias correction)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ₊₁ = θₜ - η · m̂ₜ / (√v̂ₜ + ε)
```
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (defaults)

### 8.2 AdamW (Adam with Decoupled Weight Decay)[42][45][48]

**Vấn đề với Adam + L2 regularization**:
- L2 reg thêm λ·||W||² vào loss
- Gradient: ∇L + λ·W
- Trong Adam, gradient này đi qua adaptive scaling → weight decay bị couple với learning rate

**AdamW solution**[42]:
```
mₜ = β₁·mₜ₋₁ + (1-β₁)·∇L(θₜ)
vₜ = β₂·vₜ₋₁ + (1-β₂)·∇L(θₜ)²

m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ₊₁ = θₜ - η · [m̂ₜ / (√v̂ₜ + ε) + λ·θₜ]
```
**Key difference**: Weight decay λ·θₜ applied AFTER adaptive update

**Tuning strategy**[48]:
- Khi double learning rate → halve weight decay
- Maintain α·λ constant (α = learning rate, λ = weight decay)

**Trong dự án**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,  # Typical value
    betas=(0.9, 0.999)
)
```

### 8.3 Cosine Annealing Scheduler[43][46][49]

**Công thức**[43]:
```
ηₜ = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
```
Trong đó:
- ηₜ: Learning rate tại epoch t
- η_max: Initial learning rate
- η_min: Minimum learning rate (thường 0)
- T: Total epochs

**Lợi ích**:
- Smooth decay (không abrupt như step decay)
- Large learning rate ban đầu → fast exploration
- Small learning rate cuối → fine-tuning
- Restart variants (SGDR) để escape local minima

**PyTorch implementation**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Số epochs
    eta_min=0   # Min LR
)
```

**Nguồn đọc thêm**:
- 📄 *Decoupled Weight Decay Regularization* - Loshchilov & Hutter (ICLR 2019) ⭐
- 📄 *SGDR: Stochastic Gradient Descent with Warm Restarts* - Loshchilov & Hutter (2017)
- 🌐 AdamW Paper: https://arxiv.org/abs/1711.05101

---

## 9. REGULARIZATION TECHNIQUES

### 9.1 L2 Regularization / Weight Decay[143][146][149]

**Công thức**:
```
L_total = L_original + λ · Σᵢ wᵢ²
```

**Gradient**:
```
∂L_total/∂w = ∂L_original/∂w + 2λw
```

**Weight update**:
```
w := w - η(∂L_original/∂w + 2λw)
  = (1 - 2ηλ)w - η·∂L_original/∂w
```
→ Weight decay: (1 - 2ηλ) < 1

**Lợi ích**:
- Encourage smaller weights → simpler model
- Prevent overfitting
- Improve generalization

**Trong dự án**: λ = 1e-2 trong AdamW

### 9.2 Dropout

**Training time**:
```
During training:
  hᵢ = 0 with probability p
  hᵢ = hᵢ / (1-p) with probability (1-p)
```

**Inference time** (standard):
```
Use all neurons, no dropout
```

**MC Dropout** (dự án này):
```
Keep dropout ON during inference for uncertainty quantification
```

**Trong dự án**:
- Dropout rate p = 0.5 ở classification head
- Permanent dropout → enables MC sampling

### 9.3 Data Augmentation

**Mục đích**: Increase data diversity → reduce overfitting

**Xem Section 10 để biết chi tiết**

### 9.4 Early Stopping[102][105][108]

**Algorithm**:
```
best_loss = ∞
patience_counter = 0
patience = 15  # Trong dự án

For each epoch:
    val_loss = validate(model, val_data)
    
    if val_loss < best_loss:
        best_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        stop training
        load best_model()
```

**Monitor metric**: Validation QWK (Quadratic Weighted Kappa) trong dự án

**Nguồn đọc thêm**:
- 📘 *Deep Learning* - Goodfellow et al. (Chapter 7: Regularization)
- 🌐 CS231n: Regularization for Neural Networks

---

## 10. DATA AUGMENTATION

### 10.1 Traditional Augmentations

**Geometric**:
- Rotation: ±180° (hoặc ±360° cho fundus images - symmetrical)
- Horizontal/Vertical Flip
- Affine transformations
- Elastic deformations

**Photometric**:
- Brightness adjustment
- Contrast adjustment
- Color jitter (hue, saturation)
- Gaussian noise

### 10.2 Advanced Augmentations

#### MixUp[61][64][76]

**Công thức**[61]:
```
x̃ = λ·xᵢ + (1-λ)·xⱼ
ỹ = λ·yᵢ + (1-λ)·yⱼ
```
Trong đó:
- λ ~ Beta(α, α), thường α = 0.2 hoặc 0.4
- (xᵢ, yᵢ), (xⱼ, yⱼ): Hai samples ngẫu nhiên

**Lợi ích**:
- Regularization mạnh
- Smooth decision boundaries
- Better calibration
- Robust to adversarial examples

**Ví dụ**:
```
Image 1: DR Grade 2, λ = 0.7
Image 2: DR Grade 3, λ = 0.3
Mixed image: 0.7×Img1 + 0.3×Img2
Mixed label: [0, 0, 0.7, 0.3, 0] (soft label)
```

#### CutMix, GridMask, v.v.

### 10.3 Albumentations Library[163][172][175]

**Ưu điểm**:
- Fast (built on OpenCV)
- Flexible API
- Support medical imaging (16-bit images)
- Coordinate-aware (masks, bboxes, keypoints)

**Example pipeline cho DR**:
```python
import albumentations as A

transform = A.Compose([
    A.Resize(224, 224),
    A.Rotate(limit=180, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.3
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

### 10.4 Ben Graham Preprocessing[2][17]

**Method** (dùng trong Kaggle DR competition):
1. **Circular crop**: Remove black borders
2. **Local color normalization**:
   - Subtract local average color
   - Divide by local standard deviation
   - Làm cho blood vessels rõ hơn
3. **Resize** to target size (224×224)

**Nguồn đọc thêm**:
- 📄 *mixup: Beyond Empirical Risk Minimization* - Zhang et al. (ICLR 2018)
- 🌐 Albumentations Docs: https://albumentations.ai/docs/
- 🏆 Kaggle DR Competition Winning Solutions

---

## 11. EVALUATION METRICS

### 11.1 Classification Basics

#### Confusion Matrix[83][89][185]

**Binary**:
```
                 Predicted
               Pos    Neg
Actual  Pos    TP     FN
        Neg    FP     TN
```

**Multiclass** (K classes):
```
Confusion Matrix C ∈ R^(K×K)
C[i,j] = số samples thực tế class i, predicted class j
```

#### Accuracy:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Precision (Positive Predictive Value):
```
Precision = TP / (TP + FP)
```
"Trong các dự đoán positive, bao nhiêu % đúng?"

#### Recall (Sensitivity, True Positive Rate):
```
Recall = TP / (TP + FN)
```
"Trong các actual positive, bao nhiêu % được detect?"

#### Specificity (True Negative Rate):
```
Specificity = TN / (TN + FP)
```
"Trong các actual negative, bao nhiêu % được detect?"

#### F1 Score:
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
Harmonic mean của Precision và Recall

**Multiclass metrics**[185]:
- **Macro-average**: Average metric across classes (treat all classes equally)
  ```
  Macro-F1 = (1/K) Σₖ F1ₖ
  ```
- **Micro-average**: Pool all TP, FP, FN, then compute (give more weight to large classes)
- **Weighted-average**: Weight by class frequency

### 11.2 Quadratic Weighted Kappa (QWK)[7][10][18]

**Sử dụng**: Ordinal classification (classes có thứ tự)

**Công thức**[18]:
```
κ = 1 - (Σᵢⱼ wᵢⱼ·Oᵢⱼ) / (Σᵢⱼ wᵢⱼ·Eᵢⱼ)
```
Trong đó:
- Oᵢⱼ: Observed agreement (confusion matrix)
- Eᵢⱼ: Expected agreement (if by chance)
  ```
  Eᵢⱼ = (Σₖ Oᵢₖ) · (Σₖ Oₖⱼ) / N
  ```
- wᵢⱼ: Weight matrix (quadratic weights)
  ```
  wᵢⱼ = (i - j)² / (K - 1)²
  ```

**Interpretation**:
- κ = 1: Perfect agreement
- κ = 0: Random agreement
- κ < 0: Worse than random

**Trong dự án**: Primary metric cho DR grading
- DR grades: 0, 1, 2, 3, 4 (ordinal)
- Predicting Grade 2 khi truth là Grade 3 ít severe hơn predicting Grade 0

### 11.3 ROC & AUC

**ROC Curve** (Receiver Operating Characteristic):
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Recall)
- Plot ở mọi classification thresholds

**AUC** (Area Under Curve):
- Phạm vi: 0 to 1
- AUC = 1: Perfect classifier
- AUC = 0.5: Random classifier

**Multiclass ROC**: One-vs-Rest (OvR) approach

**Binary Referable AUC (AUC-Ref)**:
- Đo lường khả năng phân biệt giữa Referable DR (Grades ≥ 2) và Non-Referable DR (Grades < 2).
- Là một metric quan trọng trong lâm sàng để quyết định bệnh nhân có cần chuyển tuyến Specialist hay không.

### 11.4 Expected Calibration Error (ECE)[162][165][168]

**Mục đích**: Đo "calibration" - predicted probabilities có match với actual frequencies không?

**Công thức**[165]:
```
ECE = Σₘ (|Bₘ|/N) · |acc(Bₘ) - conf(Bₘ)|
```
Trong đó:
- Chia predictions thành M bins theo confidence
- Bₘ: Set of samples trong bin m
- acc(Bₘ): Accuracy trong bin m
- conf(Bₘ): Average confidence trong bin m

**Ví dụ**:
- Bin [0.7, 0.8]: 100 samples, accuracy = 0.65, avg confidence = 0.75
- Contribution: (100/N) · |0.65 - 0.75| = (100/N) · 0.1

**Perfect calibration**: ECE = 0

**Trong dự án**: Evaluate calibration sau temperature scaling

### 11.5 Brier Score

**Mục đích**: Đo lường tổng thể độ chính xác của các dự đoán xác suất (probability predictions).

**Công thức**:
```
BS = (1/N) Σᵢ Σₖ (fᵢₖ - oᵢₖ)²
```
Trong đó:
- `fᵢₖ`: Predicted probability cho sample `i` thuộc class `k`
- `oᵢₖ`: True label (one-hot encoded), bằng 1 nếu đúng class `k`, bằng 0 nếu ngược lại
- Phạm vi: 0 đến 2 (với multi-class), càng gần 0 càng tốt. Brier Score phạt những dự đoán "tự tin nhưng sai".

**Trong dự án**: Đánh giá calibration và độ tin cậy thực tế ở mức xác suất, sử dụng trong file evaluate.py.

### 11.6 Predictive Entropy (H)

**Mục đích**: Lượng hóa mức độ không chắc chắn (uncertainty) của prediction thông qua entropy của phân phối xác suất dự đoán (Predictive Uncertainty).

**Công thức**:
```
H(ȳ) = -Σ_c ȳ_c · log(ȳ_c)
```
Trong đó:
- `ȳ`: Predictive mean probability vector, được tính bằng trung bình qua T=20 lần chạy MC Dropout.
- `ȳ_c`: Xác suất dự đoán trung bình của class `c`.

**Interpretation**:
- H ≈ 0: Model rất tự tin (tất cả các lần chạy MC đều đồng thuận vào một class).
- H > threshold (ví dụ > 1.0): Model không chắc chắn -> Cờ (flag) manual review bởi bác sĩ (Human-in-the-loop).

### 11.7 Referable Sensitivity / Specificity

**Mục đích**: Đánh giá hiệu suất lâm sàng cho việc sàng lọc (screening performance). Ngưỡng Referable DR được đặt ở Grade ≥ 2.
- **Referable Sensitivity**: Tỷ lệ phát hiện đúng các ca bệnh Referable DR (hạn chế False Negative vì rủi ro bệnh nhân bị mất thị lực).
- **Referable Specificity**: Tỷ lệ xác định đúng các ca bệnh Non-Referable DR (hạn chế False Positive để giảm tải cho bác sĩ chuyên khoa).

### 11.8 Correlation Analysis

**Pearson Correlation** (xem Section 1.2):
```
r = Cov(X, Y) / (σₓ · σᵧ)
```

**Trong dự án**: Correlation giữa prediction error và uncertainty
- Tính cho mỗi sample: error = |predicted_grade - true_grade|
- Uncertainty score = Predictive Entropy (H) từ MC dropout
- Target: r > 0.6

**Nguồn đọc thêm**:
- 📄 *Cohen's Kappa Statistic* - Cohen (1960)
- 📄 *On Calibration of Modern Neural Networks* - Guo et al. (ICML 2017)
- 🌐 Scikit-learn Metrics Documentation
- 🌐 Kaggle: "Understanding Quadratic Weighted Kappa"

---

## 12. TRAINING TECHNIQUES

### 12.1 Train/Validation/Test Split

**Standard split**:
- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%

**Trong dự án** (APTOS):
- Train: 80% (2930 images)
- Validation: 10% (366 images)
- Internal test: 10% (366 images)
- External test: Messidor-2 (1748 images, zero exposure)

### 12.2 Stratified Split[103][106][109]

**Mục đích**: Maintain class distribution trong mỗi split

**Vấn đề với random split**:
- Với imbalanced data, có thể một class bị thiếu ở validation
- Model không học hoặc evaluate đúng minority class

**Stratified K-Fold**[103]:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Each fold preserves class distribution
```

**Trong dự án**: Patient-level stratified split (tránh data leakage nếu cùng bệnh nhân có nhiều ảnh)

### 12.3 Training Loop

**Pseudocode**:
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch.images)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch.images)
            val_loss = criterion(outputs, batch.labels)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 12.4 Mixed Precision Training

**FP16 (half precision)**:
- Faster computation (2-3x speedup on modern GPUs)
- Less memory usage (2x reduction)
- Slight accuracy trade-off (thường negligible)

**PyTorch implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # Mixed precision
        outputs = model(batch.images)
        loss = criterion(outputs, batch.labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 12.5 Gradient Clipping

**Mục đích**: Prevent exploding gradients

**Công thức**:
```
if ||g|| > threshold:
    g = g · (threshold / ||g||)
```

**PyTorch**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Nguồn đọc thêm**:
- 📘 *Deep Learning* - Goodfellow et al. (Chapter 8: Optimization)
- 🌐 PyTorch Training Tutorial
- 🌐 Mixed Precision Training Guide (NVIDIA)

---

## 13. MODEL CALIBRATION & UNCERTAINTY ANALYSIS

### 13.1 Calibration Concept[66]

**Định nghĩa**: Model is calibrated if
```
P(Y = ŷ | Confidence = p) = p
```
"Khi model nói 70% confident, thì đúng 70% thời gian"

**Vấn đề**: Modern neural networks often miscalibrated
- Overconfident predictions
- Deep networks, large capacity → worse calibration

### 13.2 Temperature Scaling[63][66][72]

**Simplest post-hoc calibration method**

**Công thức**[66]:
```
Before: p = softmax(z) = exp(zᵢ) / Σⱼ exp(zⱼ)

After: p = softmax(z/T) = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)
```
Trong đó:
- T: Temperature parameter (learnable scalar)
- T > 1: Soften probabilities (less confident)
- T < 1: Sharpen probabilities (more confident)

**Learning T**:
1. Freeze model weights
2. Use validation set
3. Optimize T to minimize NLL (Negative Log-Likelihood)
```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature

# Optimize on validation set
optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
criterion = nn.CrossEntropyLoss()

def eval():
    optimizer.zero_grad()
    loss = criterion(temp_model(logits), labels)
    loss.backward()
    return loss

optimizer.step(eval)
```

**Lợi ích**:
- 2 lines of code
- No retraining needed
- Preserves accuracy (chỉ re-scale probabilities)

### 13.3 Reliability Diagram

**Plot**:
- X-axis: Predicted confidence (binned)
- Y-axis: Observed accuracy
- Perfect calibration: y = x line

**Trong dự án**: Visualize before/after temperature scaling

### 13.4 Uncertainty-Aware Inference[11]

**Pipeline**:
```
For each test image:
    1. MC Dropout: T=20 forward passes
    2. Compute mean prediction & variance
    3. Final prediction: argmax(mean)
    4. Uncertainty score: total variance
    5. If uncertainty > threshold:
           Flag for expert review
       Else:
           Accept prediction
```

**Threshold selection**:
- Plot error rate vs uncertainty threshold
- Choose threshold that catches high % of errors
- Trade-off: Sensitivity vs workload (% flagged for review)

**Example**:
- Threshold = 90th percentile of uncertainty
- Catches 75% of errors
- Flags 10% of cases for review

### 13.5 Attention Visualization - Grad-CAM[62][65][68]

**Mục đích**: "Where is the model looking?"

**Algorithm**[62]:
1. Forward pass: Get class score yᶜ and final conv feature maps Aᵏ
2. Backward pass: Compute gradients
   ```
   ∂yᶜ/∂Aᵏ
   ```
3. Global average pooling of gradients:
   ```
   αₖᶜ = (1/Z) Σᵢ Σⱼ ∂yᶜ/∂Aᵢⱼᵏ
   ```
4. Weighted combination:
   ```
   Lᶜ = ReLU(Σₖ αₖᶜ · Aᵏ)
   ```
5. Upsample to input size

**Output**: Heatmap showing important regions

**Trong dự án**:
- Verify model attends to lesions (microaneurysms, hemorrhages)
- Compare correct vs incorrect predictions
- Clinical interpretation

**Nguồn đọc thêm**:
- 📄 *Grad-CAM: Visual Explanations from Deep Networks* - Selvaraju et al. (ICCV 2017) ⭐
- 📄 *On Calibration of Modern Neural Networks* - Guo et al. (ICML 2017)
- 🌐 Grad-CAM GitHub: https://github.com/ramprs/grad-cam

---

## 14. MEDICAL IMAGING - DIABETIC RETINOPATHY

### 14.1 Clinical Background

**Diabetic Retinopathy (DR)**[161][164]:
- Biến chứng của diabetes mellitus
- Leading cause of blindness trong working-age adults
- Progression: No DR → Mild → Moderate → Severe → Proliferative DR

**DR Grading (ETDRS scale)**:
- **Grade 0 (No DR)**: No abnormalities
- **Grade 1 (Mild NPDR)**: Microaneurysms only
- **Grade 2 (Moderate NPDR)**: More than just microaneurysms but less than Severe
- **Grade 3 (Severe NPDR)**: 
  - 4-2-1 rule: Any of:
    - Severe hemorrhages in 4 quadrants
    - Venous beading in 2+ quadrants
    - IRMA in 1+ quadrant
- **Grade 4 (Proliferative DR - PDR)**: 
  - Neovascularization
  - Vitreous hemorrhage

### 14.2 Fundus Imaging

**Fundus Camera**:
- Non-mydriatic retinal camera
- Field of view: 30-50° (macula-centered)
- Color image: 3 channels (RGB)

**Key Lesions** (detectable in fundus images)[161][164]:

#### Red Lesions:
1. **Microaneurysms (MA)**: 
   - Earliest clinical sign of DR
   - Small red dots (<125 μm)
   - Capillary wall weakness

2. **Hemorrhages**:
   - Dot & blot hemorrhages (deep layers)
   - Flame-shaped hemorrhages (superficial layers)
   - Darker red than MA

#### Bright Lesions:
3. **Hard Exudates**: Yellow lipid deposits
4. **Cotton-wool spots**: Nerve fiber layer infarcts

#### Advanced Signs:
5. **Venous beading**: Irregular vein diameter
6. **IRMA** (Intraretinal Microvascular Abnormalities)
7. **Neovascularization**: New abnormal blood vessels

### 14.3 Image Quality Issues

**Challenges**[83]:
- Variable illumination
- Different camera models/vendors
- Focus/blur
- Image artifacts
- Patient movement

**Preprocessing** (Ben Graham method)[2]:
- Circular crop
- Local color normalization
- Contrast enhancement

### 14.4 Datasets

#### APTOS 2019[Project]:
- 3,662 images
- 5-class grading
- Severe class imbalance
- Multiple camera models
- Public (Kaggle)

#### Messidor-2[Project]:
- 1,748 images
- Different acquisition protocol
- External validation
- Public: http://www.adcis.net/en/third-party/messidor2/

**Importance of external validation**:
- Test generalization to different population/equipment
- Avoid overfitting to APTOS quirks
- Zero exposure ensures unbiased evaluation

### 14.5 Clinical Deployment Considerations

**Safety requirements**:
- High sensitivity for referable DR (Grades 3-4)
- Minimize false negatives (missing severe cases)
- Uncertainty quantification for borderline cases
- Human-in-the-loop for high-uncertainty predictions

**FDA regulatory pathway**:
- Class II/III medical device
- Clinical validation studies
- Performance targets: Sensitivity >85%, Specificity >90%

**Nguồn đọc thêm**:
- 📄 *Diabetic Retinopathy Grading using Deep Learning* - Gulshan et al. (JAMA 2016)
- 📄 *Development and Validation of a Deep Learning Algorithm for DR* - Ting et al. (JAMA 2017)
- 🌐 NHS Diabetic Eye Screening Programme Guidelines
- 🌐 ETDRS Severity Scale: https://www.aao.org/education/clinical-statement/diabetic-retinopathy-ppp

---

## 15. IMPLEMENTATION FRAMEWORK

### 15.1 PyTorch Ecosystem

**Core Libraries**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
```

**Dataset class**:
```python
class DRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### 15.2 Model Definition

**Uncertainty-Aware Attention CNN**:
```python
import torch.nn as nn
from torchvision.models import resnet50

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x

class UncertaintyAwareResNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        # Load pretrained ResNet-50
        resnet = resnet50(pretrained=True)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet blocks with CBAM
        self.layer1 = nn.Sequential(resnet.layer1, CBAM(256))
        self.layer2 = nn.Sequential(resnet.layer2, CBAM(512))
        self.layer3 = nn.Sequential(resnet.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(resnet.layer4, CBAM(2048))
        
        self.avgpool = resnet.avgpool
        
        # Classification head with permanent dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Always active!
        x = self.fc(x)
        
        return x
    
    def mc_dropout_predict(self, x, T=20):
        """Monte Carlo Dropout prediction"""
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(T):
                pred = self(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)  # [T, batch, classes]
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean_pred, variance
```

### 15.3 Training Script Skeleton

```python
# Hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-2
patience = 15

# Model
model = UncertaintyAwareResNet(num_classes=5)
model = model.cuda()

# Loss & Optimizer
class_weights = torch.tensor([0.1, 0.2, 0.2, 0.25, 0.25]).cuda()
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
best_qwk = -1
patience_counter = 0

for epoch in range(num_epochs):
    # Train
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    qwk = quadratic_weighted_kappa(val_labels, val_preds)
    
    # Early stopping
    if qwk > best_qwk:
        best_qwk = qwk
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
    
    scheduler.step()
```

### 15.4 Experiment Tracking - Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="dr-uncertainty", config={
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 32,
    "architecture": "ResNet50-CBAM",
    "dropout": 0.5
})

# Log metrics
wandb.log({
    "train_loss": loss.item(),
    "val_qwk": qwk,
    "learning_rate": optimizer.param_groups[0]['lr']
})

# Log images
wandb.log({"attention_maps": [wandb.Image(img) for img in attention_imgs]})
```

### 15.5 Inference Pipeline

```python
def predict_with_uncertainty(model, image, T=20):
    """
    Predict DR grade with uncertainty quantification
    
    Returns:
        predicted_grade (int): 0-4
        uncertainty (float): Uncertainty score
        class_probs (np.array): Probability distribution
    """
    model.eval()
    image = preprocess(image).unsqueeze(0).cuda()
    
    # MC Dropout
    mean_pred, variance = model.mc_dropout_predict(image, T=T)
    
    predicted_grade = torch.argmax(mean_pred, dim=1).item()
    uncertainty = variance.sum(dim=1).item()
    class_probs = mean_pred.cpu().numpy()[0]
    
    return predicted_grade, uncertainty, class_probs

# Usage
image = load_image("fundus.jpg")
grade, uncertainty, probs = predict_with_uncertainty(model, image)

print(f"Predicted Grade: {grade}")
print(f"Uncertainty Score: {uncertainty:.4f}")
print(f"Class Probabilities: {probs}")

if uncertainty > threshold:
    print("⚠️ High uncertainty - Flag for expert review")
```

**Nguồn đọc thêm**:
- 📘 *PyTorch Official Tutorials*: https://pytorch.org/tutorials/
- 📘 *Deep Learning with PyTorch* - Stevens et al.
- 🌐 Weights & Biases Documentation
- 🌐 PyTorch Lightning (high-level framework)

---

## APPENDIX A: CÔNG THỨC TỔNG HỢP

### Loss Functions
```
Binary Cross-Entropy:
  BCE = -(1/N) Σᵢ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

Categorical Cross-Entropy:
  CCE = -(1/N) Σᵢ Σⱼ yᵢⱼ·log(pᵢⱼ)

Focal Loss:
  FL(pₜ) = -αₜ(1 - pₜ)^γ log(pₜ)
```

### Activation Functions
```
ReLU: f(x) = max(0, x)
Sigmoid: σ(x) = 1 / (1 + e⁻ˣ)
Tanh: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
Softmax: softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

### Optimization
```
SGD: θₜ₊₁ = θₜ - η·∇L(θₜ)

Adam:
  mₜ = β₁·mₜ₋₁ + (1-β₁)·∇L(θₜ)
  vₜ = β₂·vₜ₋₁ + (1-β₂)·∇L(θₜ)²
  θₜ₊₁ = θₜ - η·m̂ₜ/(√v̂ₜ + ε)

Cosine Annealing:
  ηₜ = η_min + (η_max - η_min)·(1 + cos(πt/T))/2
```

### Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2·(Precision·Recall) / (Precision + Recall)

QWK = 1 - (Σᵢⱼ wᵢⱼ·Oᵢⱼ) / (Σᵢⱼ wᵢⱼ·Eᵢⱼ)

Pearson r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)²·Σ(yᵢ - ȳ)²]
```

---

## APPENDIX B: CHECKLIST TRIỂN KHAI

### Phase 1: Data Preparation
- [ ] Download APTOS 2019 & Messidor-2
- [ ] EDA: Class distribution, image quality
- [ ] Implement Ben Graham preprocessing
- [ ] Calculate class weights
- [ ] Create stratified splits (80/10/10)
- [ ] Setup data augmentation pipeline (Albumentations)

### Phase 2: Model Implementation
- [ ] Load pretrained ResNet-50
- [ ] Implement CBAM modules
- [ ] Insert CBAM after each ResNet block
- [ ] Implement permanent dropout in classification head
- [ ] Verify forward pass shape
- [ ] Implement MC dropout prediction method

### Phase 3: Training
- [ ] Implement Focal Loss with class weights
- [ ] Setup AdamW optimizer
- [ ] Setup Cosine Annealing scheduler
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Implement early stopping (patience=15, monitor QWK)
- [ ] Setup Weights & Biases logging
- [ ] Train model

### Phase 4: Uncertainty Quantification
- [ ] Implement MC dropout inference (T=20)
- [ ] Compute mean predictions
- [ ] Compute variance (uncertainty score)
- [ ] Evaluate on Messidor-2
- [ ] Compute QWK, accuracy, sensitivity, specificity
- [ ] Compute error-uncertainty correlation

### Phase 5: Analysis & Visualization
- [ ] Temperature scaling for calibration
- [ ] Compute ECE before/after calibration
- [ ] Generate reliability diagrams
- [ ] Implement Grad-CAM
- [ ] Visualize attention maps
- [ ] Error analysis: correctly vs incorrectly classified
- [ ] Determine uncertainty threshold for flagging

### Phase 6: Reporting
- [ ] Ablation studies (w/ vs w/o CBAM, w/ vs w/o uncertainty)
- [ ] Compare to baseline ResNet-50
- [ ] Clinical workflow proposal
- [ ] Limitations & future work

---

## APPENDIX C: NGUỒN HỌC TẬP TỔNG HỢP

### Sách Nền Tảng
1. **Deep Learning** - Goodfellow, Bengio, Courville (2016) ⭐⭐⭐
2. **Dive into Deep Learning** - Zhang et al. (d2l.ai) - Free online
3. **Pattern Recognition and Machine Learning** - Bishop (2006)
4. **Mathematics for Machine Learning** - Deisenroth et al. (2020)

### Courses
1. **CS231n: CNNs for Visual Recognition** (Stanford) - YouTube
2. **Deep Learning Specialization** (Coursera - Andrew Ng) ⭐
3. **Fast.ai Practical Deep Learning** - Free
4. **Full Stack Deep Learning** - fullstackdeeplearning.com

### Papers (Must-Read cho dự án)
1. ⭐ *Deep Residual Learning* - He et al. (2015)
2. ⭐ *CBAM* - Woo et al. (2018)
3. ⭐ *Dropout as Bayesian Approximation* - Gal & Ghahramani (2016)
4. ⭐ *Focal Loss* - Lin et al. (2017)
5. *Batch Normalization* - Ioffe & Szegedy (2015)
6. *AdamW* - Loshchilov & Hutter (2019)
7. *On Calibration of Modern NNs* - Guo et al. (2017)
8. *Grad-CAM* - Selvaraju et al. (2017)

### Blogs & Tutorials
- **colah's blog**: https://colah.github.io/ (visualization)
- **distill.pub**: https://distill.pub/ (interactive ML articles)
- **Jay Alammar's Blog**: https://jalammar.github.io/ (transformers, attention)
- **Papers With Code**: https://paperswithcode.com/

### Tools & Libraries
- **PyTorch**: https://pytorch.org/
- **Albumentations**: https://albumentations.ai/
- **Weights & Biases**: https://wandb.ai/
- **Grad-CAM**: https://github.com/jacobgil/pytorch-grad-cam

### Medical AI Resources
- **Grand Challenge**: https://grand-challenge.org/ (medical imaging competitions)
- **MICCAI**: Medical Image Computing (top conference)
- **Kaggle Medical Imaging**: https://www.kaggle.com/competitions?search=medical

---

## KẾT LUẬN

Tài liệu này tổng hợp **toàn bộ kiến thức kỹ thuật** cần thiết cho dự án **Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading**, từ:

1. **Nền tảng toán học**: Xác suất, thống kê, đại số tuyến tính
2. **Deep Learning cơ bản**: Neural networks, backpropagation, activation functions
3. **CNN & ResNet**: Convolution, residual learning, transfer learning
4. **Attention mechanisms**: CBAM architecture
5. **Uncertainty quantification**: MC dropout, Bayesian deep learning
6. **Loss functions & optimization**: Focal Loss, AdamW, cosine annealing
7. **Regularization**: L2, dropout, data augmentation
8. **Evaluation metrics**: QWK, sensitivity, specificity, ECE, correlation
9. **Training techniques**: Stratified split, early stopping, mixed precision
10. **Calibration & visualization**: Temperature scaling, Grad-CAM
11. **Medical imaging**: DR grading, fundus imaging, clinical deployment

Mỗi section đều có:
- ✅ **Lý thuyết**: Định nghĩa, khái niệm
- ✅ **Công thức**: Mathematical formulations
- ✅ **Ứng dụng trong dự án**: Cách áp dụng cụ thể
- ✅ **Code snippets**: PyTorch implementation
- ✅ **Nguồn đọc thêm**: Papers, books, courses

**Cách sử dụng tài liệu**:
1. **Học tuần tự**: Đọc từ Section 1 → 15 để build foundation
2. **Học theo nhu cầu**: Jump to specific sections khi implement
3. **Thực hành**: Implement code snippets trong PyTorch
4. **Đọc papers**: Follow "Nguồn đọc thêm" để đào sâu
5. **Debug**: Quay lại review lý thuyết khi gặp vấn đề

**Lưu ý quan trọng**:
- Không có synthetic data trong dự án này - mọi số liệu đều từ nguồn thực
- Citations đầy đủ [1]-[199] tham chiếu đến research notes đã thu thập
- Implementation phải tuân theo best practices trong medical AI

**Good luck với dự án! 🚀**

---

**Tài liệu được biên soạn**: January 2026  
**Version**: 1.0  
**Nguồn**: Comprehensive research trên 199 academic papers, tutorials, và documentation
