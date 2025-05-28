# Capsule Transformer for Video Frame Prediction on Moving MNIST

This project implements a Capsule Transformer model for predicting future frames in video sequences, demonstrated on the Moving MNIST dataset. The model leverages capsule networks for robust feature representation and a transformer architecture for capturing temporal dependencies. This implementation uses JAX and Flax.

## 📝 Table of Contents

* [✨ Features](#-features)
* [⚙️ Requirements](#️-requirements)
* [💾 Dataset](#-dataset)
* [🏗️ Model Architecture](#️-model-architecture)
    * [CapsuleLayer](#capsulelayer)
    * [SimpleAttention](#simpleattention)
    * [CapsuleTransformer](#capsuletransformer)
* [🚀 Getting Started](#-getting-started)
    * [Installation](#installation)
    * [Running the Code](#running-the-code)
* [📈 Training Process](#-training-process)
* [📊 Results and Visualization](#-results-and-visualization)
* [📂 Code Structure](#-code-structure)
* [💡 Key Hyperparameters](#-key-hyperparameters)
* [🔮 Future Improvements](#-future-improvements)

---

## ✨ Features

* **Capsule Network Integration:** Utilizes a memory-efficient `CapsuleLayer` for hierarchical feature extraction.
* **Transformer Backbone:** Employs a simplified self-attention mechanism (`SimpleAttention`) within a transformer-like architecture to model temporal relationships between frames.
* **Video Frame Prediction:** Predicts the next frame in a sequence given a series of preceding frames.
* **JAX & Flax Implementation:** Built with JAX for high-performance numerical computing and Flax for neural network construction.
* **Moving MNIST Dataset:** Uses the standard Moving MNIST dataset for training and evaluation.
* **Clear Visualization:** Generates plots for training/test loss curves and visual comparisons of input, ground truth, and predicted frames.

---

## ⚙️ Requirements

The project relies on the following Python libraries:

* `jax` and `jaxlib`
* `flax`
* `optax`
* `numpy`
* `tensorflow` (primarily for `tf.data` and `tensorflow_datasets`)
* `tensorflow-datasets` (for loading Moving MNIST)
* `matplotlib` (for generating visualizations)
* `tqdm` (for progress bars)

You can install these dependencies, preferably in a virtual environment:

```bash
pip install jax jaxlib flax optax numpy tensorflow tensorflow-datasets matplotlib tqdm
````

**Note on JAX Installation:** Depending on your hardware (CPU/GPU/TPU), JAX installation might require specific commands. Please refer to the [official JAX installation guide](https://www.google.com/search?q=https://github.com/google/jax%23installation) for detailed instructions.

The code includes a check for JAX devices upon execution:

```python
print("JAX devices:", jax.devices())
print("Using device:", jax.devices()[0])
```

-----

## 💾 Dataset

The model is trained and evaluated on the **Moving MNIST** dataset. This dataset consists of sequences of 64x64 grayscale frames, each showing two handwritten digits moving independently within the frame.

  * **Loading:** The `load_moving_mnist` function handles downloading (if necessary), preprocessing, and batching the dataset using `tensorflow_datasets`.
  * **Preprocessing Steps:**
    1.  Extracts `seq_len + 1` frames from each sequence (input sequence + target frame).
    2.  Casts frames to `tf.float32`.
    3.  Resizes frames from the original 64x64 to 32x32 pixels using area interpolation.
    4.  Ensures frames are single-channel (grayscale).
    5.  Normalizes pixel values to the range `[-1.0, 1.0]`.
    6.  Splits sequences into `inputs` (first `seq_len` frames) and `target` (the `(seq_len + 1)`-th frame).
  * **Data Splits:** The `load_moving_mnist` function creates training and testing datasets. The number of samples for each can be configured (default: 5000 train, 1000 test).
  * **Batching:** The datasets are batched and prefetched for efficient training.

-----

## 🏗️ Model Architecture

The core of the project is the `CapsuleTransformer` model, which combines capsule layers for spatial feature extraction and attention mechanisms for temporal modeling.

### CapsuleLayer

The `CapsuleLayer` is a memory-efficient implementation of a primary capsule layer.

  * **Convolutional Base:** It applies a 2D convolution to the input.
  * **Reshaping to Capsules:** The output of the convolution is reshaped to form `num_capsules` of dimension `capsule_dim`.
  * **Squashing Activation:** A non-linear "squashing" activation function is applied to normalize the length of capsule vectors, ensuring that short vectors are shrunk to near zero and long vectors get shrunk to a length just below 1.
    $$ \text{squash}(\mathbf{s}_j) = \frac{\|\mathbf{s}_j\|^2}{1 + \|\mathbf{s}_j\|^2} \frac{\mathbf{s}_j}{\|\mathbf{s}_j\| + \epsilon} $$

### SimpleAttention

The `SimpleAttention` module implements a standard multi-head self-attention mechanism without dropout, suitable for sequence processing.

  * **Input:** Takes a sequence of feature vectors.
  * **Multi-Head Attention:**
    1.  Linearly projects the input into queries (Q), keys (K), and values (V) for multiple heads.
    2.  Computes scaled dot-product attention:
        $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
    3.  Concatenates the outputs of the multiple heads.
  * **Output Projection:** Applies a final linear layer to produce the output sequence.

### CapsuleTransformer

The `CapsuleTransformer` model orchestrates the prediction task:

1.  **Input Handling:** Accepts a batch of video sequences of shape `(batch, num_frames, height, width, channels)`. If input is `(batch, num_frames, height, width)`, it adds a channel dimension.
2.  **Frame Encoding (Shared Capsule Layer):**
      * Each frame in the input sequence is processed independently by a shared `CapsuleLayer`.
      * Frames are reshaped and fed into the `CapsuleLayer` to produce capsule representations.
      * The capsule outputs are then flattened and reshaped back to form a sequence of capsule features: `(batch, num_frames, flattened_capsule_features)`.
3.  **Positional Encoding:** Learned positional encodings are added to the sequence of capsule features to provide information about the order of frames.
4.  **Layer Normalization:** Applied after positional encoding.
5.  **Transformer Encoder Block:**
      * **Self-Attention:** The `SimpleAttention` module processes the sequence of frame features to capture temporal relationships. A residual connection is used.
      * **Layer Normalization.**
      * **Feed-Forward Network (FFN):** A position-wise FFN (two dense layers with GELU activation) is applied. A residual connection is used.
      * **Layer Normalization.**
6.  **Temporal Aggregation:**
      * A dense layer computes temporal attention weights for each frame's features.
      * These weights are softmax-normalized across the time dimension.
      * The frame features are weighted and summed to produce a single aggregated feature vector representing the entire input sequence.
7.  **Decoder (Frame Generation):**
      * The aggregated feature vector is passed through a series of dense and transposed convolution layers to reconstruct the predicted next frame.
      * GELU activations are used in intermediate layers.
      * The final layer is a 2D convolution producing a single-channel image.
      * A `tanh` activation function is applied to the output, ensuring pixel values are in the `[-1.0, 1.0]` range, consistent with the input normalization.

-----

## 🚀 Getting Started

### Installation

1.  **Clone the repository (if applicable) or save the code:**
    Save the provided Python script as (e.g., `jax_moving_mnist.py`).

2.  **Install dependencies:**
    As mentioned in the [Requirements](https://www.google.com/search?q=%23%EF%B8%8F-requirements) section, ensure all necessary libraries are installed.

### Running the Code

You can run the training script directly from your terminal:

```bash
python jax_moving_mnist.py
```

The main execution block (`if __name__ == "__main__":`) sets the training hyperparameters:

```python
if __name__ == "__main__":
    state, train_losses, test_losses = train_model(
        num_epochs=300,      # Number of training epochs
        batch_size=16,       # Batch size for training and testing
        seq_len=16,          # Number of input frames in a sequence
        learning_rate=1e-4   # Learning rate for the Adam optimizer
    )
    # ...
```

You can modify these values in the script to experiment with different configurations.

-----

## 📈 Training Process

  * **Initialization:**
      * A JAX random key (`PRNGKey`) is initialized for reproducibility.
      * The `create_train_state` function initializes the `CapsuleTransformer` model and the Adam optimizer (with gradient clipping by global norm). It returns a `TrainState` object from Flax, which conveniently bundles model parameters, apply function, and optimizer state.
      * The total number of model parameters is printed.
  * **Loss Function:** The Mean Squared Error (MSE) is used to measure the difference between the predicted frame and the ground truth frame.
    $$ \text{MSE}(\text{pred}, \text{target}) = \text{mean}((\text{pred} - \text{target})^2) $$
  * **Training Step (`train_step`):**
      * This function is JIT-compiled with `@jax.jit` for performance.
      * It takes the current training state and a batch of data (input sequences `x`, target frames `y`).
      * It computes the loss and gradients of the loss with respect to the model parameters.
      * Gradients are clipped to the range `[-1.0, 1.0]` to prevent exploding gradients.
      * The optimizer updates the model parameters using the clipped gradients.
      * Returns the loss, updated state, and predictions for the batch.
  * **Evaluation Step (`eval_step`):**
      * Also JIT-compiled.
      * Takes the current training state and a batch of evaluation data.
      * Computes the predictions and the loss without updating model parameters.
      * Returns the loss and predictions for the batch.
  * **Main Training Loop (`train_model`):**
      * Loads the Moving MNIST dataset.
      * Initializes the training state.
      * Iterates for `num_epochs`:
          * **Training Phase:** Iterates through the training dataset using `tqdm` for a progress bar. For each batch, `train_step` is called. The average training loss for the epoch is recorded.
          * **Evaluation Phase:** Iterates through the test dataset. For each batch, `eval_step` is called. The average test loss for the epoch is recorded. Predictions, targets, and inputs from the first few test batches are saved for visualization.
          * Prints the average training and test loss for the epoch.
      * After training, calls `visualize_results` to save plots.
      * Returns the final training state and lists of training and test losses.

-----

## 📊 Results and Visualization

The `visualize_results` function is called at the end of training to generate and save two types of plots in a directory named `results/`:

1.  **Loss Curves (`results/loss_curves.png`):**

      * A plot showing the training loss (blue line) and test loss (red line) over epochs.
      * Helps in diagnosing training progress, overfitting, etc.

2.  **Prediction Visualizations (`results/predictions_batch_<batch_idx+1>.png`):**

      * For the first few batches of the test set (up to 3 batches):
          * A grid of images is generated (4 rows, 4 columns).
          * Each row displays:
              * The last two input frames from a sequence (`Input N-1`, `Input N`).
              * The ground truth target frame (`Ground Truth`).
              * The model's predicted frame (`Prediction`).
          * Pixel values are denormalized from `[-1, 1]` to `[0, 1]` for display.
      * This provides a qualitative assessment of the model's prediction quality.

The script also prints the final training and test loss values to the console upon completion.

-----

## 📂 Code Structure

The Python script is organized into the following main components:

  * **Imports:** Standard library and third-party library imports.
  * **Device Check:** Prints available JAX devices.
  * **`visualize_results(train_losses, test_losses, inputs, targets, predictions, seq_len)`:** Generates and saves plots of loss curves and predicted frames.
  * **`CapsuleLayer(nn.Module)`:** Defines the capsule layer.
      * `__call__(self, inputs)`: Forward pass for the capsule layer.
  * **`SimpleAttention(nn.Module)`:** Defines the self-attention mechanism.
      * `__call__(self, inputs)`: Forward pass for the attention layer.
  * **`CapsuleTransformer(nn.Module)`:** Defines the main model architecture.
      * `__call__(self, x)`: Forward pass for the full model.
  * **`mse_loss(pred, target)`:** Calculates the Mean Squared Error loss.
  * **`train_step(state, batch)`:** Performs a single training step (loss, gradients, update).
  * **`eval_step(state, batch)`:** Performs a single evaluation step (loss, predictions).
  * **`create_train_state(rng, learning_rate, seq_len)`:** Initializes the model and optimizer.
  * **`load_moving_mnist(batch_size, seq_len, train_samples, test_samples)`:** Loads and preprocesses the Moving MNIST dataset.
  * **`train_model(num_epochs, batch_size, seq_len, learning_rate)`:** Orchestrates the entire training and evaluation process.
  * **`if __name__ == "__main__":`:** Entry point of the script; sets hyperparameters and calls `train_model`.

-----

## 💡 Key Hyperparameters

The following hyperparameters can be adjusted in the `if __name__ == "__main__":` block or by modifying the `train_model` function's defaults:

  * **`num_epochs`:** Number of complete passes through the training dataset (default in `main`: 300).
  * **`batch_size`:** Number of sequences processed in one iteration (default in `main`: 16).
  * **`seq_len`:** Length of the input frame sequence used for prediction (default in `main`: 16).
  * **`learning_rate`:** Step size for the Adam optimizer (default in `main`: `1e-4`).

Model-specific hyperparameters are defined within the `CapsuleTransformer` class:

  * **`num_capsules`:** Number of capsules in the `CapsuleLayer` (default: 16).
  * **`capsule_dim`:** Dimensionality of each capsule (default: 8).
  * **`num_heads`:** Number of attention heads in `SimpleAttention` (default: 2).
  * **`head_dim`:** Dimensionality of each attention head (default: 16).
  * **`hidden_dim`:** Dimensionality of the FFN and decoder intermediate layers (default: 256).

-----

## 🔮 Future Improvements

  * **More Sophisticated Attention:** Explore more advanced attention mechanisms (e.g., Transformer-XL, Longformer) for handling longer sequences more effectively.
  * **Dynamic Routing for Capsules:** Implement dynamic routing or EM routing between capsule layers for potentially better hierarchical representations, though this adds complexity.
  * **Different Datasets:** Test the model on more complex video prediction datasets.
  * **Hyperparameter Optimization:** Perform systematic hyperparameter tuning (e.g., using Optuna or Ray Tune) to find optimal configurations.
  * **Advanced Decoder:** Improve the decoder architecture, potentially using skip connections or more sophisticated upsampling methods.
  * **Stochasticity in Predictions:** For multi-modal futures, explore variational autoencoders (VAEs) or generative adversarial networks (GANs) to model a distribution of possible future frames.
  * **Regularization:** Experiment with different regularization techniques (e.g., dropout in attention/FFN, weight decay) if overfitting is observed.

<!-- end list -->
