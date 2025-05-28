import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import time
from tqdm.auto import tqdm

# Check for GPU availability
print("JAX devices:", jax.devices())
print("Using device:", jax.devices()[0])

def visualize_results(train_losses, test_losses, inputs, targets, predictions, seq_len):
    """Create visualization plots."""
    os.makedirs('results', exist_ok=True)
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress - Capsule Transformer')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curves.png')
    plt.close()
    
    # Predictions
    for batch_idx in range(min(3, len(predictions))):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(min(4, inputs[batch_idx].shape[0])):
            # Last 2 input frames
            for j in range(2):
                idx = seq_len - 2 + j
                ax = axes[i, j]
                img = (inputs[batch_idx][i, idx].squeeze() + 1.0) / 2.0
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Input {idx+1}')
                ax.axis('off')
            
            # Ground truth
            ax = axes[i, 2]
            gt = (targets[batch_idx][i].squeeze() + 1.0) / 2.0
            ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
            ax.set_title('Ground Truth')
            ax.axis('off')
            
            # Prediction
            ax = axes[i, 3]
            pred = (predictions[batch_idx][i].squeeze() + 1.0) / 2.0
            ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
            ax.set_title('Prediction')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/predictions_batch_{batch_idx+1}.png', dpi=150)
        plt.close()
    
    print("Results saved in 'results' directory!")


# Memory-efficient Capsule Layer
class CapsuleLayer(nn.Module):
    num_capsules: int
    capsule_dim: int
    kernel_size: int = 3
    strides: int = 2

    @nn.compact
    def __call__(self, inputs):
        # Convolution with strided downsampling
        x = nn.Conv(
            self.num_capsules * self.capsule_dim,
            (self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding='SAME'
        )(inputs)
        
        # Get spatial dimensions
        batch, h, w, _ = x.shape
        
        # Reshape to capsules
        x = x.reshape(batch, h * w, self.num_capsules, self.capsule_dim)
        
        # Squashing activation
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = x_norm**2 / (1 + x_norm**2)
        squash = scale * x / jnp.maximum(x_norm, 1e-8)
        
        return squash

# Simple Self-Attention without dropout
class SimpleAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, inputs):
        batch, seq_len, features = inputs.shape
        
        # Multi-head attention
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(inputs)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn_weights = jax.nn.softmax(attn, axis=-1)
        
        output = jnp.matmul(attn_weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        
        # Output projection
        return nn.Dense(features)(output)

# Simplified Motion Transformer
class CapsuleTransformer(nn.Module):
    num_capsules: int = 16
    capsule_dim: int = 8
    num_heads: int = 2
    head_dim: int = 16
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # Handle input shape
        if len(x.shape) == 4:
            x = jnp.expand_dims(x, axis=-1)
        
        batch, num_frames, h, w, c = x.shape
        
        # Share capsule encoder across frames
        capsule_layer = CapsuleLayer(
            num_capsules=self.num_capsules,
            capsule_dim=self.capsule_dim
        )
        
        # Process all frames
        x_flat = x.reshape(batch * num_frames, h, w, c)
        capsules = capsule_layer(x_flat)
        
        # Reshape back
        _, spatial_pos, num_caps, cap_dim = capsules.shape
        capsules = capsules.reshape(batch, num_frames, spatial_pos * num_caps * cap_dim)
        
        # Add learned positional encoding
        pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (1, num_frames, capsules.shape[-1])
        )
        x = capsules + pos_encoding
        
        # Layer norm
        x = nn.LayerNorm()(x)
        
        # Self-attention block
        attn_out = SimpleAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim
        )(x)
        x = x + attn_out  # Residual connection
        x = nn.LayerNorm()(x)
        
        # Feed-forward block
        ffn_out = nn.Dense(self.hidden_dim)(x)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dense(x.shape[-1])(ffn_out)
        x = x + ffn_out  # Residual connection
        x = nn.LayerNorm()(x)
        
        # Temporal aggregation with learned weights
        temporal_weights = nn.Dense(1)(x)
        temporal_weights = jax.nn.softmax(temporal_weights, axis=1)
        x = jnp.sum(x * temporal_weights, axis=1)
        
        # Decoder
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(8 * 8 * 32)(x)
        x = x.reshape(batch, 8, 8, 32)
        
        # Upsample to 32x32
        x = nn.ConvTranspose(16, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(8, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.Conv(1, (3, 3), padding='SAME')(x)
        
        return jnp.tanh(x)

# Loss function
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

# Training step
@jax.jit
def train_step(state, batch):
    x, y = batch
    
    # Ensure shapes
    if len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=-1)
    if len(y.shape) == 3:
        y = jnp.expand_dims(y, axis=-1)
    
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, x)
        loss = mse_loss(pred, y.squeeze(axis=1))
        return loss, pred
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    
    # Gradient clipping
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    state = state.apply_gradients(grads=grads)
    return loss, state, pred

# Evaluation step
@jax.jit
def eval_step(state, batch):
    x, y = batch
    
    if len(x.shape) == 4:
        x = jnp.expand_dims(x, axis=-1)
    if len(y.shape) == 3:
        y = jnp.expand_dims(y, axis=-1)
    
    pred = state.apply_fn({'params': state.params}, x)
    loss = mse_loss(pred, y.squeeze(axis=1))
    return loss, pred

# Create training state
def create_train_state(rng, learning_rate, seq_len=4):
    model = CapsuleTransformer()
    dummy_input = jnp.ones((1, seq_len, 32, 32, 1))
    
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

# Data loading function
def load_moving_mnist(batch_size=8, seq_len=4, train_samples=500, test_samples=100):
    """Load Moving MNIST dataset."""
    
    dataset = tfds.load('moving_mnist', split='test', shuffle_files=True)
    
    def preprocess(example):
        sequence = example['image_sequence']
        
        # Take required frames
        frames_needed = seq_len + 1
        sequence = sequence[:frames_needed]
        
        # Convert and resize
        sequence = tf.cast(sequence, tf.float32)
        
        # Resize frames
        resized_frames = []
        for i in range(frames_needed):
            frame = sequence[i]
            if len(frame.shape) == 2:
                frame = tf.expand_dims(frame, axis=-1)
            elif len(frame.shape) == 3 and frame.shape[-1] > 1:
                frame = frame[..., :1]
            
            resized = tf.image.resize(frame, [32, 32], method=tf.image.ResizeMethod.AREA)
            resized_frames.append(resized)
        
        frames = tf.stack(resized_frames, axis=0)
        
        # Normalize
        frames = frames / 127.5 - 1.0
        
        # Split input/target
        inputs = frames[:seq_len]
        target = frames[seq_len:seq_len+1]
        
        return inputs, target
    
    # Process dataset
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Create splits
    total_samples = train_samples + test_samples
    dataset = dataset.take(total_samples).cache()
    
    train_ds = dataset.take(train_samples)
    test_ds = dataset.skip(train_samples)
    
    # Batch
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(2)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(2)
    
    return train_ds, test_ds

# Main training function
def train_model(num_epochs=5, batch_size=8, seq_len=4, learning_rate=5e-4):
    """Train the capsule-transformer model."""
    
    print("Loading dataset...")
    train_ds, test_ds = load_moving_mnist(
        batch_size=batch_size,
        seq_len=seq_len,
        train_samples=500,
        test_samples=100
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, learning_rate, seq_len)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Total parameters: {param_count:,}")
    
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        epoch_train_losses = []
        for batch in tqdm(tfds.as_numpy(train_ds), desc=f"Epoch {epoch+1} Train"):
            x, y = batch
            loss, state, _ = train_step(state, (x, y))
            epoch_train_losses.append(float(loss))
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Evaluate
        epoch_test_losses = []
        test_predictions = []
        test_targets = []
        test_inputs = []
        
        for i, batch in enumerate(tqdm(tfds.as_numpy(test_ds), desc=f"Epoch {epoch+1} Test")):
            x, y = batch
            loss, pred = eval_step(state, (x, y))
            epoch_test_losses.append(float(loss))
            
            if i < 3:  # Save for visualization
                test_predictions.append(np.array(pred))
                test_targets.append(np.array(y))
                test_inputs.append(np.array(x))
        
        avg_test_loss = np.mean(epoch_test_losses)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")
    
    # Visualize results
    visualize_results(train_losses, test_losses, test_inputs, test_targets, test_predictions, seq_len)
    
    return state, train_losses, test_losses


if __name__ == "__main__":
    # Train with small settings for testing
    state, train_losses, test_losses = train_model(
        num_epochs=5,
        batch_size=8,
        seq_len=4,
        learning_rate=5e-4
    )

    print(f"\nTraining completed!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final test loss: {test_losses[-1]:.4f}")
