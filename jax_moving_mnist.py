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
# Add tqdm for progress tracking
from tqdm.auto import tqdm

# Check for GPU availability
print("JAX devices:", jax.devices())
print("Using device:", jax.devices()[0])  # Should show RTX 3090

# Capsule Layer Implementation (unchanged)
class CapsuleLayer(nn.Module):
    num_capsules: int
    capsule_dim: int
    kernel_size: int = 3
    strides: int = 1

    @nn.compact
    def __call__(self, inputs):
        # Convolution to get capsule parameters
        x = nn.Conv(self.num_capsules * self.capsule_dim, 
                   (self.kernel_size, self.kernel_size),
                   strides=(self.strides, self.strides),
                   padding='SAME')(inputs)
        # Reshape to capsules
        x = x.reshape(*inputs.shape[:-3], -1, self.capsule_dim)
        # Squashing activation
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = x_norm**2 / (1 + x_norm**2)
        squash = scale * x / jnp.where(x_norm > 0, x_norm, 1.0)  # Avoid division by zero
        return squash

# Self-Attention Layer (unchanged)
class SelfAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, inputs):
        batch, seq_len, features = inputs.shape
        scale = 1.0 / jnp.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(inputs)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Attention computation
        attn = jnp.einsum('bth,bsh->bts', q, k) * scale
        attn_weights = jax.nn.softmax(attn)
        output = jnp.einsum('bts,bsh->bth', attn_weights, v)
        
        return output

# Encoder-Decoder Architecture (modified for 32x32 Moving MNIST)
class MotionTransformer(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Check the shape and handle different input formats
        if len(x.shape) == 4:  # [batch, seq_len, height, width]
            # Add channel dimension if missing
            x = jnp.expand_dims(x, axis=-1)
        
        # Now x should be: [batch, num_frames, height, width, channels]
        batch, num_frames, h, w, c = x.shape
        
        # Capsule encoder per frame
        capsules = []
        for i in range(num_frames):
            cap = CapsuleLayer(num_capsules=32, capsule_dim=16, name=f'capsule_{i}')(x[:, i])
            capsules.append(cap)
        
        # Stack capsules across frames
        x = jnp.stack(capsules, axis=1)  # [batch, num_frames, num_capsules, capsule_dim]
        
        # Flatten capsule dimensions
        x = x.reshape(batch, num_frames, -1)
        
        # Self-attention over time
        x = SelfAttention(num_heads=4, head_dim=32)(x)
        
        # Decoder to reconstruct next frame - modified for 32x32 output
        x = nn.Dense(1024)(x.mean(axis=1))  # Reduced size for 32x32 output (1024 = 16x16x4)
        x = x.reshape(batch, 16, 16, 4)  # Reshape to image-like structure
        print(f"Decoder intermediate shape: {x.shape}")
        
        # CNN decoder (adjusted for 32x32 output)
        x = nn.ConvTranspose(64, (3,3), strides=(2,2), padding='SAME')(x)  # 16x16 -> 32x32
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (3,3), strides=(1,1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, (3,3), strides=(1,1), padding='SAME')(x)
        
        return jnp.tanh(x)
    

# Loss function
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

# Training step
@partial(jax.jit, static_argnames=('model',))
def train_step(state, model, batch):
    x, y = batch
    
    # Ensure x has the right shape [batch, seq_len, height, width, channels]
    if len(x.shape) == 4:  # [batch, seq_len, height, width]
        x = jnp.expand_dims(x, axis=-1)
    
    # Ensure y has the right shape [batch, 1, height, width, channels]
    if len(y.shape) == 4:  # [batch, 1, height, width]
        y = jnp.expand_dims(y, axis=-1)
    
    def loss_fn(params):
        pred = model.apply({'params': params}, x)
        # Reshape prediction to match target if needed
        if pred.shape != y.shape:
            # If prediction is [batch, height, width, channels]
            # and target is [batch, 1, height, width, channels]
            if len(pred.shape) == 4 and len(y.shape) == 5:
                pred = jnp.expand_dims(pred, axis=1)
        loss = mse_loss(pred, y)
        return loss, pred
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    return loss, state.apply_gradients(grads=grads)


# Evaluation step
@jax.jit
def eval_step(state, model, batch):
    x, y = batch
    
    # Ensure x has the right shape [batch, seq_len, height, width, channels]
    if len(x.shape) == 4:  # [batch, seq_len, height, width]
        x = jnp.expand_dims(x, axis=-1)
    
    # Ensure y has the right shape [batch, 1, height, width, channels]
    if len(y.shape) == 4:  # [batch, 1, height, width]
        y = jnp.expand_dims(y, axis=-1)
    
    pred = model.apply({'params': state.params}, x)
    
    # Reshape prediction to match target if needed
    if pred.shape != y.shape:
        # If prediction is [batch, height, width, channels]
        # and target is [batch, 1, height, width, channels]
        if len(pred.shape) == 4 and len(y.shape) == 5:
            pred = jnp.expand_dims(pred, axis=1)
            
    return mse_loss(pred, y), pred

def create_train_state(rng, model, learning_rate):
    # Moving MNIST sequence shape with batch dimension (changed to 32x32)
    dummy_input = jnp.ones((1, 4, 32, 32, 1))
    print(f"Initializing model with input shape: {dummy_input.shape}")
    params = model.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Load and prepare Moving MNIST dataset (modified for 32x32)
def load_moving_mnist(batch_size=32, seq_len=10, train_samples=1000, test_samples=200):
    """
    Load a subset of the Moving MNIST dataset for training and testing.
    Resizes images from 64x64 to 32x32 to reduce memory usage.
    
    Args:
        batch_size: Batch size for training
        seq_len: Number of input frames to use
        train_samples: Number of examples to use for training
        test_samples: Number of examples to use for testing
    
    Returns:
        train_ds, test_ds: Training and testing datasets
    """
    # The Moving MNIST dataset only has a 'test' split
    dataset = tfds.load('moving_mnist', split='test', shuffle_files=True)
    
    # Moving MNIST format: Each example has 'image_sequence' with shape [20, 64, 64]
    def preprocess(example):
        # Get the sequence
        sequence = example['image_sequence']
        
        # Resize from 64x64 to 32x32 (handling each frame separately)
        resized_frames = []
        for i in range(sequence.shape[0]):
            frame = sequence[i]  # Get single frame [64, 64]
            # Add channel dimension required by resize
            frame = tf.expand_dims(frame, axis=-1)  # [64, 64, 1]
            # Resize the frame
            resized = tf.image.resize(
                frame, 
                [32, 32], 
                method=tf.image.ResizeMethod.BILINEAR
            )  # [32, 32, 1]
            resized = tf.squeeze(resized, axis=-1)  # [32, 32]
            resized_frames.append(resized)
        
        # Stack frames back into sequence
        sequence = tf.stack(resized_frames, axis=0)  # [20, 32, 32]
        
        # Normalize to [-1, 1]
        frames = tf.cast(sequence, tf.float32) / 255.0 * 2.0 - 1.0
        
        # Add channel dimension if not present
        if len(frames.shape) == 3:  # [time, height, width]
            frames = tf.expand_dims(frames, axis=-1)  # [time, height, width, channels]
        
        # Input: first seq_len frames, Target: next frame
        inputs = frames[:seq_len]
        target = frames[seq_len:seq_len+1]
        
        return inputs, target
    
    # Process dataset
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Get total dataset size (without loading full dataset to save memory)
    full_dataset = tfds.load('moving_mnist', split='test')
    total_dataset_size = 10000  # Moving MNIST has 10,000 sequences
    print(f"Total dataset size: {total_dataset_size} examples")
    
    # Ensure we're not requesting more samples than available
    available_samples = min(total_dataset_size, train_samples + test_samples)
    train_samples = min(train_samples, int(available_samples * 0.8))
    test_samples = min(test_samples, available_samples - train_samples)
    
    print(f"Using {train_samples} training samples and {test_samples} testing samples")
    
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    
    # Take only the number of examples we want
    smaller_dataset = dataset.take(train_samples + test_samples)
    
    # Split into train and test
    train_ds = smaller_dataset.take(train_samples)
    test_ds = smaller_dataset.skip(train_samples).take(test_samples)
    
    # Batch and prefetch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


# Main training function
def train_and_evaluate(model, num_epochs=5, batch_size=32, seq_len=10, learning_rate=1e-3, train_samples=1000, test_samples=200):
    """
    Train and evaluate the model.
    
    Args:
        model: The model to train
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        seq_len: Number of input frames to use
        learning_rate: Learning rate for the optimizer
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        
    Returns:
        state: Trained model state
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
    """
    # Load Moving MNIST dataset
    train_ds, test_ds = load_moving_mnist(batch_size, seq_len, train_samples, test_samples)
    
    # Initialize model and training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate)
    
    # Metrics tracking
    train_losses = []
    test_losses = []
    
    # Training loop with tqdm
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # Training
        epoch_train_losses = []
        start_time = time.time()
        
        # Estimate the number of batches (to avoid consuming the iterator)
        train_ds_size = train_samples // batch_size + (1 if train_samples % batch_size else 0)
        
        # Create the training dataset iterator
        train_ds_iter = iter(tfds.as_numpy(train_ds))
        
        # Training progress bar
        train_pbar = tqdm(range(train_ds_size), desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for _ in train_pbar:
            try:
                batch = next(train_ds_iter)
                
                # Handle the batch
                if isinstance(batch, dict):
                    # If batch is a dict, extract x and y (should not happen with our preprocessing)
                    raise ValueError("Unexpected batch format: dict instead of tuple")
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # If batch is a tuple with 2 elements (as expected)
                    x, y = batch
                else:
                    # If batch format is unexpected
                    raise ValueError(f"Unexpected batch format: {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                
                # Training step
                loss, state = train_step(state, model, (x, y))
                epoch_train_losses.append(float(loss))
                
                # Update progress bar description with current loss
                train_pbar.set_postfix(loss=f"{float(loss):.4f}")
                
            except StopIteration:
                break
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        epoch_test_losses = []
        test_predictions = []
        test_targets = []
        test_inputs = []
        
        # Estimate the number of test batches (to avoid consuming the iterator) 
        test_ds_size = test_samples // batch_size + (1 if test_samples % batch_size else 0)
        
        # Create the test dataset iterator
        test_ds_iter = iter(tfds.as_numpy(test_ds))
        
        # Testing progress bar
        test_pbar = tqdm(range(test_ds_size), desc=f"Epoch {epoch+1}/{num_epochs} - Testing", leave=False)
        for _ in test_pbar:
            try:
                batch = next(test_ds_iter)
                
                # Handle the batch
                if isinstance(batch, dict):
                    # If batch is a dict, extract x and y (should not happen with our preprocessing)
                    raise ValueError("Unexpected batch format: dict instead of tuple")
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # If batch is a tuple with 2 elements (as expected)
                    x, y = batch
                else:
                    # If batch format is unexpected
                    raise ValueError(f"Unexpected batch format: {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                
                # Evaluation step
                loss, pred = eval_step(state, model, (x, y))
                epoch_test_losses.append(float(loss))
                
                # Update progress bar description with current loss
                test_pbar.set_postfix(loss=f"{float(loss):.4f}")
                
                # Save predictions for visualization (just a few batches)
                if len(test_predictions) < 5:
                    test_predictions.append(np.array(pred))
                    test_targets.append(np.array(y))
                    test_inputs.append(np.array(x))
                    
            except StopIteration:
                break
        
        # Calculate average test loss for the epoch
        avg_test_loss = np.mean(epoch_test_losses)
        test_losses.append(avg_test_loss)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot training and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs+1), test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curves.png')
    plt.close()
    
    # Visualize sample predictions
    for i in range(len(test_predictions)):
        plt.figure(figsize=(15, 8))
        
        # Get batch data
        inputs = test_inputs[i]
        targets = test_targets[i]
        preds = test_predictions[i]
        
        # Show 5 examples from the batch
        for j in range(min(5, inputs.shape[0])):
            # Get input sequence, target, and prediction
            input_seq = inputs[j]
            target = targets[j].squeeze()
            pred = preds[j].squeeze()
            
            # Convert from [-1,1] to [0,1] for visualization
            input_seq = (input_seq + 1.0) / 2.0
            target = (target + 1.0) / 2.0
            pred = (pred + 1.0) / 2.0
            
            # Plot the last 3 input frames
            for k in range(3):
                idx = seq_len - 3 + k
                plt.subplot(5, 5, j*5 + k + 1)
                plt.imshow(input_seq[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
                plt.title(f'Input Frame {idx+1}')
                plt.axis('off')
            
            # Plot ground truth next frame
            plt.subplot(5, 5, j*5 + 4)
            plt.imshow(target, cmap='gray', vmin=0, vmax=1)
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Plot predicted next frame
            plt.subplot(5, 5, j*5 + 5)
            plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
            plt.title('Prediction')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/predictions_batch_{i+1}.png')
        plt.close()
    
    return state, train_losses, test_losses


if __name__ == '__main__':
    # Initialize model
    model = MotionTransformer()
    
    # Train the model with smaller dataset to test
    state, train_losses, test_losses = train_and_evaluate(
        model, 
        num_epochs=5,  # Using a few epochs as requested
        batch_size=4,
        seq_len=4,
        learning_rate=1e-3,
        train_samples=100,  # Using 1000 training samples
        test_samples=20     # Using 200 test samples
    )
    
    print("Training completed! Results saved in 'results' directory.")