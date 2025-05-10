import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from functools import partial

# Capsule Layer Implementation
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
        squash = scale * x / x_norm
        return squash

# Self-Attention Layer
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

# Encoder-Decoder Architecture
class MotionTransformer(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: [batch, num_frames, height, width, channels]
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
        
        # Decoder to reconstruct next frame
        x = nn.Dense(1024)(x.mean(axis=1))  # Aggregate temporal information
        x = x.reshape(batch, 32, 32, 1)  # Reshape to image-like structure
        
        # CNN decoder
        x = nn.ConvTranspose(64, (3,3), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (3,3), strides=(1,1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, (3,3), strides=(2,2), padding='SAME')(x)
        
        return jnp.tanh(x)

# Loss function
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

# Training step
@partial(jax.jit, static_argnames=('model',))
def train_step(state, model, batch):
    x, y = batch
    def loss_fn(params):
        pred = model.apply({'params': params}, x)
        loss = mse_loss(pred, y)
        return loss, pred
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    return loss, state.apply_gradients(grads=grads)

# Evaluation step
@jax.jit
def eval_step(state, model, batch):
    x, y = batch
    pred = model.apply({'params': state.params}, x)
    return mse_loss(pred, y)

# Initialize model
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 10, 28, 28, 1)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Example usage
if __name__ == '__main__':
    # Initialize model
    model = MotionTransformer()
    
    # Create training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, 1e-3)
    
    # Dummy data
    batch_size, seq_len = 32, 10
    dummy_input = jnp.ones((batch_size, seq_len, 28, 28, 1))
    
    # Forward pass
    output = model.apply({'params': state.params}, dummy_input)
    print(f'Output shape: {output.shape}')
