import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load and inspect the dataset
dataset = tfds.load('moving_mnist', split='test', shuffle_files=True)

# Take one example and inspect its structure
for example in dataset.take(1):
    print("Example keys:", example.keys())
    
    # Check the image sequence
    if 'image_sequence' in example:
        sequence = example['image_sequence']
        print(f"Sequence shape: {sequence.shape}")
        print(f"Sequence dtype: {sequence.dtype}")
        print(f"Sequence min/max values: {tf.reduce_min(sequence)}, {tf.reduce_max(sequence)}")
        
        # Try different ways to handle the sequence
        print("\nTesting different approaches:")
        
        # Check individual frame shape
        frame = sequence[0]
        print(f"Single frame shape: {frame.shape}")
        
        # Test if we need to handle the resize differently
        try:
            # Method 1: Resize each frame separately
            print("\nMethod 1: Resize frames separately")
            seq_len = 4
            frames_to_process = sequence[:seq_len + 1]
            
            resized_frames = []
            for i in range(seq_len + 1):
                frame = frames_to_process[i]
                # Check if frame needs channel dimension
                if len(frame.shape) == 2:
                    frame = tf.expand_dims(frame, axis=-1)
                print(f"Frame {i} shape before resize: {frame.shape}")
                
                # Resize single frame
                resized = tf.image.resize(frame, [32, 32], method=tf.image.ResizeMethod.AREA)
                resized_frames.append(resized)
            
            # Stack frames back
            resized_sequence = tf.stack(resized_frames, axis=0)
            print(f"Resized sequence shape: {resized_sequence.shape}")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        try:
            # Method 2: Batch resize
            print("\nMethod 2: Batch resize with proper shape")
            frames_to_process = sequence[:seq_len + 1]
            
            # Add channel dimension if needed
            if len(frames_to_process.shape) == 3:
                frames_to_process = tf.expand_dims(frames_to_process, axis=-1)
            
            print(f"Frames shape for batch resize: {frames_to_process.shape}")
            
            # Resize as a batch
            resized = tf.image.resize(frames_to_process, [32, 32], method=tf.image.ResizeMethod.AREA)
            print(f"Batch resized shape: {resized.shape}")
            
        except Exception as e:
            print(f"Method 2 failed: {e}")