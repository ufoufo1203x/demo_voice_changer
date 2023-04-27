import pyaudio
import numpy as np
import time
from scipy.signal import lfilter

# Parameters for the pitch shift effect
pitch_factor = 1.5
block_size = 2048
overlap = block_size // 2

# Create a PyAudio object
p = pyaudio.PyAudio()

# Open an audio stream. 1 octerb up program
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                output=True,
                frames_per_buffer=block_size)

# Define a pitch shift function
def pitch_shift(samples, pitch_factor):
    """
    Applies a pitch shift effect to an array of audio samples.
    """
    # Create a time-stretching factor based on the pitch factor
    stretch_factor = 1 / pitch_factor
    
    # Create a low-pass filter to remove high-frequency artifacts
    b = np.array([1 - stretch_factor])
    a = np.array([1, -stretch_factor])
    filt = lfilter(b, a, np.concatenate(([samples[0]], samples)))
    
    # Resample the filtered signal to the original sample rate
    resampled = np.interp(np.arange(0, len(samples), stretch_factor),
                          np.arange(0, len(filt)),
                          filt)
    
    # Remove any extra samples introduced by resampling
    resampled = resampled[:len(samples)]
    
    return resampled

# Start the audio stream
stream.start_stream()

while True:
    # Read a block of audio data from the stream
    block = stream.read(block_size)
    
    # Convert the raw bytes to a NumPy array of floats
    samples = np.frombuffer(block, dtype=np.float32)
    
    # Apply the pitch shift effect
    processed_samples = pitch_shift(samples, pitch_factor)
    
    # Convert the processed samples back to raw bytes
    processed_block = processed_samples.astype(np.float32).tobytes()
    
    # Write the processed audio data to the output stream
    stream.write(processed_block)
    
    # Wait for a short time to avoid overloading the CPU
    time.sleep(0.01)

# Stop the audio stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()
