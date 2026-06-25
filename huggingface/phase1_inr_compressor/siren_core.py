import numpy as np
import json
import struct

class SIRENLayer:
    """Single fully-connected layer with sine activation (SIREN)."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # SIREN initialization (Sitzmann et al. 2020)
        if is_first:
            limit = 1.0 / in_features
        else:
            limit = np.sqrt(6.0 / in_features) / omega_0
        
        self.W = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.b = np.zeros((out_features, 1), dtype=np.float32)
        
        # Adam optimizer state
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0
    
    def forward(self, x):
        """x: (in_features, N) -> returns (out_features, N)"""
        self.x = x
        self.z = self.W @ x + self.b  # pre-activation
        if self.is_first:
            self.a = np.sin(self.omega_0 * self.z)
        else:
            self.a = np.sin(self.omega_0 * self.z)
        return self.a
    
    def backward(self, grad_output):
        """grad_output: (out_features, N)"""
        grad_z = grad_output * (self.omega_0 * np.cos(self.omega_0 * self.z))
        self.grad_W = grad_z @ self.x.T / self.x.shape[1]
        self.grad_b = np.mean(grad_z, axis=1, keepdims=True)
        grad_input = self.W.T @ grad_z
        return grad_input
    
    def step(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.grad_W ** 2)
        m_W_hat = self.m_W / (1 - beta1 ** self.t)
        v_W_hat = self.v_W / (1 - beta2 ** self.t)
        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.grad_b ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    
    def get_params(self):
        return {'W': self.W.tolist(), 'b': self.b.tolist(), 'omega_0': self.omega_0, 'is_first': self.is_first}
    
    def set_params(self, params):
        self.W = np.array(params['W'], dtype=np.float32)
        self.b = np.array(params['b'], dtype=np.float32)
        self.omega_0 = params['omega_0']
        self.is_first = params['is_first']

class SIREN:
    """Multi-layer SIREN for INR compression."""
    def __init__(self, layers_config, omega_0=30.0):
        """layers_config: list of ints, e.g. [1, 64, 64, 1] for 1D signal."""
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(SIRENLayer(layers_config[i], layers_config[i+1], omega_0=omega_0 if not is_first else 30.0, is_first=is_first))
    
    def forward(self, x):
        """x: (input_dim, N)"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def train_step(self, x, y, lr=1e-4):
        """Single training step. x: (input_dim, N), y: (output_dim, N)"""
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        grad = 2 * (pred - y) / y.shape[1]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        for layer in self.layers:
            layer.step(lr)
        return loss
    
    def fit(self, x, y, epochs=5000, lr=1e-4, batch_size=None, verbose=100):
        """Train the INR to map x -> y."""
        N = x.shape[1]
        if batch_size is None:
            batch_size = N
        
        for epoch in range(epochs):
            indices = np.random.permutation(N)[:batch_size]
            x_batch = x[:, indices]
            y_batch = y[:, indices]
            loss = self.train_step(x_batch, y_batch, lr)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6e}")
        return loss
    
    def save_weights(self, path):
        data = {'layers': [layer.get_params() for layer in self.layers]}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_weights(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        for layer, params in zip(self.layers, data['layers']):
            layer.set_params(params)

class DataINRCompressor:
    """High-level API: compress any 1D byte sequence into SIREN weights."""
    def __init__(self, hidden_dim=64, num_layers=3, omega_0=30.0):
        layers = [1] + [hidden_dim] * num_layers + [1]
        self.net = SIREN(layers, omega_0=omega_0)
    
    def compress(self, byte_array, epochs=3000, lr=1e-3):
        """
        byte_array: bytes or np.ndarray of uint8
        Returns: compressed metadata dict
        """
        if isinstance(byte_array, bytes):
            data = np.frombuffer(byte_array, dtype=np.uint8)
        else:
            data = np.array(byte_array, dtype=np.uint8)
        
        N = len(data)
        # Normalize to [-1, 1]
        y = (data.astype(np.float32) / 127.5) - 1.0
        
        # Coordinates in [0, 1]
        x = np.linspace(0, 1, N).reshape(1, N).astype(np.float32)
        
        self.net.fit(x, y.reshape(1, N), epochs=epochs, lr=lr, verbose=500)
        
        final_pred = self.net.forward(x)
        mse = np.mean((final_pred - y.reshape(1, N)) ** 2)
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        
        return {
            'original_size': N,
            'mse': float(mse),
            'psnr': float(psnr),
        }
    
    def reconstruct(self, num_samples):
        """Reconstruct the original byte sequence."""
        x = np.linspace(0, 1, num_samples).reshape(1, num_samples).astype(np.float32)
        y_pred = self.net.forward(x)
        # Denormalize from [-1, 1] to [0, 255]
        data = np.clip((y_pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return data.flatten()
    
    def save_recipe(self, path):
        self.net.save_weights(path)
    
    def load_recipe(self, path):
        self.net.load_weights(path)
