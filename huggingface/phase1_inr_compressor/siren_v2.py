import numpy as np
import json

class SIRENLayerV2:
    """Enhanced SIREN layer with optional quantization support."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        if is_first:
            limit = 1.0 / in_features
        else:
            limit = np.sqrt(6.0 / in_features) / omega_0
        
        self.W = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.b = np.zeros((out_features, 1), dtype=np.float32)
        
        # Adam state
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0
    
    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.b
        self.a = np.sin(self.omega_0 * self.z)
        return self.a
    
    def backward(self, grad_output):
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
        return {'W': self.W.tolist(), 'b': self.b.tolist(), 
                'omega_0': self.omega_0, 'is_first': self.is_first}
    
    def set_params(self, params):
        self.W = np.array(params['W'], dtype=np.float32)
        self.b = np.array(params['b'], dtype=np.float32)
        self.omega_0 = params['omega_0']
        self.is_first = params['is_first']
    
    def quantize_weights(self, bits=8):
        """Quantize weights to INT8 or INT4."""
        W_min, W_max = self.W.min(), self.W.max()
        b_min, b_max = self.b.min(), self.b.max()
        
        if bits == 8:
            scale_W = (W_max - W_min) / 255.0
            scale_b = (b_max - b_min) / 255.0
            W_q = np.round((self.W - W_min) / scale_W).astype(np.uint8)
            b_q = np.round((self.b - b_min) / scale_b).astype(np.uint8)
        elif bits == 4:
            scale_W = (W_max - W_min) / 15.0
            scale_b = (b_max - b_min) / 15.0
            W_q = np.round((self.W - W_min) / scale_W).astype(np.uint8)
            b_q = np.round((self.b - b_min) / scale_b).astype(np.uint8)
        else:
            raise ValueError("bits must be 8 or 4")
        
        return {
            'W': W_q.tolist(), 'b': b_q.tolist(),
            'W_min': float(W_min), 'W_max': float(W_max),
            'b_min': float(b_min), 'b_max': float(b_max),
            'bits': bits, 'omega_0': self.omega_0, 'is_first': self.is_first
        }
    
    def dequantize_weights(self, quantized):
        """Dequantize weights back to FP32."""
        bits = quantized['bits']
        W_q = np.array(quantized['W'], dtype=np.float32)
        b_q = np.array(quantized['b'], dtype=np.float32)
        W_min, W_max = quantized['W_min'], quantized['W_max']
        b_min, b_max = quantized['b_min'], quantized['b_max']
        
        if bits == 8:
            scale_W = (W_max - W_min) / 255.0
            scale_b = (b_max - b_min) / 255.0
        elif bits == 4:
            scale_W = (W_max - W_min) / 15.0
            scale_b = (b_max - b_min) / 15.0
        
        self.W = W_q * scale_W + W_min
        self.b = b_q * scale_b + b_min
        self.omega_0 = quantized['omega_0']
        self.is_first = quantized['is_first']

class SIREN2D:
    """2D SIREN with positional encoding for image compression."""
    def __init__(self, layers_config, omega_0=30.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(SIRENLayerV2(layers_config[i], layers_config[i+1], 
                                            omega_0=omega_0 if not is_first else 30.0, 
                                            is_first=is_first))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def train_step(self, x, y, lr=1e-4):
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        grad = 2 * (pred - y) / y.shape[1]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        for layer in self.layers:
            layer.step(lr)
        return loss
    
    def fit(self, x, y, epochs=5000, lr=1e-4, batch_size=None, verbose=100):
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
    
    def save_weights(self, path, quantized=False, bits=8):
        if quantized:
            data = {'layers': [layer.quantize_weights(bits) for layer in self.layers],
                    'quantized': True}
        else:
            data = {'layers': [layer.get_params() for layer in self.layers],
                    'quantized': False}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_weights(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get('quantized', False):
            for layer, qparams in zip(self.layers, data['layers']):
                layer.dequantize_weights(qparams)
        else:
            for layer, params in zip(self.layers, data['layers']):
                layer.set_params(params)

class ImageINRCompressor:
    """Compress 2D images using 2D SIREN with positional encoding."""
    def __init__(self, hidden_dim=64, num_layers=3, omega_0=30.0):
        # 2D coordinates + optional positional encoding
        layers = [2] + [hidden_dim] * num_layers + [3]  # RGB output
        self.net = SIREN2D(layers, omega_0=omega_0)
        self.H = None
        self.W = None
    
    def compress(self, image_array, epochs=3000, lr=1e-4):
        """
        image_array: (H, W, 3) uint8 RGB image
        """
        H, W, C = image_array.shape
        self.H, self.W = H, W
        
        # Normalize to [-1, 1]
        y = (image_array.astype(np.float32) / 127.5) - 1.0
        y = y.reshape(-1, C).T  # (3, H*W)
        
        # 2D coordinates
        x_coords = np.linspace(-1, 1, W)
        y_coords = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=0).astype(np.float32)
        
        self.net.fit(coords, y, epochs=epochs, lr=lr, verbose=500)
        
        final_pred = self.net.forward(coords)
        mse = np.mean((final_pred - y) ** 2)
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        
        return {'original_shape': (H, W, C), 'mse': float(mse), 'psnr': float(psnr)}
    
    def reconstruct(self):
        if self.H is None or self.W is None:
            raise ValueError("Must compress first")
        
        x_coords = np.linspace(-1, 1, self.W)
        y_coords = np.linspace(-1, 1, self.H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=0).astype(np.float32)
        
        y_pred = self.net.forward(coords)
        img = np.clip((y_pred.T + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return img.reshape(self.H, self.W, 3)
    
    def save_recipe(self, path, quantized=True, bits=8):
        self.net.save_weights(path, quantized=quantized, bits=bits)
    
    def load_recipe(self, path):
        self.net.load_weights(path)
