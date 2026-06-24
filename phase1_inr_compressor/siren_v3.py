import numpy as np
import struct

# ============ BINARY PACKING UTILS ============

def binary_pack_weights(layers_data, bits=8):
    """Pack quantized weights into compact binary format (.blkh)."""
    header = struct.pack('<H', len(layers_data))  # num layers
    header += struct.pack('<B', bits)  # bits per weight
    
    body = b''
    for layer in layers_data:
        W = np.array(layer['W'], dtype=np.uint8)
        b = np.array(layer['b'], dtype=np.uint8)
        W_shape = W.shape
        b_shape = b.shape
        
        meta = struct.pack('<HH', W_shape[0], W_shape[1])
        meta += struct.pack('<H', b_shape[0])
        meta += struct.pack('<ffff', layer['W_min'], layer['W_max'], layer['b_min'], layer['b_max'])
        
        body += meta + W.tobytes() + b.tobytes()
    
    return header + body

def binary_unpack_weights(data):
    """Unpack binary weights back to quantized format."""
    num_layers = struct.unpack('<H', data[:2])[0]
    bits = struct.unpack('<B', data[2:3])[0]
    offset = 3
    
    layers = []
    for _ in range(num_layers):
        W_rows, W_cols = struct.unpack('<HH', data[offset:offset+4])
        offset += 4
        b_rows = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        W_min, W_max, b_min, b_max = struct.unpack('<ffff', data[offset:offset+16])
        offset += 16
        
        W_size = W_rows * W_cols
        b_size = b_rows
        W = np.frombuffer(data[offset:offset+W_size], dtype=np.uint8).reshape(W_rows, W_cols)
        offset += W_size
        b = np.frombuffer(data[offset:offset+b_size], dtype=np.uint8).reshape(b_rows, 1)
        offset += b_size
        
        layers.append({
            'W': W.tolist(), 'b': b.tolist(),
            'W_min': W_min, 'W_max': W_max,
            'b_min': b_min, 'b_max': b_max,
            'bits': bits, 'omega_0': 30.0, 'is_first': False
        })
    
    if layers:
        layers[0]['is_first'] = True
        layers[0]['omega_0'] = 30.0
    
    return layers

# ============ SIREN LAYER V3 ============

class SIRENLayerV3:
    """Lightweight SIREN layer with INT8 quantization."""
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
    
    def quantize(self, bits=8):
        W_min, W_max = self.W.min(), self.W.max()
        b_min, b_max = self.b.min(), self.b.max()
        
        if bits == 8:
            scale_W = (W_max - W_min) / 255.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 255.0 if b_max != b_min else 1.0
        elif bits == 4:
            scale_W = (W_max - W_min) / 15.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 15.0 if b_max != b_min else 1.0
        else:
            raise ValueError("bits must be 8 or 4")
        
        W_q = np.round((self.W - W_min) / scale_W).astype(np.uint8) if W_max != W_min else np.zeros_like(self.W, dtype=np.uint8)
        b_q = np.round((self.b - b_min) / scale_b).astype(np.uint8) if b_max != b_min else np.zeros_like(self.b, dtype=np.uint8)
        
        return {
            'W': W_q, 'b': b_q,
            'W_min': float(W_min), 'W_max': float(W_max),
            'b_min': float(b_min), 'b_max': float(b_max),
            'bits': bits, 'omega_0': self.omega_0, 'is_first': self.is_first
        }
    
    def dequantize(self, q):
        bits = q['bits']
        W_q = np.array(q['W'], dtype=np.float32)
        b_q = np.array(q['b'], dtype=np.float32)
        W_min, W_max = q['W_min'], q['W_max']
        b_min, b_max = q['b_min'], q['b_max']
        
        if bits == 8:
            scale_W = (W_max - W_min) / 255.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 255.0 if b_max != b_min else 1.0
        elif bits == 4:
            scale_W = (W_max - W_min) / 15.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 15.0 if b_max != b_min else 1.0
        
        self.W = W_q * scale_W + W_min
        self.b = b_q * scale_b + b_min
        self.omega_0 = q['omega_0']
        self.is_first = q['is_first']

# ============ SIREN 2D NETWORK ============

class SIREN2DV3:
    """2D SIREN network with binary weight packing."""
    def __init__(self, layers_config, omega_0=30.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(SIRENLayerV3(
                layers_config[i], layers_config[i+1],
                omega_0=omega_0 if not is_first else 30.0, is_first=is_first
            ))
    
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
    
    def save_binary(self, path, bits=8):
        q_layers = [layer.quantize(bits) for layer in self.layers]
        binary = binary_pack_weights(q_layers, bits)
        with open(path, 'wb') as f:
            f.write(binary)
    
    def load_binary(self, path):
        with open(path, 'rb') as f:
            binary = f.read()
        q_layers = binary_unpack_weights(binary)
        for layer, q in zip(self.layers, q_layers):
            layer.dequantize(q)

# ============ IMAGE COMPRESSOR ============

class ImageINRV3:
    """High-level 2D image compressor using binary-packed SIREN."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [2] + [hidden_dim] * num_layers + [3]
        self.net = SIREN2DV3(layers, omega_0=omega_0)
        self.H = None
        self.W = None
    
    def compress(self, image_array, epochs=3000, lr=1e-4):
        H, W, C = image_array.shape
        self.H, self.W = H, W
        
        y = (image_array.astype(np.float32) / 127.5) - 1.0
        y = y.reshape(-1, C).T
        
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
    
    def save_recipe(self, path, bits=8):
        self.net.save_binary(path, bits=bits)
    
    def load_recipe(self, path):
        self.net.load_binary(path)

# ============ 1D SIGNAL COMPRESSOR ============

class SignalINRV3:
    """1D signal compressor (audio, time series)."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [1] + [hidden_dim] * num_layers + [1]
        self.net = SIREN2DV3(layers, omega_0=omega_0)
        self.N = None
    
    def compress(self, signal, epochs=3000, lr=1e-4):
        if isinstance(signal, bytes):
            data = np.frombuffer(signal, dtype=np.uint8)
        else:
            data = np.array(signal, dtype=np.uint8)
        
        self.N = len(data)
        y = (data.astype(np.float32) / 127.5) - 1.0
        x = np.linspace(-1, 1, self.N).reshape(1, self.N).astype(np.float32)
        
        self.net.fit(x, y.reshape(1, self.N), epochs=epochs, lr=lr, verbose=500)
        
        final_pred = self.net.forward(x)
        mse = np.mean((final_pred - y.reshape(1, self.N)) ** 2)
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        
        return {'original_size': self.N, 'mse': float(mse), 'psnr': float(psnr)}
    
    def reconstruct(self, N=None):
        if N is None:
            N = self.N
        x = np.linspace(-1, 1, N).reshape(1, N).astype(np.float32)
        y_pred = self.net.forward(x)
        data = np.clip((y_pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return data.flatten()
    
    def save_recipe(self, path, bits=8):
        self.net.save_binary(path, bits=bits)
    
    def load_recipe(self, path):
        self.net.load_binary(path)
