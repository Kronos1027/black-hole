import numpy as np
import struct
import os

# ============ 4-BIT QUANTIZATION WITH PACKING ============

def pack_4bit(array):
    """Pack two uint4 values into one uint8."""
    flat = array.flatten()
    if len(flat) % 2 == 1:
        flat = np.concatenate([flat, np.array([0], dtype=np.uint8)])
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed

def unpack_4bit(packed, total_elements):
    """Unpack uint8 into two uint4 values."""
    flat = np.zeros(total_elements, dtype=np.uint8)
    unpacked_high = (packed >> 4) & 0x0F
    unpacked_low = packed & 0x0F
    flat[0::2] = unpacked_high[:len(flat[0::2])]
    flat[1::2] = unpacked_low[:len(flat[1::2])]
    return flat[:total_elements]

def binary_pack_4bit(layers_data):
    """Pack 4-bit quantized weights into ultra-compact binary."""
    header = struct.pack('<H', len(layers_data))
    header += struct.pack('<B', 4)  # 4 bits
    
    body = b''
    for layer in layers_data:
        W = np.array(layer['W'], dtype=np.uint8)
        b = np.array(layer['b'], dtype=np.uint8)
        W_shape = W.shape
        b_shape = b.shape
        
        meta = struct.pack('<HH', W_shape[0], W_shape[1])
        meta += struct.pack('<H', b_shape[0])
        meta += struct.pack('<ffff', layer['W_min'], layer['W_max'], layer['b_min'], layer['b_max'])
        
        # Pack 4-bit weights
        W_packed = pack_4bit(W)
        b_packed = pack_4bit(b)
        
        body += meta + W_packed.tobytes() + b_packed.tobytes()
    
    return header + body

def binary_unpack_4bit(data):
    """Unpack 4-bit binary weights."""
    num_layers = struct.unpack('<H', data[:2])[0]
    bits = struct.unpack('<B', data[2:3])[0]
    assert bits == 4, "Expected 4-bit data"
    offset = 3
    
    layers = []
    for _ in range(num_layers):
        W_rows, W_cols = struct.unpack('<HH', data[offset:offset+4])
        offset += 4
        b_rows = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        W_min, W_max, b_min, b_max = struct.unpack('<ffff', data[offset:offset+16])
        offset += 16
        
        W_elements = W_rows * W_cols
        b_elements = b_rows
        
        W_packed_size = (W_elements + 1) // 2
        b_packed_size = (b_elements + 1) // 2
        
        W_packed = np.frombuffer(data[offset:offset+W_packed_size], dtype=np.uint8)
        offset += W_packed_size
        b_packed = np.frombuffer(data[offset:offset+b_packed_size], dtype=np.uint8)
        offset += b_packed_size
        
        W = unpack_4bit(W_packed, W_elements).reshape(W_rows, W_cols)
        b = unpack_4bit(b_packed, b_elements).reshape(b_rows, 1)
        
        layers.append({
            'W': W.tolist(), 'b': b.tolist(),
            'W_min': W_min, 'W_max': W_max,
            'b_min': b_min, 'b_max': b_max,
            'bits': 4, 'omega_0': 30.0, 'is_first': False
        })
    
    if layers:
        layers[0]['is_first'] = True
        layers[0]['omega_0'] = 30.0
    
    return layers

# ============ PRUNING UTILS ============

def prune_weights(W, threshold=0.01):
    """Prune small weights to zero (magnitude-based pruning)."""
    W_pruned = W.copy()
    mask = np.abs(W) < threshold * np.abs(W).max()
    W_pruned[mask] = 0
    return W_pruned, mask

def count_sparsity(W):
    """Return sparsity percentage."""
    return np.mean(W == 0) * 100

# ============ SIREN LAYER V4 ============

class SIRENLayerV4:
    """Ultra-lightweight SIREN with 4-bit quantization + pruning."""
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
    
    def quantize(self, bits=4, prune_threshold=0.0):
        W_pruned, mask = prune_weights(self.W, prune_threshold) if prune_threshold > 0 else (self.W, None)
        b_pruned, _ = prune_weights(self.b, prune_threshold) if prune_threshold > 0 else (self.b, None)
        
        W_min, W_max = W_pruned.min(), W_pruned.max()
        b_min, b_max = b_pruned.min(), b_pruned.max()
        
        if bits == 8:
            scale_W = (W_max - W_min) / 255.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 255.0 if b_max != b_min else 1.0
        elif bits == 4:
            scale_W = (W_max - W_min) / 15.0 if W_max != W_min else 1.0
            scale_b = (b_max - b_min) / 15.0 if b_max != b_min else 1.0
        else:
            raise ValueError("bits must be 4 or 8")
        
        W_q = np.round((W_pruned - W_min) / scale_W).astype(np.uint8) if W_max != W_min else np.zeros_like(W_pruned, dtype=np.uint8)
        b_q = np.round((b_pruned - b_min) / scale_b).astype(np.uint8) if b_max != b_min else np.zeros_like(b_pruned, dtype=np.uint8)
        
        return {
            'W': W_q, 'b': b_q,
            'W_min': float(W_min), 'W_max': float(W_max),
            'b_min': float(b_min), 'b_max': float(b_max),
            'bits': bits, 'omega_0': self.omega_0, 'is_first': self.is_first,
            'sparsity': float(count_sparsity(W_pruned)) if prune_threshold > 0 else 0.0
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

# ============ SIREN 2D NETWORK V4 ============

class SIREN2DV4:
    """2D SIREN with 4-bit quantization and pruning."""
    def __init__(self, layers_config, omega_0=30.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(SIRENLayerV4(
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
        
        # Learning rate schedule: warmup + decay
        for epoch in range(epochs):
            # Cosine annealing
            lr_current = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            
            indices = np.random.permutation(N)[:batch_size]
            x_batch = x[:, indices]
            y_batch = y[:, indices]
            loss = self.train_step(x_batch, y_batch, lr_current)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6e}, LR: {lr_current:.6e}")
        return loss
    
    def save_binary(self, path, bits=4, prune_threshold=0.01):
        q_layers = [layer.quantize(bits, prune_threshold) for layer in self.layers]
        if bits == 4:
            binary = binary_pack_4bit(q_layers)
        else:
            # Import from siren_v3 module (works as script or package)
            try:
                from .siren_v3 import binary_pack_weights
            except ImportError:
                from siren_v3 import binary_pack_weights
            binary = binary_pack_weights(q_layers, bits)
        with open(path, 'wb') as f:
            f.write(binary)
    
    def load_binary(self, path):
        with open(path, 'rb') as f:
            binary = f.read()
        bits = struct.unpack('<B', binary[2:3])[0]
        if bits == 4:
            q_layers = binary_unpack_4bit(binary)
        else:
            try:
                from .siren_v3 import binary_unpack_weights
            except ImportError:
                from siren_v3 import binary_unpack_weights
            q_layers = binary_unpack_weights(binary)
        for layer, q in zip(self.layers, q_layers):
            layer.dequantize(q)

# ============ IMAGE COMPRESSOR V4 ============

class ImageINRV4:
    """v4: 4-bit quantization + pruning + cosine LR + 2D encoding."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [2] + [hidden_dim] * num_layers + [3]
        self.net = SIREN2DV4(layers, omega_0=omega_0)
        self.H = None
        self.W = None
    
    def compress(self, image_array, epochs=3000, lr=1e-3):
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
    
    def save_recipe(self, path, bits=4, prune_threshold=0.01):
        self.net.save_binary(path, bits=bits, prune_threshold=prune_threshold)
    
    def load_recipe(self, path):
        self.net.load_binary(path)

# ============ SIGNAL COMPRESSOR V4 ============

class SignalINRV4:
    """1D signal compressor with v4 features."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [1] + [hidden_dim] * num_layers + [1]
        self.net = SIREN2DV4(layers, omega_0=omega_0)
        self.N = None
    
    def compress(self, signal, epochs=3000, lr=1e-3):
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
    
    def save_recipe(self, path, bits=4, prune_threshold=0.01):
        self.net.save_binary(path, bits=bits, prune_threshold=prune_threshold)
    
    def load_recipe(self, path):
        self.net.load_binary(path)

# ============ META-LEARNING V4 ============

class MetaSIRENLayerV4:
    """Meta-learning layer: base weights + modulations."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        if is_first:
            limit = 1.0 / in_features
        else:
            limit = np.sqrt(6.0 / in_features) / omega_0
        
        self.W_base = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.b_base = np.zeros((out_features, 1), dtype=np.float32)
        self.modulation = np.ones((out_features, 1), dtype=np.float32)
        
        self.m_W = np.zeros_like(self.W_base)
        self.v_W = np.zeros_like(self.W_base)
        self.m_b = np.zeros_like(self.b_base)
        self.v_b = np.zeros_like(self.b_base)
        self.m_mod = np.zeros_like(self.modulation)
        self.v_mod = np.zeros_like(self.modulation)
        self.t = 0
    
    def forward(self, x):
        self.x = x
        self.z = self.W_base @ x + self.b_base
        self.a = np.sin(self.omega_0 * self.z) * self.modulation
        return self.a
    
    def backward(self, grad_output):
        grad_mod = np.mean(grad_output * np.sin(self.omega_0 * self.z), axis=1, keepdims=True)
        grad_z = grad_output * self.modulation * (self.omega_0 * np.cos(self.omega_0 * self.z))
        self.grad_W = grad_z @ self.x.T / self.x.shape[1]
        self.grad_b = np.mean(grad_z, axis=1, keepdims=True)
        self.grad_mod = grad_mod
        grad_input = self.W_base.T @ grad_z
        return grad_input
    
    def step(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, train_base=True):
        self.t += 1
        
        if train_base:
            self.m_W = beta1 * self.m_W + (1 - beta1) * self.grad_W
            self.v_W = beta2 * self.v_W + (1 - beta2) * (self.grad_W ** 2)
            m_W_hat = self.m_W / (1 - beta1 ** self.t)
            v_W_hat = self.v_W / (1 - beta2 ** self.t)
            self.W_base -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.grad_b
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.grad_b ** 2)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)
            self.b_base -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        
        self.m_mod = beta1 * self.m_mod + (1 - beta1) * self.grad_mod
        self.v_mod = beta2 * self.v_mod + (1 - beta2) * (self.grad_mod ** 2)
        m_mod_hat = self.m_mod / (1 - beta1 ** self.t)
        v_mod_hat = self.v_mod / (1 - beta2 ** self.t)
        self.modulation -= lr * m_mod_hat / (np.sqrt(v_mod_hat) + eps)
    
    def get_modulation(self):
        return self.modulation.copy()
    
    def set_modulation(self, mod):
        self.modulation = mod.copy()
    
    def quantize_modulation(self, bits=4):
        m_min, m_max = self.modulation.min(), self.modulation.max()
        if bits == 4:
            scale = (m_max - m_min) / 15.0 if m_max != m_min else 1.0
        else:
            scale = (m_max - m_min) / 255.0 if m_max != m_min else 1.0
        m_q = np.round((self.modulation - m_min) / scale).astype(np.uint8) if m_max != m_min else np.zeros_like(self.modulation, dtype=np.uint8)
        return {'m': m_q.tolist(), 'm_min': float(m_min), 'm_max': float(m_max), 'bits': bits}
    
    def dequantize_modulation(self, q):
        bits = q['bits']
        m_q = np.array(q['m'], dtype=np.float32)
        m_min, m_max = q['m_min'], q['m_max']
        if bits == 4:
            scale = (m_max - m_min) / 15.0 if m_max != m_min else 1.0
        else:
            scale = (m_max - m_min) / 255.0 if m_max != m_min else 1.0
        self.modulation = m_q * scale + m_min

class MetaSIREN2DV4:
    """Meta-learning 2D network."""
    def __init__(self, layers_config, omega_0=30.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(MetaSIRENLayerV4(
                layers_config[i], layers_config[i+1],
                omega_0=omega_0 if not is_first else 30.0, is_first=is_first
            ))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def train_step(self, x, y, lr=1e-4, train_base=True):
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        grad = 2 * (pred - y) / y.shape[1]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        for layer in self.layers:
            layer.step(lr, train_base=train_base)
        return loss
    
    def fit(self, x, y, epochs=5000, lr=1e-4, batch_size=None, verbose=100, train_base=True):
        N = x.shape[1]
        if batch_size is None:
            batch_size = N
        for epoch in range(epochs):
            indices = np.random.permutation(N)[:batch_size]
            x_batch = x[:, indices]
            y_batch = y[:, indices]
            loss = self.train_step(x_batch, y_batch, lr, train_base=train_base)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6e}")
        return loss
    
    def save_base(self, path):
        import json
        data = {'base': [{'W': l.W_base.tolist(), 'b': l.b_base.tolist()} for l in self.layers]}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_base(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        for layer, p in zip(self.layers, data['base']):
            layer.W_base = np.array(p['W'], dtype=np.float32)
            layer.b_base = np.array(p['b'], dtype=np.float32)
    
    def save_modulations(self, path, bits=4):
        mods = [layer.quantize_modulation(bits) for layer in self.layers]
        import json
        with open(path, 'w') as f:
            json.dump({'mods': mods}, f)
    
    def load_modulations(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        for layer, q in zip(self.layers, data['mods']):
            layer.dequantize_modulation(q)

class MetaImageCompressorV4:
    """High-level meta-learning compressor."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [2] + [hidden_dim] * num_layers + [3]
        self.net = MetaSIREN2DV4(layers, omega_0=omega_0)
        self.H = None
        self.W = None
        self.base_trained = False
    
    def train_base(self, images, epochs=3000, lr=1e-4):
        print(f"[Meta] Training base on {len(images)} images...")
        for epoch in range(epochs):
            img = images[np.random.randint(len(images))]
            H, W, C = img.shape
            y = (img.astype(np.float32) / 127.5) - 1.0
            y = y.reshape(-1, C).T
            
            x_coords = np.linspace(-1, 1, W)
            y_coords = np.linspace(-1, 1, H)
            xx, yy = np.meshgrid(x_coords, y_coords)
            coords = np.stack([xx.ravel(), yy.ravel()], axis=0).astype(np.float32)
            
            loss = self.net.train_step(coords, y, lr=lr, train_base=True)
            if epoch % 500 == 0:
                print(f"  Epoch {epoch}/{epochs}, Loss: {loss:.6e}")
        
        self.base_trained = True
        print("[Meta] Base trained.")
    
    def compress(self, image_array, epochs=500, lr=1e-3):
        if not self.base_trained:
            raise ValueError("Base must be trained first")
        
        H, W, C = image_array.shape
        self.H, self.W = H, W
        
        y = (image_array.astype(np.float32) / 127.5) - 1.0
        y = y.reshape(-1, C).T
        
        x_coords = np.linspace(-1, 1, W)
        y_coords = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=0).astype(np.float32)
        
        # Reset modulations
        for layer in self.net.layers:
            layer.modulation = np.ones_like(layer.modulation)
        
        # Only train modulations (fast!)
        self.net.fit(coords, y, epochs=epochs, lr=lr, verbose=100, train_base=False)
        
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
    
    def save_modulations(self, path, bits=4):
        self.net.save_modulations(path, bits=bits)
    
    def load_modulations(self, path):
        self.net.load_modulations(path)
