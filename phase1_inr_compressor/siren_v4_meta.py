import numpy as np
import struct
import os

# ============ META-LEARNING: COIN++ STYLE ============
# Instead of training a full network per image, we train a BASE network once
# and store only LIGHTWEIGHT MODULATIONS per image.

class MetaSIRENLayer:
    """SIREN layer with modulation support (COIN++ style)."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Base weights (shared across all images - meta-learned)
        if is_first:
            limit = 1.0 / in_features
        else:
            limit = np.sqrt(6.0 / in_features) / omega_0
        
        self.W_base = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.b_base = np.zeros((out_features, 1), dtype=np.float32)
        
        # Modulation (per-image, tiny)
        self.modulation = np.ones((out_features, 1), dtype=np.float32)
        
        # Adam state
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
        # Grad through modulation
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
        
        # Always train modulation
        self.m_mod = beta1 * self.m_mod + (1 - beta1) * self.grad_mod
        self.v_mod = beta2 * self.v_mod + (1 - beta2) * (self.grad_mod ** 2)
        m_mod_hat = self.m_mod / (1 - beta1 ** self.t)
        v_mod_hat = self.v_mod / (1 - beta2 ** self.t)
        self.modulation -= lr * m_mod_hat / (np.sqrt(v_mod_hat) + eps)
    
    def get_modulation(self):
        return self.modulation.copy()
    
    def set_modulation(self, mod):
        self.modulation = mod.copy()
    
    def get_base_params(self):
        return {'W': self.W_base.tolist(), 'b': self.b_base.tolist()}
    
    def set_base_params(self, params):
        self.W_base = np.array(params['W'], dtype=np.float32)
        self.b_base = np.array(params['b'], dtype=np.float32)

class MetaSIREN2D:
    """2D Meta-SIREN: one base network, lightweight modulations per image."""
    def __init__(self, layers_config, omega_0=30.0):
        self.layers = []
        for i in range(len(layers_config) - 1):
            is_first = (i == 0)
            self.layers.append(MetaSIRENLayer(
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
    
    def get_modulations(self):
        return [layer.get_modulation() for layer in self.layers]
    
    def set_modulations(self, mods):
        for layer, mod in zip(self.layers, mods):
            layer.set_modulation(mod)
    
    def get_base_params(self):
        return [layer.get_base_params() for layer in self.layers]
    
    def set_base_params(self, params):
        for layer, p in zip(self.layers, params):
            layer.set_base_params(p)
    
    def save_base(self, path):
        import json
        data = {'base': self.get_base_params(), 'layers_config': [l.in_features for l in self.layers] + [self.layers[-1].out_features]}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_base(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.set_base_params(data['base'])
    
    def save_modulations(self, path):
        import json
        mods = self.get_modulations()
        data = {'modulations': [m.tolist() for m in mods]}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_modulations(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        mods = [np.array(m, dtype=np.float32) for m in data['modulations']]
        self.set_modulations(mods)

class MetaImageCompressor:
    """High-level image compressor using meta-learning."""
    def __init__(self, hidden_dim=32, num_layers=2, omega_0=30.0):
        layers = [2] + [hidden_dim] * num_layers + [3]
        self.net = MetaSIREN2D(layers, omega_0=omega_0)
        self.H = None
        self.W = None
        self.base_trained = False
    
    def train_base(self, images, epochs=5000, lr=1e-4):
        """Train the base network on multiple images."""
        print(f"[Meta] Training base network on {len(images)} images...")
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
        print("[Meta] Base network trained.")
    
    def compress(self, image_array, epochs=1000, lr=1e-3):
        """Compress an image by only learning modulations (fast!)."""
        if not self.base_trained:
            raise ValueError("Base network must be trained first")
        
        H, W, C = image_array.shape
        self.H, self.W = H, W
        
        y = (image_array.astype(np.float32) / 127.5) - 1.0
        y = y.reshape(-1, C).T
        
        x_coords = np.linspace(-1, 1, W)
        y_coords = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=0).astype(np.float32)
        
        # Reset modulations to ones
        for layer in self.net.layers:
            layer.modulation = np.ones_like(layer.modulation)
        
        # Only train modulations, freeze base
        self.net.fit(coords, y, epochs=epochs, lr=lr, verbose=200, train_base=False)
        
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
    
    def save_modulations(self, path):
        self.net.save_modulations(path)
    
    def load_modulations(self, path):
        self.net.load_modulations(path)
