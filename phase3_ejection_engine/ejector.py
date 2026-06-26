#!/usr/bin/env python3
"""
Black Hole - Phase 3: Ejection Engine
Zero-copy ejection from recipe state to volatile memory (RAM).
Simulates direct memory mapping without disk round-trip.
"""
import numpy as np
import sys
import os
import time

class EjectionEngine:
    """
    The Jet of Information: mount reconstructed data directly into memory.
    """
    def __init__(self, recipe_dir='./recipes'):
        self.recipe_dir = recipe_dir
        self.cache = {}  # In-memory cache of reconstructed data
    
    def eject(self, recipe_path, target_size=None):
        """
        Load recipe, reconstruct, and return data already in memory.
        No disk write. Direct to RAM.
        """
        import importlib.util
        core_path = os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor', 'siren_core.py')
        spec = importlib.util.spec_from_file_location("siren_core", core_path)
        if spec is None:
            raise RuntimeError("Cannot load siren_core")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        DataINRCompressor = mod.DataINRCompressor
        
        compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
        compressor.load_recipe(recipe_path)
        
        if target_size is None:
            # Infer from metadata if available
            target_size = 1000  # fallback
        
        start = time.time()
        data = compressor.reconstruct(target_size)
        elapsed = time.time() - start
        
        self.cache[recipe_path] = data
        return data, elapsed
    
    def eject_direct(self, recipe_path, target_size):
        """
        Simulated zero-copy: return a memory view that appears 'instant'.
        In real implementation this would use io_uring / DirectStorage / mmap.
        """
        print(f"[Ejection] Zero-copy mounting: {recipe_path}")
        data, t = self.eject(recipe_path, target_size)
        print(f"[Ejection] Mounted in {t*1000:.2f} ms. No disk write.")
        return data
    
    def memory_pressure(self):
        """Report current in-memory pressure."""
        total = sum(d.nbytes for d in self.cache.values())
        return total

def demo():
    # First compress something
    core_path = os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor', 'siren_core.py')
    spec = importlib.util.spec_from_file_location("siren_core", core_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DataINRCompressor = mod.DataINRCompressor
    
    text = b"Black Hole ejection test. " * 50
    compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
    compressor.compress(text, epochs=2000, lr=1e-3)
    
    os.makedirs('./recipes', exist_ok=True)
    recipe = './recipes/ejection_test.recipe.json'
    compressor.save_recipe(recipe)
    
    engine = EjectionEngine()
    data = engine.eject_direct(recipe, len(text))
    
    match = np.mean(data == np.frombuffer(text, dtype=np.uint8))
    print(f"[Ejection] Accuracy: {match*100:.2f}%")
    print(f"[Ejection] Memory pressure: {engine.memory_pressure()} bytes")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        print("Usage: python ejector.py demo")
