#!/usr/bin/env python3
"""
Black Hole - Phase 2: Opportunistic Compute Daemon
Monitors system idle cycles and pre-calculates INR recipes in background.
"""
import time
import os
import sys
import json
import threading
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[Daemon] Warning: psutil not found. Using dummy CPU monitor.")

class OpportunisticDaemon:
    """
    The Horizon of Events: pre-calculates data recipes during idle CPU cycles.
    """
    def __init__(self, recipe_dir='./recipes', idle_threshold=15.0, check_interval=2.0):
        self.recipe_dir = recipe_dir
        self.idle_threshold = idle_threshold  # CPU % below this = idle
        self.check_interval = check_interval
        self.queue = deque()
        self.running = False
        self.stats = {'idle_time': 0.0, 'work_done': 0, 'energy_saved_estimate': 0.0}
        os.makedirs(recipe_dir, exist_ok=True)
    
    def cpu_idle_percent(self):
        if HAS_PSUTIL:
            return psutil.cpu_percent(interval=0.5)
        return 0.0  # Assume always idle if no psutil
    
    def is_idle(self):
        return self.cpu_idle_percent() < self.idle_threshold
    
    def enqueue(self, filepath, priority=1):
        """Add a file to the pre-calculation queue."""
        task = {'file': filepath, 'priority': priority, 'status': 'pending'}
        self.queue.append(task)
        print(f"[Daemon] Enqueued: {filepath} (priority {priority})")
    
    def precompute_task(self, task):
        """Run a lightweight training step (simulated or real)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("siren_core", 
            os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor', 'siren_core.py'))
        if spec is None:
            print(f"[Daemon] Cannot load siren_core, skipping.")
            return False
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        DataINRCompressor = mod.DataINRCompressor
        
        filepath = task['file']
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Limit to small files for background processing
        if len(data) > 50000:
            print(f"[Daemon] File too large for background: {filepath}")
            return False
        
        compressor = DataINRCompressor(hidden_dim=32, num_layers=2)
        compressor.compress(data, epochs=500, lr=1e-3)
        
        recipe_path = os.path.join(self.recipe_dir, os.path.basename(filepath) + '.recipe.json')
        compressor.save_recipe(recipe_path)
        task['recipe'] = recipe_path
        return True
    
    def run(self):
        self.running = True
        print(f"[Daemon] Horizon of Events started. Idle threshold: {self.idle_threshold}%")
        while self.running:
            if self.is_idle() and len(self.queue) > 0:
                task = self.queue.popleft()
                print(f"[Daemon] Opportunistic cycle detected. Processing: {task['file']}")
                start = time.time()
                success = self.precompute_task(task)
                elapsed = time.time() - start
                if success:
                    self.stats['work_done'] += 1
                    self.stats['idle_time'] += elapsed
                    self.stats['energy_saved_estimate'] += elapsed * 0.5  # heuristic
                    print(f"[Daemon] Pre-computed in {elapsed:.2f}s -> {task.get('recipe')}")
                else:
                    self.queue.appendleft(task)  # retry later
            else:
                time.sleep(self.check_interval)
    
    def stop(self):
        self.running = False
        print(f"[Daemon] Stopping. Stats: {self.stats}")
    
    def save_stats(self, path):
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)

def demo():
    daemon = OpportunisticDaemon(idle_threshold=50.0)  # generous for demo
    # Create a dummy file to process
    dummy = 'dummy_test.txt'
    with open(dummy, 'w') as f:
        f.write('Hello Black Hole! ' * 100)
    
    daemon.enqueue(dummy)
    
    # Run for a limited time
    def limited_run():
        daemon.run()
    
    t = threading.Thread(target=limited_run)
    t.start()
    time.sleep(15)
    daemon.stop()
    t.join()
    daemon.save_stats('daemon_stats.json')
    print("[Daemon] Demo complete. Check daemon_stats.json")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        daemon = OpportunisticDaemon()
        daemon.run()
