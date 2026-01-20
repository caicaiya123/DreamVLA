import torch
import torch.multiprocessing as mp
import time
import threading
import sys
import os
import gc
import signal
import argparse

# 全局标志，用于优雅退出
shutdown = False

def signal_handler(sig, frame):
    global shutdown
    print("\n🛑 Received interrupt signal. Shutting down gracefully...")
    shutdown = True

class GPUTaskController:
    def __init__(self, device, target_bytes, check_interval=5):
        self.device = device
        self.target_bytes = target_bytes
        self.check_interval = check_interval
        self.pause = False

    def should_pause(self):
        used_bytes = torch.cuda.memory_allocated(self.device)
        if self.pause:
            if used_bytes < 10 * 1024**3:  # 恢复阈值：10GB
                self.pause = False
                print(f"✅ GPU {self.device.index}: Resuming (used: {used_bytes / 1024**3:.1f} GB)")
        else:
            if used_bytes > (self.target_bytes + 5 * 1024**3):  # 超过目标+5GB则暂停
                self.pause = True
                print(f"⏸️ GPU {self.device.index}: Pausing due to high memory usage ({used_bytes / 1024**3:.1f} GB)")
        return self.pause

def gpu_task(gpu_id, target_gb, chunk_gb):
    global shutdown
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    target_bytes = int(target_gb * 1024**3)
    chunk_bytes = int(chunk_gb * 1024**3)

    print(f"🚀 Starting GPU task on GPU {gpu_id} (target: {target_gb} GB)")

    controller = GPUTaskController(device, target_bytes=target_bytes)
    allocated_tensors = []

    try:
        while not shutdown:
            if controller.should_pause():
                del allocated_tensors[:]
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(controller.check_interval)
                continue

            current_bytes = torch.cuda.memory_allocated(device)
            remaining_bytes = target_bytes - current_bytes

            if remaining_bytes <= 0:
                time.sleep(1)
                continue

            this_chunk = min(chunk_bytes, remaining_bytes)
            num_elements = this_chunk // 2  # float16 = 2 bytes/element

            if num_elements <= 0:
                time.sleep(0.5)
                continue

            try:
                tensor = torch.empty(num_elements, dtype=torch.float16, device=device)
                allocated_tensors.append(tensor)
                new_total_gb = (current_bytes + this_chunk) / 1024**3
                print(f"📈 GPU {gpu_id}: Allocated {this_chunk / 1024**3:.1f} GB → Total: {new_total_gb:.1f} GB")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ GPU {gpu_id}: CUDA OOM at ~{current_bytes / 1024**3:.1f} GB. Stopping allocation.")
                    break
                else:
                    raise

            time.sleep(0.1)

    except Exception as e:
        print(f"❌ GPU {gpu_id} error: {e}")
    finally:
        print(f"🧹 GPU {gpu_id}: Cleaning up...")
        del allocated_tensors[:]
        gc.collect()
        torch.cuda.empty_cache()

def cpu_memory_task():
    global shutdown
    while not shutdown:
        try:
            _ = [0] * (10**6)
            time.sleep(2)
        except KeyboardInterrupt:
            break

def worker(gpu_id, target_gb, chunk_gb):
    gpu_task(gpu_id, target_gb, chunk_gb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Memory Stress Test")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs (e.g., '0,1' or '0,1,2,3')")
    parser.add_argument("--target_gb", type=float, default=80, help="Target memory per GPU in GB (e.g., 80 or 20)")
    parser.add_argument("--chunk_gb", type=float, default=4, help="Allocation chunk size in GB (default: 4)")

    args = parser.parse_args()

    # 解析 GPU 列表
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    except ValueError:
        print("❌ Invalid GPU list format. Use e.g., --gpus 0,1")
        sys.exit(1)

    NUM_GPUS = len(gpu_ids)
    TARGET_GB = args.target_gb
    CHUNK_GB = args.chunk_gb

    # 验证 GPU 可用性
    available_gpus = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid >= available_gpus:
            print(f"❌ GPU {gid} is not available. System has {available_gpus} GPUs.")
            sys.exit(1)

    print(f"Intialized pressure test:")
    print(f"  - GPUs: {gpu_ids}")
    print(f"  - Target per GPU: {TARGET_GB} GB")
    print(f"  - Chunk size: {CHUNK_GB} GB")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动进程
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=worker, args=(gpu_id, TARGET_GB, CHUNK_GB))
        p.start()
        processes.append(p)

    # CPU thread (optional)
    cpu_thread = threading.Thread(target=cpu_memory_task, daemon=True)
    cpu_thread.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("⏳ Waiting for all processes to terminate...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("✅ All done.")