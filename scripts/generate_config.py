import yaml
import torch
import platform
import os

def get_arch_from_capability(major, minor):
    """
    根據 Compute Capability (sm_xx) 映射到架構名稱
    參照: NVIDIA CUDA Architecture mapping
    """
    # Blackwell (sm_100, sm_101, sm_120...)
    # 根據您的截圖與新一代定義，Blackwell 為 10.x
    if major >= 10:
        return "blackwell"
    
    # Hopper (sm_90)
    if major == 9:
        return "hopper"
    
    if major == 8:
        # Ada Lovelace (sm_89) - RTX 40 Series
        if minor == 9:
            return "ada"
        # Ampere (sm_80, sm_86, sm_87) - RTX 30 Series, A100
        return "ampere"
    
    if major == 7:
        # Turing (sm_75) - RTX 20 Series
        if minor == 5:
            return "turing"
        # Volta (sm_70, sm_72) - V100, Xavier
        return "volta"
    
    if major == 6:
        # Pascal (sm_60, sm_61) - GTX 10 Series
        return "pascal"
        
    return f"unknown_sm{major}{minor}"

def get_platform_info():
    info = {
        "platform": f"windows_{platform.machine().lower()}",
        "architecture": platform.machine(),
        "cuda_version": "cpu",
        "memory_gb": 0,
        "gpu_name": "cpu",
        "compute_capability": "N/A"
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        
        # 獲取第一張 GPU 的資訊
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        
        # 1. 精準記憶體計算 (轉換為 GB 並保留一位小數)
        info["memory_gb"] = round(props.total_memory / (1024**3), 1)
        info["gpu_name"] = props.name
        
        # 2. 透過架構編號判斷 (The correct way)
        major, minor = torch.cuda.get_device_capability(device_id)
        info["compute_capability"] = f"{major}.{minor}"
        
        arch_name = get_arch_from_capability(major, minor)
        info["platform"] += f"_{arch_name}"
        
        print(f"[DEBUG] Detected GPU: {props.name}")
        print(f"[DEBUG] Compute Capability: {major}.{minor} -> Architecture: {arch_name}")

    # 模型配置保持預設或設為 auto
    info["model_config"] = {
        "hidden_dim": 512,
        "num_layers": 6,
        "attention_backend": "auto",
        "batch_size": 32
    }
    
    info["vector_index"] = {
        "backend": "auto"
    }

    return info

def main():
    config_dir = "configs"
    config_path = os.path.join(config_dir, "platform.yaml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # 強制重新生成 (如果想保留舊設定可改回 if not os.path.exists...)
    # 建議: 這裡設為覆蓋，因為硬體變更時通常需要更新
    print(f"[INFO] Detecting hardware and generating {config_path}...")
    try:
        data = get_platform_info()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"[OK] Configuration generated: {data['platform']} (SM {data.get('compute_capability')})")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate config: {e}")
        # Fallback for minimal config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump({"platform": "error_fallback"}, f)

if __name__ == "__main__":
    main()