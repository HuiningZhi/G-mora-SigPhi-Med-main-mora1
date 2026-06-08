#!/usr/bin/env python3
"""
检查训练后模型的门控配置
判断是单门控还是双门控
打印 gate_A 和 gate_B 在不同 task 上的专家权重
"""

import torch
import os
import sys
from pathlib import Path

# 设置模型路径
MODEL_PATH = "/mnt/nvme1/Processed/output/48G/tiny-llava-Phi-4-mini-instruct-siglip2-so400m-patch14-384-base-finetune-down-moelora"

def check_gate_config(model_path):
    """检查门控配置"""
    print("\n" + "="*80)
    print("🔍 CHECK GATE CONFIGURATION")
    print("="*80)
    print(f"Model Path: {model_path}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # 1. 检查 state_dict 中的门控参数
    print("[1/3] 检查保存的权重文件...")
    
    import glob
    from safetensors import safe_open
    
    safetensor_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
    
    if not safetensor_files:
        print("❌ 没有找到 safetensors 文件")
        return False
    
    print(f"✅ 找到 {len(safetensor_files)} 个 safetensors 文件\n")
    
    # 从权重文件中检查门控参数
    state_dict_keys = []
    for shard_file in sorted(safetensor_files):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            state_dict_keys.extend(f.keys())
    
    # 分类门控参数
    gate_single_keys = [k for k in state_dict_keys if 'global_task_gate' in k and 'gate_A' not in k and 'gate_B' not in k]
    gate_a_keys = [k for k in state_dict_keys if 'global_task_gate_A' in k]
    gate_b_keys = [k for k in state_dict_keys if 'global_task_gate_B' in k]
    
    print("📊 权重文件中的门控参数:")
    print(f"  单门控 (global_task_gate):   {len(gate_single_keys)} 个参数")
    print(f"  双门控 A (global_task_gate_A): {len(gate_a_keys)} 个参数")
    print(f"  双门控 B (global_task_gate_B): {len(gate_b_keys)} 个参数")
    
    if gate_single_keys:
        print("\n  📝 单门控参数名称:")
        for key in gate_single_keys[:5]:
            print(f"     - {key}")
        if len(gate_single_keys) > 5:
            print(f"     ... 还有 {len(gate_single_keys) - 5} 个")
    
    if gate_a_keys:
        print("\n  📝 双门控 A 参数名称:")
        for key in gate_a_keys[:5]:
            print(f"     - {key}")
        if len(gate_a_keys) > 5:
            print(f"     ... 还有 {len(gate_a_keys) - 5} 个")
    
    if gate_b_keys:
        print("\n  📝 双门控 B 参数名称:")
        for key in gate_b_keys[:5]:
            print(f"     - {key}")
        if len(gate_b_keys) > 5:
            print(f"     ... 还有 {len(gate_b_keys) - 5} 个")
    
    # 2. 判断门控类型
    print("\n[2/3] 判断门控配置类型...")
    
    if len(gate_a_keys) > 0 and len(gate_b_keys) > 0:
        gate_type = "DUAL_INDEPENDENT"
        print("✅ 检测到：双独立门控配置 (Gate A & Gate B)")
    elif len(gate_single_keys) > 0:
        gate_type = "SINGLE_SHARED"
        print("✅ 检测到：单全局共享门控配置 (Global Gate)")
    else:
        print("❌ 无法确定门控类型")
        return False
    
    # 3. 加载模型并打印专家权重
    print("\n[3/3] 加载模型并打印专家权重...")
    
    try:
        from tinyllava.model.load_model import load_model
        
        model, tokenizer, image_processor, context_len = load_model(
            model_path,
            device_map="auto",
            is_distributed=False,
            load_moelora=True
        )
        
        print("✅ 模型加载成功\n")
        
        # 获取 LLM 模型
        if hasattr(model, 'language_model'):
            llm_model = model.language_model
        elif hasattr(model, 'llm'):
            llm_model = model.llm
        else:
            print("❌ 无法找到 language_model 或 llm")
            return False
        
        # 打印专家权重
        print("="*80)
        print("🎯 EXPERT GATE WEIGHTS")
        print("="*80)
        
        task_mapping = {
            0: 'TASK_0',
            1: 'TASK_1'
        }
        
        if gate_type == "DUAL_INDEPENDENT":
            print("\n🔀 双独立门控配置\n")
            
            if not (hasattr(llm_model, 'global_task_gate_A') and hasattr(llm_model, 'global_task_gate_B')):
                print("❌ 模型中未找到 global_task_gate_A 和 global_task_gate_B")
                return False
            
            gate_a = llm_model.global_task_gate_A
            gate_b = llm_model.global_task_gate_B
            
            # 获取设备
            gate_a_device = next(gate_a.parameters()).device
            gate_b_device = next(gate_b.parameters()).device
            
            print(f"Gate A 设备: {gate_a_device}")
            print(f"Gate B 设备: {gate_b_device}\n")
            
            with torch.no_grad():
                for task_id, task_name in task_mapping.items():
                    print(f"\n{'='*80}")
                    print(f"📊 {task_name} (ID={task_id}):")
                    print('='*80)
                    
                    # Task tensor
                    task_tensor_a = torch.tensor([task_id], dtype=torch.long, device=gate_a_device)
                    task_tensor_b = torch.tensor([task_id], dtype=torch.long, device=gate_b_device)
                    
                    # Gate A 权重
                    expert_weights_a = gate_a(task_tensor_a)
                    weights_a = expert_weights_a[0].cpu().numpy()
                    
                    print(f"\n  🚪 Gate A (gate_up_proj) 专家权重:")
                    print(f"     {weights_a}")
                    max_expert_a = weights_a.argmax()
                    print(f"     最强专家: Expert {max_expert_a} ({weights_a[max_expert_a]:.4f})")
                    
                    # 可视化
                    for i, weight in enumerate(weights_a):
                        bar = "█" * int(weight * 50)
                        marker = " ⭐" if i == max_expert_a else ""
                        print(f"       Expert {i}: {weight:.4f} |{bar:<50}|{marker}")
                    
                    # Gate B 权重
                    expert_weights_b = gate_b(task_tensor_b)
                    weights_b = expert_weights_b[0].cpu().numpy()
                    
                    print(f"\n  🚪 Gate B (down_proj) 专家权重:")
                    print(f"     {weights_b}")
                    max_expert_b = weights_b.argmax()
                    print(f"     最强专家: Expert {max_expert_b} ({weights_b[max_expert_b]:.4f})")
                    
                    # 可视化
                    for i, weight in enumerate(weights_b):
                        bar = "█" * int(weight * 50)
                        marker = " ⭐" if i == max_expert_b else ""
                        print(f"       Expert {i}: {weight:.4f} |{bar:<50}|{marker}")
                    
                    # 对比
                    print(f"\n  📈 Gate A vs Gate B 对比:")
                    print(f"     相同: {torch.allclose(expert_weights_a, expert_weights_b)}")
                    if not torch.allclose(expert_weights_a, expert_weights_b):
                        diff = (weights_a - weights_b).max()
                        print(f"     最大差异: {diff:.6f}")
            
        elif gate_type == "SINGLE_SHARED":
            print("\n🔗 单全局共享门控配置\n")
            
            if not hasattr(llm_model, 'global_task_gate'):
                print("❌ 模型中未找到 global_task_gate")
                return False
            
            gate = llm_model.global_task_gate
            gate_device = next(gate.parameters()).device
            
            print(f"Gate 设备: {gate_device}\n")
            
            with torch.no_grad():
                for task_id, task_name in task_mapping.items():
                    print(f"\n{'='*80}")
                    print(f"📊 {task_name} (ID={task_id}):")
                    print('='*80)
                    
                    task_tensor = torch.tensor([task_id], dtype=torch.long, device=gate_device)
                    expert_weights = gate(task_tensor)
                    weights = expert_weights[0].cpu().numpy()
                    
                    print(f"\n  🚪 Global Gate (共享) 专家权重:")
                    print(f"     {weights}")
                    max_expert = weights.argmax()
                    print(f"     最强专家: Expert {max_expert} ({weights[max_expert]:.4f})")
                    
                    # 可视化
                    for i, weight in enumerate(weights):
                        bar = "█" * int(weight * 50)
                        marker = " ⭐" if i == max_expert else ""
                        print(f"       Expert {i}: {weight:.4f} |{bar:<50}|{marker}")
        
        print("\n" + "="*80)
        print("✅ 检查完成!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载模型或处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_gate_config(MODEL_PATH)
    sys.exit(0 if success else 1)
