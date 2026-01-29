#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from protein_design_hub.core.config import get_settings
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.design.registry import DesignerRegistry
    import torch
except ImportError as e:
    print(f"CRITICAL: Failed to import core modules: {e}")
    sys.exit(1)

def check_health():
    print("=" * 60)
    print("🔬 Protein Design Hub - System Health Check")
    print("=" * 60)
    
    settings = get_settings()
    
    # 1. Environment
    print("\n[Environment]")
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    
    # GPU
    if torch.cuda.is_available():
        print(f"GPU:            ✅ Available ({torch.cuda.get_device_name(0)})")
        print(f"CUDA Version:   {torch.version.cuda}")
    else:
        print("GPU:            ⚠️  Not detected (running in CPU mode)")

    # 2. Predictors
    print("\n[Predictors]")
    predictors = PredictorRegistry.list_available()
    if not predictors:
        print("CRITICAL: No predictors registered! Check imports.")
    
    for name in predictors:
        try:
            pred = PredictorRegistry.get(name, settings)
            # Check installation
            # Note: BasePredictor usually has 'installer' or 'is_installed' logic
            # Let's check typical installer pattern
            is_installed = False
            msg = "Not installed"
            
            if hasattr(pred, 'installer') and hasattr(pred.installer, 'is_installed'):
                 # It's an installer object
                 is_installed = pred.installer.is_installed()
            elif hasattr(pred, 'is_installed'):
                 is_installed = pred.is_installed()
            else:
                 msg = "Unknown check method"
            
            status = "✅ Installed" if is_installed else f"❌ Missing ({msg})"
            print(f"{name:<15} {status}")
            
        except Exception as e:
            print(f"{name:<15} ⚠️  Error initializing: {e}")

    # 3. Designers
    print("\n[Designers]")
    designers = DesignerRegistry.list_available()
    for name in designers:
        try:
            des = DesignerRegistry.get(name, settings)
            
            is_installed = False
            if hasattr(des, 'is_installed'): # BaseDesigner usually has this
                is_installed = des.is_installed()
            else:
                 # Fallback check
                 is_installed = True # Assume python module
                 
            status = "✅ Ready" if is_installed else "❌ Missing dependencies"
            print(f"{name:<15} {status}")
        except Exception as e:
            print(f"{name:<15} ⚠️  Error initializing: {e}")

    # 4. External Tools (TMalign, etc)
    print("\n[External Tools]")
    
    # TMalign
    tmalign_path = settings.evaluation.tm_score.tmalign_path
    if tmalign_path and tmalign_path.exists():
         print(f"TMalign:        ✅ Found at {tmalign_path}")
    elif shutil.which("tm_align"):
         print(f"TMalign:        ✅ Found in PATH ({shutil.which('tm_align')})")
    elif shutil.which("tmalign"):
         print(f"TMalign:        ✅ Found in PATH ({shutil.which('tmalign')})")
    else:
         print("TMalign:        ❌ Not found (Metric: TM-score will fail)")

    # Kalign (for MSA)
    if shutil.which("kalign"):
        print(f"Kalign:         ✅ Found")
    else:
        print("Kalign:         ⚠️  Missing (Required for some MSA steps)")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_health()
