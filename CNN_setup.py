#!/usr/bin/env python
"""
Setup script for Changi AeroVision
Interactive configuration and environment setup
"""

import os
import sys


def print_banner():
    """Print welcome banner"""
    print("="*80)
    print(" " * 20 + "CHANGI AEROVISION SETUP")
    print("="*80)
    print()


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas',
        'matplotlib', 'seaborn', 'sklearn', 'PIL', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package if package != 'PIL' else 'PIL')
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print(f"   Run: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies installed")
        return True


def create_directories():
    """Create necessary output directories"""
    print("\nCreating output directories...")
    
    directories = [
        'outputs',
        'outputs/models',
        'outputs/plots'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}/")
    
    print("\n✓ Directory structure ready")


def configure_paths():
    """Interactive path configuration"""
    print("\n" + "="*80)
    print("DATASET CONFIGURATION")
    print("="*80)
    print("\nPlease specify the path to your organized dataset.\n")
    
    print("Expected structure:")
    print("  your_path\\aircraft_dataset\\")
    print("    ├── train\\")
    print("    │   ├── B737\\")
    print("    │   ├── A320\\")
    print("    │   └── ... (other classes)")
    print("    ├── val\\")
    print("    │   └── ...")
    print("    └── test\\")
    print("        └── ...\\n")
    
    # Get organized dataset path
    default_path = r'C:\Users\PC\Downloads\aircraft_dataset'
    print(f"Default path: {default_path}")
    print("\nYour dataset should already contain train/, val/, test/ subfolders")
    data_root = input("Enter dataset path (or press Enter for default): ").strip()
    
    if not data_root:
        data_root = default_path
    
    # Update CNN_config.py
    config_path = os.path.join('src', 'CNN_config.py')
    
    try:
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace DATA_ROOT path (use raw string for Windows paths)
        import re
        config_content = re.sub(
            r"DATA_ROOT = r?['\"].*?['\"]",
            f"DATA_ROOT = r'{data_root}'",
            config_content
        )
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"\n✓ Configuration updated in {config_path}")
        print(f"  DATA_ROOT = {data_root}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error updating configuration: {e}")
        return False


def verify_installation():
    """Verify the installation"""
    print("\n" + "="*80)
    print("VERIFYING INSTALLATION")
    print("="*80)
    
    checks = []
    
    # Check if src package exists
    if os.path.exists('src') and os.path.isdir('src'):
        print("  ✓ Source package (src/) exists")
        checks.append(True)
    else:
        print("  ✗ Source package (src/) not found")
        checks.append(False)
    
    # Check if main scripts exist
    scripts = ['main.py', 'train.py', 'evaluate.py', 'inference.py']
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script} exists")
            checks.append(True)
        else:
            print(f"  ✗ {script} not found")
            checks.append(False)
    
    # Check if requirements.txt exists
    if os.path.exists('requirements.txt'):
        print("  ✓ requirements.txt exists")
        checks.append(True)
    else:
        print("  ✗ requirements.txt not found")
        checks.append(False)
    
    return all(checks)


def print_next_steps():
    """Print next steps after setup"""
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("\n1. Install dependencies (if not already done):")
    print("   pip install -r requirements.txt")
    print("\n2. Run the complete pipeline:")
    print("   python main.py")
    print("\nOR run individual steps:")
    print("   python CNN_train.py")
    print("   python CNN_evaluate.py --visualize")
    print("   python CNN_inference.py path\\to\\image.jpg")
    print("\n" + "="*80)
    print("\nFor detailed documentation, see:")
    print("  - README.md (full documentation)")
    print("="*80)


def main():
    """Main setup function"""
    print_banner()
    
    # Check current directory
    if not os.path.exists('src'):
        print("⚠️  Warning: 'src' directory not found")
        print("   Please run this script from the project root directory")
        print("   (the directory containing src/, train.py, etc.)")
        sys.exit(1)
    
    # Run setup steps
    deps_ok = check_dependencies()
    
    create_directories()
    
    if deps_ok:
        configure_paths()
    else:
        print("\n⚠️  Please install dependencies first:")
        print("   pip install -r requirements.txt")
        print("\nThen run this setup script again.")
    
    verify_installation()
    
    print_next_steps()


if __name__ == '__main__':
    main()
