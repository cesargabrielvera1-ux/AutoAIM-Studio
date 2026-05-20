"""Main entry point for Materials AutoML Studio."""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    required = [
        'numpy', 'pandas', 'sklearn', 'torch', 'PyQt6'
    ]
    
    optional = [
        'xgboost', 'lightgbm', 'catboost', 'optuna', 'shap', 'matminer'
    ]
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Error: Missing required dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install required packages:")
        print("  pip install numpy pandas scikit-learn torch PyQt6")
        return False
    
    # Check optional dependencies
    missing_optional = []
    for package in optional:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        print("Warning: Some optional dependencies are missing:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nFor full functionality, install optional packages:")
        print("  pip install xgboost lightgbm catboost optuna shap matminer")
    
    return True

def setup_environment():
    """Setup environment for optimal performance."""
    # Set number of threads for NumPy/BLAS
    import os
    cpu_count = os.cpu_count() or 4
    n_threads = max(1, cpu_count - 1)
    
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    # PyTorch settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           Materials AutoML Studio v1.1.0                      ║
    ║                                                               ║
    ║   Automated Machine Learning for Materials Science           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Import GUI after checking dependencies
    try:
        from PyQt6.QtWidgets import QApplication
        from .gui.main_window import MainWindow
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        sys.exit(1)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Materials AutoML Studio")
    app.setApplicationVersion("1.1.0")
    app.setOrganizationName("MaterialsAutoML")
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
