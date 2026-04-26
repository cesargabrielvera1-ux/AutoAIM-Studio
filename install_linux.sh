#!/bin/bash
# Installation script for Materials AutoML Studio
# Works on Linux and macOS

echo "=================================="
echo "Materials AutoML Studio Installer"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

echo "Python version: $python_version ✓"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old environment..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created ✓"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated ✓"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Warning: Failed to upgrade pip, continuing with current version"
fi
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take several minutes..."
echo ""

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Dependencies installed ✓"
echo ""

# Check for GPU
echo "Checking for GPU support..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU support detected ✓"
else
    echo "No GPU detected (CPU-only mode)"
fi
echo ""

# Run tests
echo "Running tests..."
python3 -m pytest tests/ -v --tb=short 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Tests passed ✓"
else
    echo "Warning: Some tests failed"
fi
echo ""

# Create launcher script
echo "Creating launcher script..."
cat > launch.sh << 'EOF'
#!/bin/bash
# Launcher for Materials AutoML Studio

cd "$(dirname "$0")"
source venv/bin/activate
python run.py
EOF

chmod +x launch.sh
echo "Launcher created ✓"
echo ""

# Final message
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "To start Materials AutoML Studio:"
echo "  ./launch.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "For help, see USAGE.md"
echo ""
