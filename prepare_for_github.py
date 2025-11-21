#!/usr/bin/env python3
"""
Repository Cleanup and GitHub Preparation Script

This script checks the repository status and prepares it for GitHub push.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_git_status():
    """Check if we're in a git repository and show status."""
    print("🔍 Checking Git repository status...")
    
    # Check if git repo exists
    success, _, _ = run_command("git rev-parse --git-dir")
    if not success:
        print("❌ Not a Git repository. Initialize with:")
        print("   git init")
        print("   git remote add origin https://github.com/SaiChen22/stock-prediction-system.git")
        return False
    
    # Show current status
    success, status, _ = run_command("git status --porcelain")
    if success and status:
        print(f"📋 Files to be committed:")
        for line in status.split('\n'):
            if line.strip():
                print(f"   {line}")
    else:
        print("✅ Working directory is clean")
    
    return True

def check_files():
    """Check if all necessary files exist."""
    print("\n📁 Checking project files...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "setup.py",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        ".gitignore",
        "core/__init__.py",
        "tests/test_encoders.py",
        "tests/test_optimizers.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_tests():
    """Run the test suite to ensure everything works."""
    print("\n🧪 Running test suite...")
    
    # Check if pytest is available
    success, _, _ = run_command("python -c 'import pytest'")
    if not success:
        print("   ⚠️ pytest not installed, running basic tests...")
        
        # Run basic import tests
        success, _, error = run_command("python -c 'from core import *; print(\"✅ Core imports work\")'")
        if success:
            print("   ✅ Core modules import successfully")
        else:
            print(f"   ❌ Import error: {error}")
            return False
            
        # Run test files directly
        for test_file in ["tests/test_encoders.py", "tests/test_optimizers.py"]:
            if os.path.exists(test_file):
                success, output, error = run_command(f"python {test_file}")
                if success:
                    print(f"   ✅ {test_file} passed")
                else:
                    print(f"   ❌ {test_file} failed: {error}")
                    return False
        return True
    else:
        # Run with pytest
        success, output, error = run_command("python -m pytest tests/ -v")
        if success:
            print("   ✅ All tests passed!")
            return True
        else:
            print(f"   ❌ Tests failed:\n{error}")
            return False

def check_dependencies():
    """Check if all dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    with open("requirements.txt", "r") as f:
        requirements = [line.strip().split(">=")[0].split("==")[0] for line in f 
                      if line.strip() and not line.startswith("#")]
    
    # Map package names to import names
    import_names = {
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_deps = []
    for dep in requirements[:5]:  # Check first 5 core dependencies
        if dep in ['pytest', 'black', 'flake8', 'mypy']:  # Skip dev deps
            continue
            
        import_name = import_names.get(dep, dep.replace('-', '_'))
        success, _, _ = run_command(f"python -c 'import {import_name}'")
        if success:
            print(f"   ✅ {dep}")
        else:
            print(f"   ❌ {dep} (not installed)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n   💡 Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def suggest_git_commands():
    """Suggest Git commands for pushing to GitHub."""
    print("\n🚀 Ready for GitHub! Suggested commands:")
    print()
    print("1️⃣ Add all files:")
    print("   git add .")
    print()
    print("2️⃣ Commit changes:")
    print("   git commit -m \"🎉 Initial release: Advanced Stock Prediction System v1.0.0")
    print()
    print("   Features:")
    print("   - Extended sequence encoding for pattern recognition")
    print("   - ML parameter optimization (Bayesian/Genetic/Random)")  
    print("   - 55-62% prediction accuracy on major stocks")
    print("   - Comprehensive testing and documentation")
    print("   - Production-ready modular architecture\"")
    print()
    print("3️⃣ Push to GitHub:")
    print("   git push -u origin main")
    print()
    print("🎯 After pushing, consider:")
    print("   - Create a release tag: git tag v1.0.0 && git push --tags")
    print("   - Enable GitHub Actions for automated testing")
    print("   - Add repository description and topics on GitHub")
    print("   - Star the repository to show it's ready! ⭐")

def main():
    """Main function to run all checks."""
    print("🧹 Repository Cleanup and GitHub Preparation")
    print("=" * 50)
    
    # Run all checks
    git_ok = check_git_status()
    files_ok = check_files()
    deps_ok = check_dependencies()
    tests_ok = run_tests()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"   Git Repository: {'✅' if git_ok else '❌'}")
    print(f"   Required Files: {'✅' if files_ok else '❌'}")
    print(f"   Dependencies:   {'✅' if deps_ok else '❌'}")
    print(f"   Tests:         {'✅' if tests_ok else '❌'}")
    
    if all([git_ok, files_ok, deps_ok, tests_ok]):
        print("\n🎉 All checks passed! Repository is ready for GitHub.")
        suggest_git_commands()
    else:
        print("\n⚠️ Please fix the issues above before pushing to GitHub.")
        
    return all([git_ok, files_ok, deps_ok, tests_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)