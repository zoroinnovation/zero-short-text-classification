# Installation Guide for Team Members

## Simple Setup (Recommended):
```bash
pip install transformers datasets torch jupyter openai
```

## Test Your Installation:
Create `test_install.py`:
```python
from transformers import pipeline
print("✅ Installation successful!")
```

## Common Problems & Solutions:

### 1. Python Not Found
**Problem:** `python --version` gives error
**Solution:** 
- Download Python from python.org
- Make sure to check "Add Python to PATH" during installation

### 2. Pip Not Working
**Problem:** `pip install` gives error
**Solution:**
```bash
python -m pip install --upgrade pip
```

### 3. Permission Errors
**Problem:** "Access denied" when installing packages
**Solution:**
- Run Command Prompt as Administrator
- Or use: `pip install --user transformers datasets torch jupyter openai`
