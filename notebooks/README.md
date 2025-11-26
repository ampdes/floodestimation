# Notebooks for Flood Estimation

This project uses [Jupytext](https://jupytext.readthedocs.io/) to maintain notebooks as `.py` files in version control instead of `.ipynb` files. This makes diffs cleaner and merge conflicts easier to resolve.

## Working with Notebooks

### Converting .py to .ipynb

To convert a `.py` notebook to `.ipynb` for use in Jupyter/Colab:

```bash
jupytext --to notebook colab_train.py
```

This creates `colab_train.ipynb` which you can open in Jupyter or upload to Colab.

### Pairing .py and .ipynb

To keep both formats in sync automatically:

```bash
jupytext --set-formats py:percent,ipynb colab_train.py
```

Now when you save the notebook in Jupyter, it will automatically update both the `.py` and `.ipynb` files.

### Creating New Notebooks

1. **Create as .py file** (recommended):
```bash
jupytext --to py:percent notebook_name.ipynb
```

2. **Or create directly as .py**:
Just create a `.py` file with jupytext cell markers:
```python
# %% [markdown]
# # My Notebook Title

# %%
print("Hello World")
```

### Using in Google Colab

1. Convert to `.ipynb`:
```bash
jupytext --to notebook colab_train.py
```

2. Upload the generated `.ipynb` file to Colab

3. When done, download and convert back:
```bash
jupytext --to py:percent downloaded_notebook.ipynb
```

## Available Notebooks

### colab_train.py
Complete training pipeline for Google Colab with GPU support.

**Features:**
- Automatic GPU detection
- Google Drive integration for data and results
- Full training loop with visualization
- Model checkpointing and result persistence

**To use:**
```bash
# Convert to notebook
jupytext --to notebook colab_train.py

# Upload colab_train.ipynb to Colab
```

## Benefits of Jupytext

1. **Better Version Control**: Clean diffs show actual code changes
2. **No Merge Conflicts**: Easier to merge changes from multiple contributors
3. **Code Review**: Reviewers can read notebooks as Python files
4. **Consistent Formatting**: Automatic code formatting with black/autopep8
5. **Smaller Files**: .py files are smaller than .ipynb

## Configuration

The project `.gitignore` excludes `*.ipynb` files, so only `.py` versions are tracked in git.

To work with notebooks:
1. Clone repository (contains `.py` files)
2. Convert to `.ipynb` when needed: `jupytext --to notebook *.py`
3. Edit in Jupyter/Colab
4. Convert back to `.py` before committing: `jupytext --to py:percent *.ipynb`
5. Commit only the `.py` files

## Automatic Conversion (Optional)

Add a git pre-commit hook to automatically convert notebooks:

```bash
# .git/hooks/pre-commit
#!/bin/bash
jupytext --to py:percent --pre-commit
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## References

- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [Jupytext FAQ](https://jupytext.readthedocs.io/en/latest/faq.html)
- [Using Jupytext with Version Control](https://jupytext.readthedocs.io/en/latest/using-cli.html)
