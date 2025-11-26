# Notebooks for Flood Estimation

This project uses [Jupytext](https://jupytext.readthedocs.io/) to maintain notebooks as `.py` files in version control instead of `.ipynb` files. This makes diffs cleaner and merge conflicts easier to resolve.

## Working with Notebooks

### Folder Structure

- **notebooks/**: Contains source `.py` files (tracked in git)
- **juptextsync/**: Contains generated `.ipynb` files (excluded from git)

All `.ipynb` files are automatically generated from `.py` files and stored in the `juptextsync/` folder.

### Converting .py to .ipynb

To convert a `.py` notebook to `.ipynb` for use in Jupyter/Colab:

```bash
cd notebooks
jupytext --to notebook --output ../juptextsync/colab_train.ipynb colab_train.py
```

This creates `juptextsync/colab_train.ipynb` which you can open in Jupyter or upload to Colab.

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
cd notebooks
jupytext --to notebook --output ../juptextsync/colab_train.ipynb colab_train.py
```

2. Upload `juptextsync/colab_train.ipynb` to Colab

3. When done, download and convert back:
```bash
jupytext --to py:percent --output notebooks/colab_train.py downloaded_notebook.ipynb
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
cd notebooks
jupytext --to notebook --output ../juptextsync/colab_train.ipynb colab_train.py

# Upload juptextsync/colab_train.ipynb to Colab
```

## Benefits of Jupytext

1. **Better Version Control**: Clean diffs show actual code changes
2. **No Merge Conflicts**: Easier to merge changes from multiple contributors
3. **Code Review**: Reviewers can read notebooks as Python files
4. **Consistent Formatting**: Automatic code formatting with black/autopep8
5. **Smaller Files**: .py files are smaller than .ipynb

## Configuration

The project `.gitignore` excludes:
- `*.ipynb` files (all notebook files)
- `juptextsync/` folder (generated notebooks)

Only `.py` versions in `notebooks/` are tracked in git.

**Workflow:**
1. Clone repository (contains `.py` files in `notebooks/`)
2. Generate `.ipynb` when needed:
   ```bash
   cd notebooks
   jupytext --to notebook --output ../juptextsync/colab_train.ipynb colab_train.py
   ```
3. Edit `juptextsync/*.ipynb` in Jupyter/Colab
4. Convert back to `.py` before committing:
   ```bash
   jupytext --to py:percent --output notebooks/colab_train.py ../juptextsync/colab_train.ipynb
   ```
5. Commit only the `.py` files in `notebooks/`

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
