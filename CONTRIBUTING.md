# 🤝 Contributing to Black Hole (BLKH)

First off, **thank you** for considering contributing to BLKH! 🎉

This document describes how to contribute to the project.

---

## 🌟 Ways to Contribute

- 🐛 **Report bugs** — [Open an issue](https://github.com/Kronos1027/black-hole/issues/new?template=bug_report.md)
- 💡 **Suggest features** — [Open a feature request](https://github.com/Kronos1027/black-hole/issues/new?template=feature_request.md)
- 💬 **Share feedback** — [Open a feedback issue](https://github.com/Kronos1027/black-hole/issues/new?template=feedback.md)
- 📊 **Share benchmarks** — [Share your results](https://github.com/Kronos1027/black-hole/issues/new?template=benchmark.md)
- 🔧 **Fix bugs** — Fork, fix, and submit a Pull Request
- 📝 **Improve docs** — Better docs help everyone
- 🧪 **Add tests** — More tests = more reliable

---

## 🚀 Quick Start for Developers

### Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA GPU for GPU mode testing

### Setup

```bash
# Clone the repo
git clone https://github.com/Kronos1027/black-hole.git
cd black-hole

# Install in development mode
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Run the Web Demo Locally

```bash
cd huggingface
python app.py
# Opens at http://localhost:7860
```

---

## 📋 Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/black-hole.git`
3. **Create a branch**: `git checkout -b feature/my-new-mode`
4. **Make changes** and test: `pytest tests/ -v`
5. **Commit**: `git commit -m "feat: add new compression mode"`
6. **Push**: `git push origin feature/my-new-mode`
7. **Open Pull Request** on GitHub

---

## 📝 Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Tests only
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `chore`: Build/tooling changes

### Examples

```
feat: add v5.31 with AV1 encoding support
fix: correct YCbCr color space conversion bug
docs: update README with new benchmark results
test: add 10 tests for palette mode
perf: optimize DCT with vectorized numpy
```

---

## 🧪 Testing

All contributions must pass tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dct_pytest.py -v

# Run with coverage
pytest tests/ --cov=phase1_inr_compressor
```

### Adding New Tests

When adding a new feature, also add tests:

```python
# tests/test_new_feature_pytest.py
import pytest
from phase1_inr_compressor.siren_v5_new import NewCompressor

def test_new_feature_basic():
    # Test basic functionality
    pass

def test_new_feature_edge_case():
    # Test edge cases
    pass
```

---

## 🎛️ Adding a New Compression Mode

BLKH supports many modes. To add a new one:

1. **Create the module**: `phase1_inr_compressor/siren_v5_newmode.py`
2. **Implement the class** with `compress()` and `decompress()` methods
3. **Choose a magic bytes** (4 bytes, unique)
4. **Add CLI command** in `blkh.py`
5. **Add decompress support** in `cmd_decompress()`
6. **Add to info command** in `cmd_info()`
7. **Write tests** in `tests/test_newmode_pytest.py`
8. **Update README** with new mode
9. **Update web demo** in `huggingface/app.py`

### Template

```python
# siren_v5_newmode.py
MAGIC_NEWMODE = b'BLKX'  # Choose unique magic
VERSION_NEWMODE = 1

class NewCompressor:
    def compress(self, image, verbose=False) -> dict:
        # ... your compression logic
        return {'recipe_bytes': ..., 'recipe_size': ..., 'mode': 'newmode_v5_X'}

    @staticmethod
    def decompress(recipe_bytes) -> tuple[np.ndarray, dict]:
        # ... your decompression logic
        return recovered_image, {'sha256_match': ..., 'mode': 'newmode_v5_X'}
```

---

## 📊 Performance Guidelines

When adding or modifying compression modes:

- **Always compare with ZIP** (zlib level 9) as baseline
- **Report PSNR** for lossy modes
- **Report SHA-256 verification** for lossless modes
- **Benchmark on multiple sizes**: 128x128, 256x256, 512x512, 1024x1024
- **Test on multiple content types**: photos, smooth synthetic, palette

---

## 🐛 Bug Reports

When reporting bugs, use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment info (OS, Python version, BLKH version)
- Error output (if any)
- Sample image (if applicable)

---

## 💡 Feature Requests

When suggesting features, use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Clear description of the feature
- Problem it solves
- Proposed solution
- Alternatives considered
- Willingness to contribute

---

## 📝 Code Style

- Follow PEP 8
- Use type hints: `def compress(self, image: np.ndarray) -> dict:`
- Add docstrings to all public functions
- Keep functions focused (single responsibility)
- Use descriptive variable names

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 📞 Questions?

- **Issues**: [GitHub Issues](https://github.com/Kronos1027/black-hole/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kronos1027/black-hole/discussions)
- **Email**: darlan1027pc@gmail.com

---

**Author**: Darlan Pereira da Silva (Kronos1027)

Thank you for contributing! 🕳️✨
