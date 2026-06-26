#!/bin/bash
# ============================================================
# Build .whl and upload to GitHub Release v1.0-preview
# Run on Kaggle T4x2 after cloning the repo.
#
# Prerequisites:
#   - GitHub CLI installed and authenticated:  gh auth login
#   - CUDA 13.0 + PyTorch 2.10.0 (cu130)
#
# Usage:
#   bash scripts/build_and_upload_release.sh
# ============================================================
set -e

RELEASE_TAG="v1.0-preview"
REPO="THE-ANGEL-AI/SageAttention-SM75-path"

echo "========================================="
echo "Building .whl for $RELEASE_TAG"
echo "========================================="

# 1. Clean any previous builds
echo ""
echo "[1/4] Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info 2>/dev/null || true

# 2. Build wheel
echo ""
echo "[2/4] Building wheel..."
python setup.py bdist_wheel

# Find the generated .whl file
WHL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -z "$WHL_FILE" ]; then
    echo "ERROR: No .whl file found in dist/"
    ls -la dist/ 2>/dev/null
    exit 1
fi
echo "  Built: $WHL_FILE"

# 3. Also create a source distribution
echo ""
echo "[3/4] Building source distribution..."
python setup.py sdist
SDIST_FILE=$(ls dist/*.tar.gz 2>/dev/null | head -1)
echo "  Built: $SDIST_FILE"

# 4. Upload to GitHub Release
echo ""
echo "[4/4] Uploading to GitHub Release $RELEASE_TAG..."
echo "  Repo: $REPO"

# Upload wheel
gh release upload "$RELEASE_TAG" "$WHL_FILE" \
    --repo "$REPO" \
    --clobber
echo "  ✓ Wheel uploaded"

# Upload source dist
if [ -n "$SDIST_FILE" ]; then
    gh release upload "$RELEASE_TAG" "$SDIST_FILE" \
        --repo "$REPO" \
        --clobber
    echo "  ✓ Source dist uploaded"
fi

echo ""
echo "========================================="
echo "DONE — Release assets uploaded!"
echo "========================================="
echo ""
echo "Release URL: https://github.com/$REPO/releases/tag/$RELEASE_TAG"
echo ""
echo "Users can now install via:"
echo "  pip install https://github.com/$REPO/releases/download/$RELEASE_TAG/$(basename $WHL_FILE)"
