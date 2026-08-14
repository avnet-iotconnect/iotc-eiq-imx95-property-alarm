#!/bin/bash
#
# Build the NXP eIQ GenAI Flow library tarball, from nothing, in one command.
#
#   scripts/dm-eiq-package.sh
#
# Clones NXP's public repo, pulls the LFS objects, keeps the ~140 MB our demo actually loads,
# writes dm-eiq-genai-flow-lib.tgz into the current directory, and cleans up after itself. Takes no
# arguments and needs no prior checkout -- run it anywhere.
#
# Rename the tarball for hosting however you like; VERSION.txt inside records which upstream commit
# it came from, so the provenance travels with it.
#
# The approach to pruning is deliberately coarse: copy whole Python packages, then delete the model
# weights we never load. Only weights are big enough to be worth excluding -- an unused __main__.py
# or a spare audio backend is a few kilobytes, and being surgical about those costs readability for
# nothing.
#
# Dropped wholesale: llm/ (848 MB), rag/ (95 MB), gui/, tests/, assets/.
# Pruned after copying: whisper-small weights (288 MB), the fp32 TTS trio (152 MB), moonshine-tiny.
#
# The result unpacks onto the gitignored library directory inside the demo, on the development PC
# and the board alike:
#   tar xzf dm-eiq-genai-flow-lib.tgz -C src/nxp-lib/
#
# NXP's licence forbids editing these files, so everything is copied verbatim and LICENSE.txt rides
# along as clause 3.4 requires. Our own code lives in src/, never in the payload.

set -euo pipefail

# Absolute, because the script changes directory several times before writing the tarball.
output=$PWD/dm-eiq-genai-flow-lib.tgz
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

# --- fetch upstream ---------------------------------------------------------------------------

if ! git lfs version > /dev/null 2>&1; then
    echo "ERROR: git-lfs is not installed, and the models are all LFS objects." >&2
    echo "       Install it first:  apt install git-lfs   (or dnf/brew install git-lfs)" >&2
    exit 1
fi
git lfs install

echo "=== cloning NXP's repo at release/v3.0 ==="
rm -rf dm-eiq-genai-flow-demonstrator
# SKIP_SMUDGE leaves the LFS files as pointer stubs during clone. Without it git fetches all 1.5 GB
# here and the selective pull below has nothing left to skip.
GIT_LFS_SKIP_SMUDGE=1 git clone --branch release/v3.0 --depth 1 \
    https://github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator

cd dm-eiq-genai-flow-demonstrator

# llm/ and rag/ are 943 MB of LFS objects we throw away anyway, so never download them.
echo "=== pulling LFS objects (~550 MB) ==="
git lfs pull --exclude="eiq_genai_flow/llm,eiq_genai_flow/rag"

# LFS pointer stubs are ~131 bytes; a real model is megabytes. Catch a failed pull here rather than
# shipping a tarball full of text files.
if [ "$(stat -c %s eiq_genai_flow/vit/src/vit/vit_ops.cpython-313-aarch64-linux-gnu.so)" -lt 100000 ]; then
    echo "ERROR: LFS objects did not download -- binaries are still pointer stubs." >&2
    exit 1
fi

# --- collect ----------------------------------------------------------------------------------

# stage <package-name> <source-directory> -- copies a whole Python package as-is.
stage() {
    mkdir -p "$staging/src/$1"
    cp -a "$2/." "$staging/src/$1/"
}

echo "=== collecting ==="

# Licence text, verbatim -- required on any copy by clause 3.4.
mkdir -p "$staging/src"
cp -a LICENSE.txt "$staging/src/"

# The six packages our flow imports. NXP's pyproject.toml declares each of these as its own package
# root; flattening away the intervening */src/ is what collapses their nine roots to our one.
stage shared_utils    eiq_genai_flow/shared_utils
stage vit             eiq_genai_flow/vit/src/vit
stage tts             eiq_genai_flow/tts/src/tts
stage speech_to_text  eiq_genai_flow/speech_to_text/src/speech_to_text
stage audio_manager   eiq_genai_flow/audio_manager/src/audio_manager
stage vlm             vlm/src/vlm

# Reference material, so each capability can be validated without working hardware: two recordings
# for wake word and STT, and a doorbell-camera image for the VLM that matches our anti-theft framing.
mkdir -p "$staging/testdata"
cp -a eiq_genai_flow/vit/tests/HeyNXP_en.wav "$staging/testdata/"
cp -a eiq_genai_flow/speech_to_text/tests/data/sample_en.wav "$staging/testdata/"
cp -a vlm/test/data/delivery.jpg "$staging/testdata/"

# Earcons -- the short blips that tell a user the board heard them. 133 KB for all three, and far
# better feedback than a line of console text nobody at a booth is looking at.
mkdir -p "$staging/earcons"
cp -a eiq_genai_flow/assets/ww_earcon.wav "$staging/earcons/"
cp -a eiq_genai_flow/assets/intent_earcon.wav "$staging/earcons/"
cp -a eiq_genai_flow/assets/tts_earcon.wav "$staging/earcons/"

{
    echo "eIQ GenAI Flow, pruned to the pieces the Avnet anti-theft demo loads."
    echo "upstream: github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator"
    echo "branch:   release/v3.0"
    echo "commit:   $(git rev-parse HEAD)"
    echo "built:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$staging/VERSION.txt"

cd ..
rm -rf dm-eiq-genai-flow-demonstrator

# --- prune ------------------------------------------------------------------------------------

# This is where the 3.2 GB actually goes.
cd "$staging/src"

# TTS ships two sets of weights. MultiSpeakerTTS16kHzQuantConfig selects the *-quant-encrypted-*
# trio; the fp32 trio beside it is 152 MB that is never opened. The glob matches only fp32 because
# the quantised names carry an extra '-quant' before '-encrypted'.
rm -f tts/saved_models/english/*-16k-encrypted-*

# STT ships three models and we select moonshine-base. whisper-small is 288 MB of weights plus its
# tokenizer; moonshine-tiny is the lighter fallback -- stop deleting it here if base proves too slow.
#
# Note what is NOT deleted: whisper/feature.so. Tested 2026-07-27 -- speech_to_text.py:33 imports
# `speech_to_text.models.whisper.feature` unconditionally at module scope, so removing it breaks
# moonshine too. 200 KB, and mandatory.
rm -rf speech_to_text/models/whisper/onnx_models
rm -rf speech_to_text/models/whisper/tokenizers
rm -f  speech_to_text/models/moonshine/onnx_models/moonshine-tiny_*
rm -rf speech_to_text/models/moonshine/tokenizers/moonshine-tiny

# Build residue from the clone, never ours.
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

# --- pack -------------------------------------------------------------------------------------

cd "$staging"
find src testdata earcons -type f -exec sha256sum {} \; | sort -k2 > MANIFEST.txt
tar czf "$output" src testdata earcons VERSION.txt MANIFEST.txt

echo
echo "Wrote $output ($(du -h "$output" | cut -f1))"
echo "  payload: $(du -sh src | cut -f1) across $(find src -type f | wc -l) files"
echo "  unpack with: tar xzf $(basename "$output") -C src/nxp-lib/"
