"""
convert_model.py — Convert Keras model to TensorFlow.js LayersModel format
Implements the TF.js binary + JSON format directly — no tensorflowjs package needed.

Usage:
    py convert_model.py

Requirements:
    pip install tensorflow-cpu numpy
"""

import os, sys, json, math
import numpy as np

MODEL_IN  = "model/inceptionresnetv2_corrected (1).keras"
MODEL_OUT = "model"
SHARD_MB  = 20           # max shard size in MB
QUANTIZE  = True         # uint8 quantization (reduces size ~4x)

# ──────────────────────────────────────────────────────────────────────────────

def quantize_uint8(arr):
    """Linearly quantize float32 → uint8. Returns (bytes, min_val, max_val)."""
    mn = float(arr.min())
    mx = float(arr.max())
    if mx == mn:
        mx = mn + 1e-7
    scaled = np.round((arr.astype(np.float64) - mn) / (mx - mn) * 255.0)
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled.tobytes(), mn, mx


def convert():
    print("=" * 60)
    print("  OcularAI - Keras to TF.js Converter")
    print("=" * 60)

    # ── Locate model ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_IN)
    out_dir    = os.path.join(script_dir, MODEL_OUT)

    if not os.path.exists(model_path):
        # Also try root of project
        alt = os.path.join(script_dir, "inceptionresnetv2_corrected (1).keras")
        if os.path.exists(alt):
            model_path = alt
        else:
            print(f"\nERROR: Model file not found at:\n  {model_path}")
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # ── Import TF ─────────────────────────────────────────────────
    import os as _os
    _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

    print("\n[1/5] Importing TensorFlow…")
    import tensorflow as tf
    print(f"      TensorFlow {tf.__version__}")

    # ── Load model ────────────────────────────────────────────────
    print(f"\n[2/5] Loading model from:\n      {model_path}")
    print("      (30-90 s for a 629 MB model...)")

    # Rebuild architecture from scratch (avoids Python bytecode mismatch on Lambda layers)
    # Architecture from model config: Input(224,224,3) -> InceptionResNetV2(pooling=max)
    #   -> BatchNorm(axis=1) -> Dense(256,relu) -> Dropout(0.45) -> Dense(4,softmax)
    print("      Rebuilding architecture (bypasses Lambda bytecode version issue)...")
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling="max",
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(axis=[1]),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    # Build model so weights can be loaded
    model.build((None, 224, 224, 3))
    print("      Loading weights from HDF5...")
    model.load_weights(model_path)
    print(f"      Input  shape: {model.input_shape}")
    print(f"      Output shape: {model.output_shape}")
    print(f"      Parameters  : {model.count_params():,}")

    # ── Extract topology ──────────────────────────────────────────
    print("\n[3/5] Extracting model topology…")
    topology = json.loads(model.to_json())

    # ── Collect weights ───────────────────────────────────────────
    print("\n[4/5] Collecting & quantizing weights…")
    weight_entries = []   # list of (name, shape, dtype_str, raw_bytes, quant_info)
    total_params = 0

    for layer in model.layers:
        for weight in layer.weights:
            arr  = weight.numpy().flatten()
            name = weight.name
            shp  = list(weight.shape)
            total_params += arr.size

            if QUANTIZE:
                raw_bytes, mn, mx = quantize_uint8(arr)
                entry = {
                    "name":  name,
                    "shape": shp,
                    "dtype": "float32",
                    "quantization": {"min": mn, "max": mx, "dtype": "uint8"},
                    "_bytes": raw_bytes,
                }
            else:
                raw_bytes = arr.astype(np.float32).tobytes()
                entry = {
                    "name":  name,
                    "shape": shp,
                    "dtype": "float32",
                    "_bytes": raw_bytes,
                }
            weight_entries.append(entry)

    raw_size  = sum(len(e["_bytes"]) for e in weight_entries)
    print(f"      Weights collected: {len(weight_entries)}")
    print(f"      Raw binary size  : {raw_size/1024/1024:.1f} MB (quantized)")

    # ── Write weight shards ───────────────────────────────────────
    print("\n[5/5] Writing weight shards and model.json…")
    shard_size = SHARD_MB * 1024 * 1024  # bytes

    # Pack all bytes into shards
    all_bytes = b"".join(e["_bytes"] for e in weight_entries)
    n_shards  = math.ceil(len(all_bytes) / shard_size)
    shard_names = []

    for i in range(n_shards):
        chunk = all_bytes[i*shard_size : (i+1)*shard_size]
        fname = f"group1-shard{i+1}of{n_shards}.bin"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(chunk)
        shard_names.append(fname)
        print(f"      {fname} — {len(chunk)/1024/1024:.1f} MB")

    # Build weights manifest
    # Each weight entry needs byte offset & size tracked in the shard
    manifest_weights = []
    for e in weight_entries:
        w = {k: v for k, v in e.items() if k != "_bytes"}
        manifest_weights.append(w)

    weights_manifest = [{
        "paths":   shard_names,
        "weights": manifest_weights,
    }]

    # ── Write model.json ──────────────────────────────────────────
    model_json = {
        "modelTopology": topology,
        "format":        "layers-model",
        "generatedBy":   f"keras v{tf.__version__}",
        "convertedBy":   "OcularAI converter (uint8 quantized)",
        "weightsManifest": weights_manifest,
    }

    json_path = os.path.join(out_dir, "model.json")
    with open(json_path, "w") as f:
        json.dump(model_json, f, separators=(",", ":"))

    json_size = os.path.getsize(json_path)
    print(f"\n      model.json — {json_size/1024:.1f} KB")

    # ── Summary ───────────────────────────────────────────────────
    total_size = sum(os.path.getsize(os.path.join(out_dir, f))
                     for f in os.listdir(out_dir))
    print(f"\n{'='*60}")
    print(f"  Conversion complete!")
    print(f"  Output dir : {out_dir}")
    print(f"  Total size : {total_size/1024/1024:.1f} MB ({n_shards} shards)")
    print(f"{'='*60}")

    over = [f for f in os.listdir(out_dir)
            if os.path.getsize(os.path.join(out_dir, f)) > 99*1024*1024]
    if over:
        print(f"\nWARNING: {len(over)} file(s) exceed 99 MB (GitHub limit).")
        print("  Reduce SHARD_MB at the top of this script and re-run.")
    else:
        print("\n  All files < 99 MB — safe for GitHub Pages.")

    print("""
Next steps:
  1.  git init && git add . && git commit -m "Add OcularAI"
  2.  Create a public GitHub repo and push
  3.  Settings → Pages → Branch: main → / (root) → Save
  4.  Site live at: https://<user>.github.io/<repo>/
  5.  NOTE: Delete 'model/inceptionresnetv2_corrected (1).keras'
           before pushing — it is 629 MB and not needed by the site.
""")


if __name__ == "__main__":
    convert()
