/**
 * OcularAI — Eye Disease Classifier
 * TensorFlow.js inference engine + drag-and-drop UI
 */

'use strict';

// ─── Constants ────────────────────────────────────────────────────────────────

// Teachable Machine MobileNet model (2.1 MB, native TF.js format)
const MODEL_PATH = './model_tm/model.json';

const CLASSES = [
  { key: 'cataract',              label: 'Cataract',               short: 'Cataract' },
  { key: 'diabetic_retinopathy',  label: 'Diabetic Retinopathy',   short: 'Diabetic Ret.' },
  { key: 'glaucoma',              label: 'Glaucoma',               short: 'Glaucoma' },
  { key: 'normal',                label: 'Normal',                 short: 'Normal' },
];

const IMG_SIZE = 224; // Model trained with 224x224 input

// ─── State ────────────────────────────────────────────────────────────────────

let model = null;
let modelLoaded = false;
let currentFile = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const dropzone        = document.getElementById('dropzone');
const fileInput       = document.getElementById('file-input');
const browseBtn       = document.getElementById('browse-btn');
const clearBtn        = document.getElementById('clear-btn');
const dropzoneIdle    = document.getElementById('dropzone-idle');
const dropzonePreview = document.getElementById('dropzone-preview');
const dropzoneAnalyze = document.getElementById('dropzone-analyzing');
const previewImg      = document.getElementById('preview-img');

const resultsPanel    = document.getElementById('results-panel');
const resultsIdle     = document.getElementById('results-idle');
const resultsOutput   = document.getElementById('results-output');
const resultsError    = document.getElementById('results-error');
const errorMessage    = document.getElementById('error-message');

const predName        = document.getElementById('pred-name');
const predConfidence  = document.getElementById('pred-confidence');
const confidenceBars  = document.getElementById('confidence-bars');
const resultsTimestamp = document.getElementById('results-timestamp');

const modelStatusTag  = document.getElementById('model-status-tag');

// ─── Model loading ────────────────────────────────────────────────────────────

async function loadModel() {
  setModelStatus('loading');
  try {
    model = await tf.loadLayersModel(MODEL_PATH);
    // Warm-up pass: run a tiny dummy tensor so first real predict is fast
    const dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
    const warmup = model.predict(dummy);
    warmup.dispose();
    dummy.dispose();

    modelLoaded = true;
    setModelStatus('ready');
    console.log('[OcularAI] Model loaded:', model.inputs[0].shape);
  } catch (err) {
    console.error('[OcularAI] Model failed to load:', err);
    setModelStatus('error');
    showError('TF.js ' + tf.version.tfjs + ' error: ' + (err.message || String(err)));
  }
}

function setModelStatus(state) {
  modelStatusTag.classList.remove('loaded', 'error');
  if (state === 'loading') {
    modelStatusTag.textContent = 'Loading model…';
  } else if (state === 'ready') {
    modelStatusTag.textContent = 'Model ready';
    modelStatusTag.classList.add('loaded');
  } else if (state === 'error') {
    modelStatusTag.textContent = 'Model not found';
    modelStatusTag.classList.add('error');
  }
}

// ─── Image preprocessing ──────────────────────────────────────────────────────

function preprocessImage(imgElement) {
  return tf.tidy(() => {
    // Teachable Machine expects: resize to 224x224, normalize [0,255] → [-1,1]
    const offset = tf.scalar(127.5);
    return tf.browser
      .fromPixels(imgElement)
      .toFloat()
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .sub(offset)
      .div(offset)
      .expandDims(0);  // shape: [1, 224, 224, 3], values in [-1, 1]
  });
}

// ─── Prediction ───────────────────────────────────────────────────────────────

async function runPrediction(imgElement) {
  if (!modelLoaded) {
    showError('Model not loaded yet. Please wait or refresh the page.');
    return;
  }

  showState('analyzing');

  // Small delay so the "Analyzing…" UI renders before the heavy tf computation
  await new Promise(r => setTimeout(r, 80));

  try {
    const tensor = preprocessImage(imgElement);
    const outputTensor = model.predict(tensor);
    const probabilities = await outputTensor.data();
    tensor.dispose();
    outputTensor.dispose();

    showResults(Array.from(probabilities));
  } catch (err) {
    console.error('[OcularAI] Prediction error:', err);
    showState('preview');
    showError('Prediction failed: ' + err.message);
  }
}

// ─── Results rendering ────────────────────────────────────────────────────────

function showResults(probs) {
  // Find top prediction
  const topIdx = probs.indexOf(Math.max(...probs));
  const topClass = CLASSES[topIdx];
  const topPct = (probs[topIdx] * 100).toFixed(1);

  // Update primary prediction
  predName.textContent = topClass.label;
  predConfidence.textContent = topPct + '%';

  // Timestamp
  const now = new Date();
  resultsTimestamp.textContent =
    now.getHours().toString().padStart(2,'0') + ':' +
    now.getMinutes().toString().padStart(2,'0') + ':' +
    now.getSeconds().toString().padStart(2,'0');

  // Build confidence bars (sorted descending)
  const sorted = CLASSES
    .map((c, i) => ({ ...c, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob);

  confidenceBars.innerHTML = '';
  sorted.forEach((item, rank) => {
    const isTop = rank === 0;
    const pct = (item.prob * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'conf-row';
    row.innerHTML = `
      <span class="conf-label${isTop ? ' top' : ''}">${item.label}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill${isTop ? ' top' : ''}" data-width="${pct}%"></div>
      </div>
      <span class="conf-pct${isTop ? ' top' : ''}">${pct}%</span>
    `;
    confidenceBars.appendChild(row);
  });

  // Switch to results view
  showState('results');

  // Animate bars after a tick
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.conf-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    });
  });
}

// ─── UI state management ──────────────────────────────────────────────────────

function showState(state) {
  // dropzone states
  dropzoneIdle.hidden    = state !== 'idle';
  dropzonePreview.hidden = state !== 'preview' && state !== 'analyzing' && state !== 'results';
  dropzoneAnalyze.hidden = state !== 'analyzing';

  // results panel states
  resultsIdle.hidden   = state !== 'idle';
  resultsOutput.hidden = state !== 'results';
  resultsError.hidden  = true;

  // Center idle state
  resultsPanel.style.alignItems = (state === 'idle') ? 'center' : 'flex-start';

  if (state === 'results') {
    // After analysis done, hide the analyzing overlay, keep preview visible
    dropzoneAnalyze.hidden = true;
    dropzonePreview.hidden = false;
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  resultsIdle.hidden   = true;
  resultsOutput.hidden = true;
  resultsError.hidden  = false;
  resultsPanel.style.alignItems = 'center';
}

// ─── File handling ────────────────────────────────────────────────────────────

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please upload a valid image file (JPG, PNG, JPEG, etc.)');
    return;
  }

  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.onload = () => {
      showState('preview');
      // Auto-run prediction
      runPrediction(previewImg);
    };
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentFile = null;
  previewImg.src = '';
  fileInput.value = '';
  showState('idle');
}

// ─── Drag and Drop ────────────────────────────────────────────────────────────

dropzone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', (e) => {
  // Only remove if leaving the dropzone itself, not a child
  if (!dropzone.contains(e.relatedTarget)) {
    dropzone.classList.remove('drag-over');
  }
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// Click to browse
dropzone.addEventListener('click', (e) => {
  if (e.target === clearBtn || clearBtn.contains(e.target)) return;
  if (dropzonePreview.hidden === false) return; // don't re-open while preview shown
  fileInput.click();
});

browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearImage();
});

// Keyboard support
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// ─── Class image cycling ──────────────────────────────────────────────────────

function startImageCycling() {
  document.querySelectorAll('.class-img-stack').forEach((stack) => {
    const imgs = Array.from(stack.querySelectorAll('img'));
    if (imgs.length < 2) return;

    let current = 0;
    // Stagger start time per card
    const delay = Math.random() * 3000;

    setTimeout(() => {
      setInterval(() => {
        imgs[current].classList.remove('active');
        imgs[current].style.opacity = '0';
        current = (current + 1) % imgs.length;
        imgs[current].style.opacity = '1';
        imgs[current].classList.add('active');
      }, 3000);
    }, delay);
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  showState('idle');
  startImageCycling();
  loadModel();
});
