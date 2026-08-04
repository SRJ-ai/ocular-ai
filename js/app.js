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

const XAI_RATIONALE = {
  cataract:             'Diffuse lens opacity reduces fundus clarity — attenuated vascular contrast across the central field drives this Cataract classification.',
  diabetic_retinopathy: 'Microaneurysms and haemorrhagic lesions concentrated in the posterior pole are the primary features driving this Diabetic Retinopathy classification.',
  glaucoma:             'Enlarged cup-to-disc ratio and optic disc pallor, particularly in the nasal region, are the dominant indicators of Glaucoma in this scan.',
  normal:               'Clear optic disc margins, regular vascular architecture, and absence of pathological lesions support a Normal retina classification.',
};

// ─── Lazy script loader ───────────────────────────────────────────────────────
const _scriptCache = new Map();
function loadScript(src) {
  if (_scriptCache.has(src)) return _scriptCache.get(src);
  const p = new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src;
    s.onload  = resolve;
    s.onerror = () => { _scriptCache.delete(src); reject(new Error(`Failed to load script: ${src}`)); };
    document.head.appendChild(s);
  });
  _scriptCache.set(src, p);
  return p;
}

// ─── State ────────────────────────────────────────────────────────────────────

let model = null;
let modelLoaded = false;
let currentFile = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const dropzone        = document.getElementById('dropzone');
const fileInput       = document.getElementById('file-input');
const browseBtn       = document.getElementById('browse-btn');
const nextBtn         = document.getElementById('next-btn');
const exportBtn       = document.getElementById('export-btn');
const historyList     = document.getElementById('history-list');
const historyEmpty    = document.getElementById('history-empty');
const historyCount    = document.getElementById('history-count');
const cameraBtn       = document.getElementById('camera-btn');
const cameraModal     = document.getElementById('camera-modal');
const cameraFeed      = document.getElementById('camera-feed');
const cameraCanvas    = document.getElementById('camera-canvas');
const captureBtn      = document.getElementById('capture-btn');
const cameraCancelBtn = document.getElementById('camera-cancel-btn');
const cameraError     = document.getElementById('camera-error');
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

const xaiSection    = document.getElementById('xai-section');
const xaiToggle     = document.getElementById('xai-toggle');
const xaiBody       = document.getElementById('xai-body');
const xaiSpinner    = document.getElementById('xai-spinner');
const xaiContent    = document.getElementById('xai-content');
const xaiRegionList = document.getElementById('xai-region-list');
const xaiRationale  = document.getElementById('xai-rationale');
const heatmapCanvas = document.getElementById('heatmap-canvas');
const xaiSlider        = document.getElementById('xai-slider');
const xaiSliderDivider = document.getElementById('xai-slider-divider');

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
    row.style.setProperty('--stagger', rank);
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

  // Fire XAI async — report renders now, explanation fills in after. Reuses the
  // top-class confidence as the occlusion baseline (no extra full-image predict).
  runXAI(previewImg, topIdx, probs[topIdx]);
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
  // Cancel any in-flight XAI pass and reset the heatmap canvas
  xaiRunId++;
  heatmapCanvas.hidden = true;
  heatmapCanvas.width  = 0;
  heatmapCanvas.height = 0;
  xaiSlider.hidden = true;
  // Reset XAI section
  xaiSection.hidden = true;
  xaiBody.hidden    = true;
  xaiToggle.setAttribute('aria-expanded', 'false');
  xaiContent.hidden = true;
  xaiSpinner.hidden = false;
  showState('idle');
}

// ─── Camera Capture ───────────────────────────────────────────────────────────

let cameraStream = null;

async function openCamera() {
  cameraError.hidden = true;
  captureBtn.disabled = false;
  cameraModal.hidden = false;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    cameraFeed.srcObject = cameraStream;
  } catch (err) {
    cameraError.textContent = 'Camera not available — please use file upload. (' + err.message + ')';
    cameraError.hidden = false;
    captureBtn.disabled = true;
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraFeed.srcObject = null;
  cameraModal.hidden = true;
}

function captureFrame() {
  const w = cameraFeed.videoWidth  || 640;
  const h = cameraFeed.videoHeight || 480;
  cameraCanvas.width  = w;
  cameraCanvas.height = h;
  cameraCanvas.getContext('2d').drawImage(cameraFeed, 0, 0, w, h);
  closeCamera();
  cameraCanvas.toBlob(blob => {
    if (blob) handleFile(new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }));
  }, 'image/jpeg', 0.92);
}

// XAI accordion expand/collapse
xaiToggle.addEventListener('click', () => {
  const open = xaiToggle.getAttribute('aria-expanded') === 'true';
  xaiToggle.setAttribute('aria-expanded', String(!open));
  xaiBody.hidden = open;
});

cameraBtn.addEventListener('click', (e) => { e.stopPropagation(); openCamera(); });
captureBtn.addEventListener('click', captureFrame);
cameraCancelBtn.addEventListener('click', closeCamera);
cameraModal.addEventListener('click', (e) => {
  if (e.target === cameraModal || e.target.classList.contains('camera-backdrop')) closeCamera();
});

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
  if (dropzonePreview.hidden === false) return; // don't re-open while preview shown
  fileInput.click();
});

browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

nextBtn.addEventListener('click', () => {
  addToHistory();   // will be implemented in Task 3; for now this is a no-op placeholder call
  clearImage();
});

exportBtn.addEventListener('click', exportPDF);   // exportPDF will be implemented in Task 2

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
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
  renderHistory();
  startImageCycling();
  loadModel();
  initSampleGallery();
});

// ─── Sample Gallery ───────────────────────────────────────────────────────────

function initSampleGallery() {
  // Tab switching
  document.querySelectorAll('.samples-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.samples-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.samples-panel').forEach(p => { p.hidden = true; });
      tab.classList.add('active');
      const panel = document.querySelector(`.samples-panel[data-class="${tab.dataset.class}"]`);
      if (panel) panel.hidden = false;
    });
  });

  // Click to load sample
  document.querySelectorAll('.sample-thumb').forEach(img => {
    img.addEventListener('click', () => loadSample(img));
  });
}

async function loadSample(img) {
  img.classList.add('loading');
  try {
    const response = await fetch(img.dataset.src);
    const blob = await response.blob();
    const ext = img.dataset.ext || 'jpg';
    const mime = ext === 'jpeg' || ext === 'jpg' ? 'image/jpeg' : 'image/png';
    const file = new File([blob], `sample.${ext}`, { type: mime });
    handleFile(file);
    // Scroll to workspace
    document.querySelector('.workspace-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    console.error('[OcularAI] Failed to load sample:', err);
  } finally {
    img.classList.remove('loading');
  }
}

// ─── XAI: Explainability (occlusion sensitivity) ──────────────────────────────

// Bumped on every new run so a stale in-flight XAI pass can bail out instead of
// drawing onto the canvas of a newer image.
let xaiRunId = 0;

// Wipe-slider position: % of the image width. Left of it shows the original,
// right of it shows the occlusion heatmap.
let xaiWipePct = 50;

// Reveal the heatmap only to the right of the wipe divider; keep the divider
// line in sync. clipPath % is relative to the canvas border-box (== slider width).
function applyXaiWipe() {
  heatmapCanvas.style.clipPath = `inset(0 0 0 ${xaiWipePct}%)`;
  xaiSliderDivider.style.left = xaiWipePct + '%';
  xaiSlider.setAttribute('aria-valuenow', String(Math.round(xaiWipePct)));
}

function setXaiWipeFromClientX(clientX) {
  const rect = xaiSlider.getBoundingClientRect();
  if (!rect.width) return;
  xaiWipePct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
  applyXaiWipe();
}

function setupXaiSlider() {
  let dragging = false;
  xaiSlider.addEventListener('pointerdown', (e) => {
    dragging = true;
    xaiSlider.setPointerCapture(e.pointerId);
    setXaiWipeFromClientX(e.clientX);
    e.preventDefault();
  });
  xaiSlider.addEventListener('pointermove', (e) => {
    if (dragging) setXaiWipeFromClientX(e.clientX);
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false;
    try { xaiSlider.releasePointerCapture(e.pointerId); } catch (_) {}
  };
  xaiSlider.addEventListener('pointerup', end);
  xaiSlider.addEventListener('pointercancel', end);
  // Keyboard: arrows nudge the divider
  xaiSlider.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft')  { xaiWipePct = Math.max(0, xaiWipePct - step); applyXaiWipe(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { xaiWipePct = Math.min(100, xaiWipePct + step); applyXaiWipe(); e.preventDefault(); }
  });
}
setupXaiSlider();

// Rendered rect of an object-fit:contain image inside its element box, so the
// heatmap overlays the visible (letterboxed) retina, not the padded element box.
function getContainRect(imgElement) {
  const rect = imgElement.getBoundingClientRect();
  const natW = imgElement.naturalWidth  || rect.width;
  const natH = imgElement.naturalHeight || rect.height;
  const scale = Math.min(rect.width / natW, rect.height / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  return {
    elemRect: rect,
    width:   dispW,
    height:  dispH,
    offsetX: (rect.width  - dispW) / 2,
    offsetY: (rect.height - dispH) / 2,
  };
}

function drawHeatmap(scores, imgElement) {
  const cr = getContainRect(imgElement);
  const parentRect = imgElement.parentElement.getBoundingClientRect();
  const w = Math.round(cr.width);
  const h = Math.round(cr.height);
  const left = Math.round(cr.elemRect.left - parentRect.left + cr.offsetX);
  const top  = Math.round(cr.elemRect.top  - parentRect.top  + cr.offsetY);

  heatmapCanvas.width  = w;
  heatmapCanvas.height = h;
  heatmapCanvas.style.width  = w + 'px';
  heatmapCanvas.style.height = h + 'px';
  heatmapCanvas.style.left   = left + 'px';
  heatmapCanvas.style.top    = top + 'px';

  const ctx = heatmapCanvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const GRID = 8;
  const pw = w / GRID;
  const ph = h / GRID;

  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = (max - min) || 1;

  scores.forEach((score, i) => {
    const row  = Math.floor(i / GRID);
    const col  = i % GRID;
    const norm = (score - min) / range; // 1 = high influence (red), 0 = low (blue)

    let r, g, b, a;
    if (norm >= 0.5) {
      r = 239; g = 68; b = 68;              // high influence
      a = (norm - 0.5) * 2 * 0.55;
    } else {
      r = 59; g = 130; b = 246;             // low influence
      a = (0.5 - norm) * 2 * 0.4;
    }

    ctx.fillStyle = `rgba(${r},${g},${b},${a.toFixed(3)})`;
    ctx.fillRect(Math.round(col * pw), Math.round(row * ph), Math.ceil(pw), Math.ceil(ph));
  });

  // Position the wipe slider over the exact same rect as the heatmap
  xaiSlider.style.left   = left + 'px';
  xaiSlider.style.top    = top + 'px';
  xaiSlider.style.width  = w + 'px';
  xaiSlider.style.height = h + 'px';
  applyXaiWipe();

  heatmapCanvas.hidden = false;
  xaiSlider.hidden = false;
}

function renderRegionBars(scores) {
  const GRID = 8;
  const regions = [
    { label: 'Superior', rows: [0, 3], cols: [0, 7] },
    { label: 'Inferior', rows: [4, 7], cols: [0, 7] },
    { label: 'Nasal',    rows: [0, 7], cols: [0, 3] },
    { label: 'Temporal', rows: [0, 7], cols: [4, 7] },
  ];

  const regionScores = regions.map(r => {
    let total = 0, count = 0;
    for (let row = r.rows[0]; row <= r.rows[1]; row++) {
      for (let col = r.cols[0]; col <= r.cols[1]; col++) {
        total += Math.max(0, scores[row * GRID + col]);
        count++;
      }
    }
    return { label: r.label, avg: total / count };
  });

  const maxAvg = Math.max(...regionScores.map(r => r.avg), 0.0001);

  xaiRegionList.innerHTML = '';
  regionScores.forEach(r => {
    const pct = Math.round((r.avg / maxAvg) * 100);
    const row = document.createElement('div');
    row.className = 'conf-row';
    row.innerHTML = `
      <span class="conf-label">${r.label}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" data-width="${pct}%"></div>
      </div>
      <span class="conf-pct">${pct}%</span>
    `;
    xaiRegionList.appendChild(row);
  });

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      xaiRegionList.querySelectorAll('.conf-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    });
  });
}

// Occlusion sensitivity: black out each of 64 (8×8) patches on the 224×224 model
// input and measure Δconfidence for the predicted class. Runs in chunked batches
// so it is one-forward-pass-per-chunk (not 64 separate predicts) and yields to the
// UI between chunks. `baselineConf` is reused from the original prediction — no
// extra full-image inference.
async function runXAI(imgElement, topIdx, baselineConf) {
  const myRun = ++xaiRunId;
  xaiWipePct = 50; // fresh image → centre the wipe divider

  xaiSection.hidden = false;
  xaiToggle.setAttribute('aria-expanded', 'true');
  xaiBody.hidden    = false;
  xaiSpinner.hidden = false;
  xaiContent.hidden = true;

  try {
    if (!modelLoaded) throw new Error('Model not loaded');

    const GRID       = 8;
    const PATCH_PX   = IMG_SIZE / GRID;       // 28px per patch
    const CHUNK      = 8;                      // patches per forward pass
    const numClasses = CLASSES.length;
    const scores     = new Array(GRID * GRID).fill(0);

    // Source canvas: image squashed to the model's 224×224 geometry (matches the
    // resizeBilinear in preprocessImage, so patch coords line up with the input).
    const src = document.createElement('canvas');
    src.width = IMG_SIZE;
    src.height = IMG_SIZE;
    src.getContext('2d').drawImage(imgElement, 0, 0, IMG_SIZE, IMG_SIZE);

    for (let start = 0; start < GRID * GRID; start += CHUNK) {
      const end = Math.min(start + CHUNK, GRID * GRID);

      const batch = tf.tidy(() => {
        const imgs = [];
        for (let k = start; k < end; k++) {
          const row = Math.floor(k / GRID);
          const col = k % GRID;
          const oc  = document.createElement('canvas');
          oc.width  = IMG_SIZE;
          oc.height = IMG_SIZE;
          const ctx = oc.getContext('2d');
          ctx.drawImage(src, 0, 0);
          ctx.fillStyle = '#000000';
          ctx.fillRect(col * PATCH_PX, row * PATCH_PX, PATCH_PX, PATCH_PX);
          // Already 224×224 → normalize [0,255] → [-1,1] to match preprocessImage
          imgs.push(tf.browser.fromPixels(oc).toFloat().sub(127.5).div(127.5));
        }
        return tf.stack(imgs);                // [n, 224, 224, 3]
      });

      const out  = model.predict(batch);      // [n, numClasses]
      const data = await out.data();
      batch.dispose();
      out.dispose();

      // A newer image started while we were awaiting — abandon this stale pass.
      if (myRun !== xaiRunId) return;

      for (let j = 0; j < end - start; j++) {
        // Positive = occluding this patch dropped confidence → patch mattered.
        scores[start + j] = baselineConf - data[j * numClasses + topIdx];
      }

      await tf.nextFrame();                    // let the UI breathe between chunks
      if (myRun !== xaiRunId) return;
    }

    drawHeatmap(scores, imgElement);
    renderRegionBars(scores);
    xaiRationale.textContent = XAI_RATIONALE[CLASSES[topIdx].key] || '';

    xaiSpinner.hidden = true;
    xaiContent.hidden = false;

  } catch (err) {
    console.error('[OcularAI] XAI error:', err);
    if (myRun !== xaiRunId) return;
    xaiSpinner.hidden = true;
    xaiContent.hidden = false;
    xaiRegionList.innerHTML = '';
    xaiRationale.textContent = 'Explanation unavailable.';
  }
}

// ─── Scan History ─────────────────────────────────────────────────────────────

const HISTORY_KEY = 'ocularai_history';
const MAX_HISTORY = 5;

function getHistory() {
  try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function saveHistory(arr) {
  sessionStorage.setItem(HISTORY_KEY, JSON.stringify(arr));
}

function addToHistory() {
  if (resultsOutput.hidden) return;
  const label  = predName.textContent;
  const conf   = predConfidence.textContent;
  const ts     = resultsTimestamp.textContent;
  const imgSrc = previewImg.src;
  if (!label || label === '—' || !imgSrc) return;

  const scan = { label, conf, ts, imgSrc, id: Date.now() };
  let arr = getHistory();
  arr.unshift(scan);
  if (arr.length > MAX_HISTORY) arr = arr.slice(0, MAX_HISTORY);
  saveHistory(arr);
  renderHistory();
}

function renderHistory() {
  const arr = getHistory();
  historyCount.textContent = `${arr.length} / ${MAX_HISTORY}`;
  Array.from(historyList.querySelectorAll('.history-card')).forEach(el => el.remove());
  if (arr.length === 0) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;
  arr.forEach(scan => {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
      <img class="history-thumb" src="${scan.imgSrc}" alt="${scan.label}" />
      <div class="history-info">
        <span class="history-label">${scan.label}</span>
        <span class="history-meta">${scan.conf} · ${scan.ts}</span>
      </div>
    `;
    card.addEventListener('click', () => restoreScan(scan));
    historyList.appendChild(card);
  });
}

function restoreScan(scan) {
  previewImg.src = scan.imgSrc;
  predName.textContent        = scan.label;
  predConfidence.textContent  = scan.conf;
  resultsTimestamp.textContent = scan.ts;
  dropzoneIdle.hidden    = true;
  dropzonePreview.hidden = false;
  dropzoneAnalyze.hidden = true;
  resultsIdle.hidden     = true;
  resultsOutput.hidden   = false;
  resultsError.hidden    = true;
  resultsPanel.style.alignItems = 'flex-start';
  confidenceBars.innerHTML = '<p style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem 0">Restored from history</p>';
  // XAI is not recomputed for history restores — cancel any in-flight pass and hide it
  xaiRunId++;
  xaiSection.hidden = true;
  heatmapCanvas.hidden = true;
  xaiSlider.hidden = true;
}

// ─── PDF Export ───────────────────────────────────────────────────────────────

async function exportPDF() {
  const btn = document.getElementById('export-btn');
  const originalHTML = btn.innerHTML;
  btn.classList.add('loading');
  btn.textContent = 'Generating…';

  try {
    await loadScript('https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js');

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageW   = pdf.internal.pageSize.getWidth();
    const pageH   = pdf.internal.pageSize.getHeight();
    const margin  = 15;
    const cW      = pageW - margin * 2;

    // ── Background ──
    pdf.setFillColor(19, 25, 36);
    pdf.rect(0, 0, pageW, pageH, 'F');

    let y = margin;

    // ── Header ──
    pdf.setFontSize(9);
    pdf.setTextColor(14, 165, 233);
    pdf.text('OCULARAI', margin, y);
    pdf.setTextColor(148, 163, 184);
    pdf.text('Retinal Fundus Classifier · Educational use only', pageW - margin, y, { align: 'right' });
    y += 4;
    pdf.setDrawColor(50, 70, 90);
    pdf.setLineWidth(0.3);
    pdf.line(margin, y, pageW - margin, y);
    y += 6;

    // ── Report meta ──
    const ts = resultsTimestamp.textContent;
    pdf.setFontSize(7);
    pdf.setTextColor(75, 95, 120);
    pdf.text('DIAGNOSTIC REPORT', margin, y);
    pdf.text('RID · ' + ts, pageW - margin, y, { align: 'right' });
    y += 4;
    pdf.text('Model: MobileNet v2  ·  Training: 4,217 images', margin, y);
    y += 5;
    pdf.setDrawColor(30, 45, 65);
    pdf.line(margin, y, pageW - margin, y);
    y += 7;

    // ── Fundus image (left) ──
    const thumbSize = 55;
    try {
      const offscreen = document.createElement('canvas');
      offscreen.width  = 224;
      offscreen.height = 224;
      offscreen.getContext('2d').drawImage(previewImg, 0, 0, 224, 224);
      const imgData = offscreen.toDataURL('image/jpeg', 0.85);
      pdf.addImage(imgData, 'JPEG', margin, y, thumbSize, thumbSize);
    } catch (_) {
      pdf.setFillColor(28, 38, 56);
      pdf.rect(margin, y, thumbSize, thumbSize, 'F');
      pdf.setFontSize(7);
      pdf.setTextColor(75, 95, 120);
      pdf.text('Fundus Image', margin + thumbSize / 2, y + thumbSize / 2, { align: 'center' });
    }

    // ── Impression (right of image) ──
    const rx = margin + thumbSize + 8;
    pdf.setFontSize(7);
    pdf.setTextColor(75, 95, 120);
    pdf.text('IMPRESSION', rx, y + 5);

    pdf.setFontSize(20);
    pdf.setTextColor(226, 232, 240);
    pdf.text(predName.textContent, rx, y + 17);

    pdf.setFontSize(16);
    pdf.setTextColor(14, 165, 233);
    pdf.text(predConfidence.textContent, rx, y + 28);
    pdf.setFontSize(7);
    pdf.setTextColor(75, 95, 120);
    pdf.text('confidence', rx, y + 33);

    y += thumbSize + 6;
    pdf.setDrawColor(30, 45, 65);
    pdf.line(margin, y, pageW - margin, y);
    y += 7;

    // ── Differential Diagnosis ──
    pdf.setFontSize(7);
    pdf.setTextColor(75, 95, 120);
    pdf.text('DIFFERENTIAL DIAGNOSIS', margin, y);
    y += 5;

    const labelColW = 44;
    const pctColW   = 12;
    const barW      = cW - labelColW - pctColW - 4;

    document.querySelectorAll('.conf-row').forEach((row) => {
      const label  = row.querySelector('.conf-label')?.textContent?.trim() || '';
      const pctTxt = row.querySelector('.conf-pct')?.textContent?.trim()   || '0%';
      const isTop  = row.querySelector('.conf-label.top') !== null;
      const pctVal = parseFloat(pctTxt) || 0;

      pdf.setFontSize(8);
      pdf.setTextColor(...(isTop ? [226, 232, 240] : [148, 163, 184]));
      pdf.setFont('helvetica', isTop ? 'bold' : 'normal');
      pdf.text(label, margin, y + 2.5);
      pdf.setFont('helvetica', 'normal');

      // Track
      pdf.setFillColor(30, 40, 55);
      pdf.roundedRect(margin + labelColW, y, barW, 3.5, 1, 1, 'F');
      // Fill
      const fillW = (pctVal / 100) * barW;
      if (fillW > 0) {
        pdf.setFillColor(...(isTop ? [14, 165, 233] : [75, 95, 120]));
        pdf.roundedRect(margin + labelColW, y, fillW, 3.5, 1, 1, 'F');
      }

      pdf.setFontSize(7.5);
      pdf.setTextColor(...(isTop ? [14, 165, 233] : [148, 163, 184]));
      pdf.text(pctTxt, pageW - margin, y + 2.5, { align: 'right' });

      y += 8;
    });

    y += 2;
    pdf.setDrawColor(30, 45, 65);
    pdf.line(margin, y, pageW - margin, y);
    y += 6;

    // ── Disclaimer ──
    pdf.setFillColor(35, 28, 8);
    pdf.roundedRect(margin, y, cW, 14, 2, 2, 'F');
    pdf.setDrawColor(245, 158, 11);
    pdf.setLineWidth(0.3);
    pdf.roundedRect(margin, y, cW, 14, 2, 2, 'S');
    pdf.setFontSize(7.5);
    pdf.setTextColor(245, 158, 11);
    pdf.text(
      'This output is NOT a clinical diagnosis. For research and educational purposes only.',
      margin + 3, y + 5.5,
      { maxWidth: cW - 6 }
    );
    pdf.setTextColor(180, 140, 60);
    pdf.text('Consult a qualified ophthalmologist for medical decisions.', margin + 3, y + 10.5);

    // ── Footer ──
    pdf.setDrawColor(50, 70, 90);
    pdf.setLineWidth(0.3);
    pdf.line(margin, pageH - 12, pageW - margin, pageH - 12);
    pdf.setFontSize(7);
    pdf.setTextColor(75, 95, 120);
    const now = new Date();
    pdf.text(`Generated ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`, margin, pageH - 7);
    pdf.text('OcularAI · MobileNet v2 · TensorFlow.js 4.22', pageW - margin, pageH - 7, { align: 'right' });

    const fileTs = now.getHours().toString().padStart(2,'0') +
                   now.getMinutes().toString().padStart(2,'0') +
                   now.getSeconds().toString().padStart(2,'0');
    pdf.save(`OcularAI-Report-${fileTs}.pdf`);

  } catch (err) {
    console.error('[OcularAI] PDF export failed:', err);
    alert('PDF export failed: ' + err.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = originalHTML;
  }
}
