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
  initCreditBanner();
  initSampleGallery();
});

// ─── Credit Banner ────────────────────────────────────────────────────────────

function initCreditBanner() {
  const banner = document.getElementById('credit-banner');
  const closeBtn = document.getElementById('credit-banner-close');
  if (!banner || !closeBtn) return;
  if (sessionStorage.getItem('banner_dismissed')) {
    banner.hidden = true;
    return;
  }
  closeBtn.addEventListener('click', () => {
    banner.hidden = true;
    sessionStorage.setItem('banner_dismissed', '1');
  });
}

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
