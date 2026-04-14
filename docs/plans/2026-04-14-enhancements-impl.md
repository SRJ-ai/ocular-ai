# OcularAI Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add camera capture, PDF export, scan history sidebar, "Analyze Next Image" CTA, and visual polish to the existing OcularAI retinal fundus classifier.

**Architecture:** All changes land in the three existing files (`index.html`, `css/style.css`, `js/app.js`). No new files. History persists in `sessionStorage`. PDF export lazy-loads `html2canvas` + `jsPDF` from CDN only on first use.

**Tech Stack:** Vanilla JS (ES6), TensorFlow.js 4.22, html2canvas 1.4.1 (lazy), jsPDF 2.5.1 (lazy), CSS Grid/Flexbox, sessionStorage, getUserMedia API

---

## Context: How the app works today

- `showState(state)` controls visibility via `hidden` attribute on panel elements. States: `'idle'`, `'preview'`, `'analyzing'`, `'results'`
- `showResults(probs)` renders the diagnostic report and calls `showState('results')`
- `handleFile(file)` → reads file → sets `previewImg.src` → calls `runPrediction(previewImg)`
- `clearImage()` resets everything to idle state
- CSS `[hidden] { display: none !important; }` enforces hidden attribute (already in place)
- CSS link: `css/style.css?v=9` — bump to `?v=10` when CSS changes
- JS script: `js/app.js?v=8` — bump to `?v=9` when JS changes

**No test framework exists.** Each task verified manually in local browser (open `index.html` directly via file:// or local server).

---

## Task 1: Remove old Clear button + add "Analyze Next Image" CTA

**Files:**
- Modify: `index.html` (upload panel topbar, results output section)
- Modify: `css/style.css` (remove `.btn-clear` styles, add `.rpt-actions` styles)
- Modify: `js/app.js` (update `showState`, update `clearBtn` references)

**Step 1: Update index.html — remove clear-btn from topbar, add rpt-actions to report**

In `index.html`, find the upload panel topbar:
```html
<div class="panel-topbar">
  <span class="panel-label">PATIENT IMAGE</span>
  <button class="btn-clear" id="clear-btn" hidden>
    <svg viewBox="0 0 14 14" fill="none"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
    Clear
  </button>
</div>
```
Replace with (remove the clear button entirely):
```html
<div class="panel-topbar">
  <span class="panel-label">PATIENT IMAGE</span>
</div>
```

Find the last `<div class="rpt-rule"></div>` before the disclaimer block inside `#results-output` and add after the disclaimer rpt-block:
```html
          <div class="rpt-rule"></div>

          <div class="rpt-actions">
            <button class="btn-export" id="export-btn">
              <svg viewBox="0 0 16 16" fill="none"><path d="M8 2v8M5 7l3 3 3-3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11v1.5A1.5 1.5 0 003.5 14h9a1.5 1.5 0 001.5-1.5V11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
              Export PDF
            </button>
            <button class="btn-next" id="next-btn">
              Analyze Next Image
              <svg viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </button>
          </div>
```

**Step 2: Update js/app.js — wire next-btn, remove clearBtn references**

At top of DOM refs section, remove `clearBtn` line. Add:
```js
const nextBtn         = document.getElementById('next-btn');
```

Remove the `clearBtn.addEventListener('click', ...)` block entirely.

Remove the `if (e.target === clearBtn || clearBtn.contains(e.target)) return;` line from the dropzone click handler.

Add after the `browseBtn` event listener:
```js
nextBtn.addEventListener('click', () => {
  addToHistory();   // save current scan before clearing (function added in Task 3)
  clearImage();
});
```

Update `showState()` — remove the `clearBtn.hidden` line since it no longer exists.

**Step 3: Add CSS for .rpt-actions, .btn-export, .btn-next**

In `css/style.css`, add after the `.rpt-disclaimer` block:
```css
/* ─── Report Actions ─────────────────────────────── */
.rpt-actions {
  display: flex;
  gap: 0.75rem;
  padding: 1.25rem 1.5rem;
}

.btn-export {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.6rem 1.1rem;
  border: 1px solid var(--border-med);
  border-radius: var(--radius);
  background: transparent;
  color: var(--text-dim);
  font-size: 0.8rem;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  transition: var(--tr);
  cursor: pointer;
}
.btn-export svg { width: 14px; height: 14px; flex-shrink: 0; }
.btn-export:hover {
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: 0 0 12px var(--accent-glow);
}
.btn-export.loading { opacity: 0.6; pointer-events: none; }

.btn-next {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.6rem 1.25rem;
  border: none;
  border-radius: var(--radius);
  background: var(--accent);
  color: #fff;
  font-size: 0.8rem;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  font-weight: 500;
  cursor: pointer;
  transition: var(--tr);
  flex: 1;
  justify-content: center;
}
.btn-next svg { width: 14px; height: 14px; flex-shrink: 0; }
.btn-next:hover { background: #0284c7; }
```

Also remove the `.btn-clear` CSS block (find and delete it).

**Step 4: Bump version strings**

In `index.html`:
- `css/style.css?v=9` → `css/style.css?v=10`
- `js/app.js?v=8` → `js/app.js?v=9`

**Step 5: Manual verification**

Open `index.html` in browser. Upload an image. After results appear:
- ✓ Two buttons visible at bottom of report: "Export PDF" (ghost) and "Analyze Next Image" (blue)
- ✓ "Analyze Next Image" resets UI to idle state
- ✓ No Clear button in upload panel topbar

**Step 6: Commit**
```bash
git add index.html css/style.css js/app.js
git commit -m "feat: replace clear btn with Analyze Next Image CTA + Export PDF button"
```

---

## Task 2: PDF Export (lazy-load html2canvas + jsPDF)

**Files:**
- Modify: `js/app.js` (add `exportPDF` function, wire export-btn)

**Step 1: Add lazy loader utility at top of app.js constants section**

```js
// ─── Lazy script loader ───────────────────────────────────────────────────────
function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}
```

**Step 2: Add exportPDF function to app.js**

Add after the `showError` function:
```js
// ─── PDF Export ───────────────────────────────────────────────────────────────

async function exportPDF() {
  const btn = document.getElementById('export-btn');
  btn.classList.add('loading');
  btn.textContent = 'Generating…';

  try {
    await loadScript('https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js');

    const reportEl = document.getElementById('results-output');
    const canvas = await html2canvas(reportEl, {
      backgroundColor: '#131924',
      scale: 2,
      useCORS: true,
      logging: false,
    });

    const imgData = canvas.toDataURL('image/png');
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentW = pageW - margin * 2;

    // Header
    pdf.setFontSize(10);
    pdf.setTextColor(150, 170, 190);
    pdf.text('OcularAI — Retinal Fundus Classifier', margin, 12);
    pdf.text('Educational use only · Not a clinical diagnosis', pageW - margin, 12, { align: 'right' });
    pdf.setDrawColor(50, 70, 90);
    pdf.line(margin, 15, pageW - margin, 15);

    // Report image
    const imgH = (canvas.height / canvas.width) * contentW;
    const maxH = pageH - 30;
    const finalH = Math.min(imgH, maxH);
    pdf.addImage(imgData, 'PNG', margin, 20, contentW, finalH);

    // Footer
    pdf.setFontSize(8);
    pdf.setTextColor(100, 120, 140);
    const now = new Date();
    pdf.text(`Generated ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`, margin, pageH - 8);

    const ts = now.getHours().toString().padStart(2,'0') +
               now.getMinutes().toString().padStart(2,'0') +
               now.getSeconds().toString().padStart(2,'0');
    pdf.save(`OcularAI-Report-${ts}.pdf`);

  } catch (err) {
    console.error('[OcularAI] PDF export failed:', err);
    alert('PDF export failed: ' + err.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = `<svg viewBox="0 0 16 16" fill="none"><path d="M8 2v8M5 7l3 3 3-3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11v1.5A1.5 1.5 0 003.5 14h9a1.5 1.5 0 001.5-1.5V11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg> Export PDF`;
  }
}
```

**Step 3: Wire export-btn in DOM refs + event listener**

Add to DOM refs:
```js
const exportBtn       = document.getElementById('export-btn');
```

Add event listener after `nextBtn` listener:
```js
exportBtn.addEventListener('click', exportPDF);
```

**Step 4: Manual verification**

Upload an image, wait for results. Click "Export PDF":
- ✓ Button shows "Generating…" for 1-2 seconds
- ✓ PDF downloads as `OcularAI-Report-HHMMSS.pdf`
- ✓ PDF contains header, report content, footer with timestamp
- ✓ Second click is instant (scripts already loaded)

**Step 5: Commit**
```bash
git add js/app.js
git commit -m "feat: lazy-load html2canvas + jsPDF for branded PDF export"
```

---

## Task 3: Scan History Sidebar

**Files:**
- Modify: `index.html` (add history aside element + update workspace grid)
- Modify: `css/style.css` (add history panel styles + update workspace grid columns)
- Modify: `js/app.js` (add history CRUD + restoreScan + call addToHistory in nextBtn)

**Step 1: Add history aside to index.html**

Inside `.workspace` div, after the `#results-panel` div, add:
```html
      <!-- History Sidebar -->
      <aside class="history-panel" id="history-panel">
        <div class="panel-topbar">
          <span class="panel-label">PRIOR SCANS</span>
          <span class="history-count" id="history-count">0 / 5</span>
        </div>
        <div class="history-list" id="history-list">
          <div class="history-empty" id="history-empty">
            <svg viewBox="0 0 40 40" fill="none">
              <rect x="6" y="4" width="28" height="32" rx="2" stroke="currentColor" stroke-width="1" opacity="0.2"/>
              <path d="M12 14h16M12 20h10" stroke="currentColor" stroke-width="1" stroke-linecap="round" opacity="0.2"/>
            </svg>
            <p>No prior scans</p>
          </div>
        </div>
      </aside>
```

**Step 2: Update workspace CSS grid to 3 columns**

Find `.workspace` in `css/style.css` and replace with:
```css
.workspace {
  display: grid;
  grid-template-columns: 1fr 1.4fr 220px;
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
```

Update the 900px breakpoint to handle history below workspace:
```css
@media (max-width: 1100px) {
  .workspace { grid-template-columns: 1fr 1.4fr; }
  .history-panel { grid-column: 1 / -1; flex-direction: row; }
  .history-list { flex-direction: row; overflow-x: auto; }
}
@media (max-width: 900px) {
  .workspace { grid-template-columns: 1fr; }
}
```

**Step 3: Add history panel CSS**

Add after `.rpt-actions` styles:
```css
/* ─── History Panel ──────────────────────────────── */
.history-panel {
  background: var(--surface);
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.history-count {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-muted);
  letter-spacing: 0.06em;
}

.history-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.history-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  height: 100%;
  min-height: 120px;
  color: var(--text-muted);
  font-size: 0.72rem;
  text-align: center;
}
.history-empty svg { width: 40px; height: 40px; }

.history-card {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.5rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: var(--tr);
  animation: slideInRight 0.25s ease;
}
.history-card:hover {
  border-color: var(--border-med);
  background: #222d42;
}

@keyframes slideInRight {
  from { transform: translateX(20px); opacity: 0; }
  to   { transform: translateX(0);    opacity: 1; }
}

.history-thumb {
  width: 44px;
  height: 44px;
  border-radius: 4px;
  object-fit: cover;
  flex-shrink: 0;
  background: var(--surface);
}

.history-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.history-label {
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.history-meta {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  color: var(--text-muted);
}
```

**Step 4: Add history JS logic to app.js**

Add to DOM refs:
```js
const historyList     = document.getElementById('history-list');
const historyEmpty    = document.getElementById('history-empty');
const historyCount    = document.getElementById('history-count');
```

Add history functions after the `exportPDF` function:
```js
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
  // Only save if results are currently shown
  if (resultsOutput.hidden) return;

  const label = predName.textContent;
  const conf  = predConfidence.textContent;
  const ts    = resultsTimestamp.textContent;
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

  // Clear existing cards (keep empty placeholder in DOM)
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
  predName.textContent = scan.label;
  predConfidence.textContent = scan.conf;
  resultsTimestamp.textContent = scan.ts;
  // Show results state (no re-inference needed)
  dropzoneIdle.hidden    = true;
  dropzonePreview.hidden = false;
  dropzoneAnalyze.hidden = true;
  resultsIdle.hidden     = true;
  resultsOutput.hidden   = false;
  resultsError.hidden    = true;
  resultsPanel.style.alignItems = 'flex-start';
  // Note: confidence bars not restored (data not stored) — show empty diff table
  confidenceBars.innerHTML = '<p style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem 0">Restored from history</p>';
}
```

**Step 5: Call renderHistory on DOMContentLoaded**

In the `DOMContentLoaded` handler, add:
```js
renderHistory();
```

**Step 6: Manual verification**

- ✓ History sidebar visible as third column on desktop
- ✓ "No prior scans" placeholder shown on load
- ✓ After analysis, click "Analyze Next Image" → scan card appears in history with thumbnail, label, confidence, time
- ✓ Up to 5 cards shown; 6th pushes oldest off
- ✓ Click a history card → image and report restored
- ✓ Cards animate slide-in from right

**Step 7: Commit**
```bash
git add index.html css/style.css js/app.js
git commit -m "feat: add scan history sidebar with sessionStorage + restoreScan"
```

---

## Task 4: Camera Capture

**Files:**
- Modify: `index.html` (add camera button in dropzone idle, add camera modal)
- Modify: `css/style.css` (add camera modal + camera button styles)
- Modify: `js/app.js` (add camera functions)

**Step 1: Add camera button to dropzone idle state in index.html**

Find the `dz-secondary` paragraph inside `#dropzone-idle`:
```html
<p class="dz-secondary">or <button class="dz-browse" id="browse-btn">browse file</button></p>
```
Replace with:
```html
<p class="dz-secondary">
  or <button class="dz-browse" id="browse-btn">browse file</button>
  <span class="dz-sep">·</span>
  <button class="dz-browse" id="camera-btn">use camera</button>
</p>
```

**Step 2: Add camera modal to index.html**

Before the closing `</body>` tag (after the footer), add:
```html
<!-- Camera Modal -->
<div id="camera-modal" class="camera-modal" hidden>
  <div class="camera-backdrop"></div>
  <div class="camera-dialog">
    <div class="camera-topbar">
      <span class="panel-label">CAMERA CAPTURE</span>
      <button class="camera-close" id="camera-cancel-btn">
        <svg viewBox="0 0 14 14" fill="none"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </button>
    </div>
    <video id="camera-feed" autoplay playsinline muted></video>
    <canvas id="camera-canvas" hidden></canvas>
    <div class="camera-footer">
      <p class="camera-hint">Position the retinal fundus image in frame</p>
      <button class="btn-capture" id="capture-btn">
        <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>
        Capture
      </button>
    </div>
    <p class="camera-error" id="camera-error" hidden></p>
  </div>
</div>
```

**Step 3: Add camera CSS**

Add after `.history-card` styles:
```css
/* ─── Camera Modal ───────────────────────────────── */
.camera-modal {
  position: fixed;
  inset: 0;
  z-index: 500;
  display: flex;
  align-items: center;
  justify-content: center;
}

.camera-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.75);
  backdrop-filter: blur(6px);
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.camera-dialog {
  position: relative;
  z-index: 1;
  background: var(--surface);
  border: 1px solid var(--border-med);
  border-radius: var(--radius-lg);
  overflow: hidden;
  width: min(480px, 92vw);
  animation: scaleIn 0.2s ease;
}

@keyframes scaleIn {
  from { transform: scale(0.95); opacity: 0; }
  to   { transform: scale(1);    opacity: 1; }
}

.camera-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1.25rem;
  border-bottom: 1px solid var(--border);
}

.camera-close {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  color: var(--text-dim);
  transition: var(--tr);
  cursor: pointer;
  background: transparent;
  border: none;
}
.camera-close svg { width: 12px; height: 12px; }
.camera-close:hover { background: var(--surface-2); color: var(--text); }

#camera-feed {
  width: 100%;
  max-height: 320px;
  object-fit: cover;
  display: block;
  background: #000;
}

.camera-footer {
  padding: 1rem 1.25rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-top: 1px solid var(--border);
}

.camera-hint {
  font-size: 0.72rem;
  color: var(--text-muted);
  flex: 1;
}

.btn-capture {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.2rem;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  font-family: var(--font-mono);
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  transition: var(--tr);
  white-space: nowrap;
}
.btn-capture svg { width: 18px; height: 18px; }
.btn-capture:hover { background: #0284c7; }

.camera-error {
  padding: 0.6rem 1.25rem;
  color: var(--red);
  font-size: 0.78rem;
  text-align: center;
  border-top: 1px solid var(--border);
}

.dz-sep {
  color: var(--text-muted);
  margin: 0 0.25rem;
}
```

**Step 4: Add camera JS functions to app.js**

Add to DOM refs:
```js
const cameraBtn       = document.getElementById('camera-btn');
const cameraModal     = document.getElementById('camera-modal');
const cameraFeed      = document.getElementById('camera-feed');
const cameraCanvas    = document.getElementById('camera-canvas');
const captureBtn      = document.getElementById('capture-btn');
const cameraCancelBtn = document.getElementById('camera-cancel-btn');
const cameraError     = document.getElementById('camera-error');
```

Add camera functions after `clearImage`:
```js
// ─── Camera Capture ───────────────────────────────────────────────────────────

let cameraStream = null;

async function openCamera() {
  cameraError.hidden = true;
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
  captureBtn.disabled = false;
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
// Close on backdrop click
cameraModal.addEventListener('click', (e) => {
  if (e.target === cameraModal || e.target.classList.contains('camera-backdrop')) closeCamera();
});
```

**Step 5: Manual verification**

- ✓ "use camera" text link appears in dropzone idle state
- ✓ Clicking it opens modal with live video feed
- ✓ "Capture" freezes frame and runs prediction
- ✓ Modal closes and analyzing spinner shows
- ✓ Cancel closes modal without triggering anything
- ✓ On a device without camera: error message shown in modal
- ✓ Backdrop click closes modal

**Step 6: Commit**
```bash
git add index.html css/style.css js/app.js
git commit -m "feat: add camera capture via getUserMedia with modal UI"
```

---

## Task 5: Visual Polish

**Files:**
- Modify: `css/style.css` (staggered confidence bar animation, scan spinner refinement)
- Modify: `js/app.js` (stagger delay on confidence bars)

**Step 1: Add stagger delay to confidence bars in showResults()**

In `showResults()`, update the `forEach` that creates bar rows to add a stagger delay:
```js
sorted.forEach((item, rank) => {
    const isTop = rank === 0;
    const pct = (item.prob * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'conf-row';
    row.style.setProperty('--stagger', rank);   // ADD THIS LINE
    row.innerHTML = `
      <span class="conf-label${isTop ? ' top' : ''}">${item.label}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill${isTop ? ' top' : ''}" data-width="${pct}%"></div>
      </div>
      <span class="conf-pct${isTop ? ' top' : ''}">${pct}%</span>
    `;
    confidenceBars.appendChild(row);
  });
```

**Step 2: Add stagger CSS to confidence rows**

Find `.conf-row` in `css/style.css` and add:
```css
.conf-row {
  /* existing styles unchanged, add: */
  animation: fadeSlideUp 0.3s ease both;
  animation-delay: calc(var(--stagger, 0) * 80ms);
}

@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
```

**Step 3: Manual verification**

Run analysis. On results:
- ✓ Confidence rows appear one by one with 80ms stagger, sliding up
- ✓ Bars animate width from 0 to final value
- ✓ History cards slide in from right when added

**Step 4: Commit**
```bash
git add css/style.css js/app.js
git commit -m "polish: staggered confidence bar animations + history card slide-in"
```

---

## Task 6: Deploy + Verify Live

**Step 1: Push to GitHub**
```bash
git push origin main
```

**Step 2: Wait for Pages build (1-3 min), then verify at:**
`https://srj-ai.github.io/ocular-ai/`

**Verification checklist:**
- ✓ CSS v10 loads (check Network tab)
- ✓ JS v9 loads
- ✓ Dropzone shows "use camera" link
- ✓ Upload image → results appear with Export PDF + Analyze Next Image buttons
- ✓ Export PDF downloads a branded PDF
- ✓ Analyze Next Image resets UI and adds card to history sidebar
- ✓ History card click restores scan
- ✓ Camera modal opens (if on device with camera)
- ✓ Confidence bars animate with stagger

**Step 3: Commit any last fixes discovered during verification**
