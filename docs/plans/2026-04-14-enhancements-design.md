# OcularAI Enhancement Design
**Date:** 2026-04-14  
**Status:** Approved  
**Target:** Medical/clinical professionals

---

## Overview

Add five enhancements to the existing OcularAI retinal fundus classifier:
1. Camera capture via webcam/mobile camera
2. PDF export of the diagnostic report
3. Scan history sidebar (last 5 analyses, sessionStorage)
4. "Analyze Next Image" CTA after results
5. Visual polish — micro-interactions and smoother animations

No new files are created. All changes land in the three existing files:
`index.html`, `css/style.css`, `js/app.js`.

---

## Feature Designs

### 1. Camera Capture

**Entry point:** A "Use Camera" button inside the dropzone idle state, next to "browse file".

**Flow:**
1. User clicks "Use Camera"
2. A modal overlay opens with a live `<video>` element fed by `navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })`
3. A "Capture" button freezes the frame: draws video to a hidden `<canvas>`, converts to a Blob, stops the stream, closes the modal
4. The Blob feeds into the existing `handleFile()` pipeline — prediction runs identically to a file upload
5. A "Cancel" button closes the modal and stops the stream without capturing

**HTML additions:**
- `#camera-btn` — button inside `#dropzone-idle`
- `#camera-modal` — full-screen overlay with `<video id="camera-feed">`, `<canvas id="camera-canvas" hidden>`, `#capture-btn`, `#camera-cancel-btn`

**No external libraries.** Native browser `getUserMedia` API.

**Error handling:** If `getUserMedia` is denied or unavailable, show an inline error message inside the modal ("Camera not available — please use file upload").

---

### 2. PDF Export

**Entry point:** "Export PDF" button in the report panel footer, visible only when results are shown.

**Flow:**
1. User clicks "Export PDF"
2. `html2canvas` and `jsPDF` are **lazy-loaded** from jsDelivr CDN (only on first click, cached after)
3. `html2canvas` captures `#results-output` as a canvas at 2× scale
4. `jsPDF` creates an A4 document, adds OcularAI header text + timestamp, embeds the canvas image
5. File downloaded as `OcularAI-Report-HH-MM-SS.pdf`

**Lazy load URLs:**
- `https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js`
- `https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js`

**HTML additions:** `#export-btn` inside a new `.rpt-actions` row at the bottom of `#results-output`.

**UX:** Button shows "Generating PDF…" spinner state while capturing; reverts to "Export PDF" on completion.

---

### 3. Scan History Sidebar

**Layout:** A third column added to the `.workspace` grid on the right side (~220px wide). On screens ≤ 900px it collapses to a horizontal scroll strip below the workspace.

**Card anatomy:**
- Thumbnail (60×60px, `object-fit: cover`, rounded)
- Predicted label (e.g. "Glaucoma")
- Confidence percentage
- Timestamp

**Behaviour:**
- Clicking a card calls `restoreScan(scan)`: repopulates `#preview-img`, re-renders the report panel with stored results data
- Max 5 cards. When a 6th scan is added the oldest is removed (FIFO)
- Persisted in `sessionStorage` as JSON array (survives page refresh within the same tab)
- Empty state: "No prior scans" placeholder

**HTML additions:** `<aside class="history-panel" id="history-panel">` with `<div class="history-list" id="history-list">` inside.

---

### 4. Analyze Next Image CTA

**Entry point:** After results are displayed, a `.rpt-actions` row at the bottom of the report panel shows two buttons side by side:

```
[↓ Export PDF]   [→ Analyze Next Image]
```

**"Analyze Next Image" flow:**
1. Current scan is auto-saved to history sidebar (if not already there)
2. `clearImage()` is called — dropzone resets to idle state
3. Report panel resets to "Awaiting Image" idle state
4. User can immediately drop/browse/camera a new image

The existing "Clear" button (`#clear-btn`) in the upload panel topbar is **removed** — replaced entirely by this CTA.

---

### 5. Visual Polish

- **Confidence bars:** Animate from 0 → final width with a staggered 80ms delay per row (already partially implemented; make stagger explicit)
- **History card entrance:** Slide-in from right with `transform: translateX(20px)` → `0` + `opacity: 0` → `1`, duration 250ms
- **Export button:** Pulse glow on hover using `box-shadow` transition
- **Camera modal:** Fade-in backdrop + scale-up video container (`transform: scale(0.96)` → `1`)
- **"Analyze Next Image" button:** Sky-blue filled, contrasts with ghost-style Export button

---

## Workspace Grid (revised)

```
.workspace {
  display: grid;
  grid-template-columns: 1fr 1.4fr 220px;  /* upload | report | history */
  gap: 1px;
}

@media (max-width: 1100px) {
  grid-template-columns: 1fr 1.4fr;  /* history moves below */
}

@media (max-width: 900px) {
  grid-template-columns: 1fr;        /* stacked */
}
```

---

## Files Changed

| File | Scope of change |
|------|----------------|
| `index.html` | +camera modal, +history sidebar, +export/next-image buttons, remove clear-btn |
| `css/style.css` | +camera modal styles, +history panel/card styles, +rpt-actions styles, +polish animations |
| `js/app.js` | +camera functions, +lazy PDF export, +history CRUD, +restoreScan, update showState/showResults |

**CSS version bump:** `?v=9` → `?v=10`  
**JS version bump:** `?v=8` → `?v=9`

---

## Out of Scope

- Persistent history across browser sessions (localStorage) — sessionStorage only
- Multi-image batch analysis
- Server-side storage or sharing links
- Camera on Safari iOS (getUserMedia limited) — graceful fallback to file upload
