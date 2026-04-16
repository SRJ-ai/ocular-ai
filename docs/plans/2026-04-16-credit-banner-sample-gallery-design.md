# Credit Banner + Sample Gallery Design
**Date:** 2026-04-16  
**Status:** Approved

## Overview
Two additions:
1. Credit/caution banner (slim top bar + footer expansion) crediting Dr. P. Rangababu, Associate Professor, IIITDM Kurnool
2. Tabbed sample image gallery (section 03) with 10 images per class that load directly into the classifier

## Credit Notice
- **Top banner:** slim amber bar below nav, dismissable (sessionStorage), shows internship + educational warning
- **Footer:** second line added crediting Dr. P. Rangababu, Associate Professor, IIITDM Kurnool

## Sample Gallery
- **Placement:** New section `03 — SAMPLE IMAGES` between conditions and workspace
- **Tabs:** Cataract · Diabetic Retinopathy · Glaucoma · Normal
- **Grid:** 5-col × 2-row, 10 thumbnails per tab (80×80px, object-fit: cover)
- **Click behaviour:** fetch() image → File → handleFile() → auto-scroll to workspace → prediction runs
- **Files:** Copy 10 images from dataset/ into samples/ as 4.jpg–13.jpg per class

## Files Changed
- `index.html` — banner, gallery section, footer update
- `css/style.css` — banner + gallery styles (v10 → v11)
- `js/app.js` — loadSample() function, tab switching (v9 → v10)
- `samples/` — 40 new image files (10 per class)
