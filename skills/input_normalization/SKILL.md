---
name: input_normalization
description: >
  Normalize various comic input formats (PDF/CBR/CBZ/ZIP/EPUB/images) 
  into an ordered sequence of page images for downstream processing (OCR / detection).
categories: [preprocessing, file_handling]
version: "0.1.0"
---

# Input Normalization Skill

## Overview
This skill handles the preprocessing of input comic files.  
It supports common formats including:
- **PDF** (multi-page)
- **CBR / CBZ** (compressed image sequences)
- **ZIP** archives containing images
- **Standalone images** (PNG/JPG)
- **EPUB** (optional extension)

The goal is to produce a **sorted list of page images** with consistent naming and indexing, which downstream skills can consume.

---

## When to Use This Skill
Use this skill when:
- The user uploads a file that is not already a simple image sequence.
- The agent needs to extract image pages in order for OCR and panel detection.
- The input format is ambiguous and requires normalization.

---

## Input Schema
| Field      | Type     | Description                                   |
|------------|----------|-----------------------------------------------|
| file_bytes | bytes    | Raw bytes of the uploaded comic file          |
| file_type  | string   | File extension/type (e.g., pdf, cbr, zip)     |

---

## Output
Returns a list of objects:
- `page_index`: the index of the page (0-based)
- `image_bytes`: image bytes of the page
- `filename`: the name or generated identifier for the page

---

## Steps to Execute
1. Identify file type from metadata or extension.
2. If PDF: render each page to an image.
3. If CBR/CBZ/ZIP: extract image files and sort.
4. If single image: wrap it as page 0.
5. Return a standardized list of images for next steps.

---

## Notes and Edge Cases
- Ensure ordering is correct even if filenames are not numerically sorted.
- Avoid loading entire content into memory when possible.
- Support unicode file names and unusual extensions.
- This skill **does not** perform OCR — it only normalizes input formats.