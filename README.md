\# Accident Report AI



Multimodal video-to-report pipeline for traffic accident detection, fact extraction, segmentation, and automatic report generation from dashcam and CCTV footage.



\## Overview

This project is designed for traffic accident analysis and automatic report generation from video.



Planned pipeline:

1\. Video is converted into frames.

2\. Upper branch performs vehicle detection, tracking, and metric extraction.

3\. Middle branch uses CLIP + ASFormer for frame-wise accident detection.

4\. Lower branch and LLM modules produce a draft, verify facts, reduce hallucinations, and generate a final structured accident report.



\## Current status

Currently implemented:

\- CLIP + ASFormer submodule

\- Two-stage training for the submodule

\- One trained checkpoint



\## Repository structure

\- `src/` - model code

\- `scripts/` - training and inference scripts

\- `configs/` - config files

\- `docs/` - notes and diagrams

\- `checkpoints/` - links or info about trained weights



\## Notes

Large datasets, videos, and heavy model weights are not stored directly in the repository.



## Architecture

The pipeline consists of three branches:
- **Upper branch** - detects and tracks vehicles and extracts motion-related facts
- **Middle branch** - uses CLIP + ASFormer to predict accident / non-accident over time
- **Lower branch** - uses LLM-based modules to generate, refine, and format the final accident report


