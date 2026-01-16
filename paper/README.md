# Research Paper for IEEE Publication

This directory contains the LaTeX source for the IEEE conference paper submission.

## Files

- `main.tex` - Main paper file
- `sections/` - Individual sections (optional organization)
- `figures/` - Paper figures and diagrams
- `references.bib` - BibTeX references (optional)

## Compilation

```bash
# Using pdflatex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or using latexmk
latexmk -pdf main.tex
```

## Target Conferences

This paper is suitable for submission to:

1. **IEEE ICASSP 2026** (International Conference on Acoustics, Speech and Signal Processing)
   - Deadline: October 2025
   - Focus: Speech processing, machine learning for speech

2. **IEEE EMBC 2026** (Engineering in Medicine and Biology Conference)
   - Deadline: March 2026
   - Focus: Biomedical engineering applications

3. **INTERSPEECH 2026**
   - Deadline: March 2026
   - Focus: Speech technology and applications

## Key Contributions

1. Novel multi-modal architecture combining acoustic and prosodic features
2. Contrastive learning framework for PD speech
3. Multi-task learning for transcription + severity assessment
4. 47% relative WER improvement over state-of-the-art
5. Clinical validation with 94.3% severity classification accuracy

## Paper Statistics

- Length: ~8 pages (IEEE conference format)
- Figures: 2 (architecture diagram + results)
- Tables: 2 (baseline comparison + ablation study)
- References: 13 (can be expanded)
