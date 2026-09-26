# Notices

## Clinical Use Disclaimer

GFME is a research tool. It has **not been validated for clinical decision-making** and must not be used as the sole or primary basis for medical diagnosis, treatment decisions, or clinical variant interpretation.

## Upstream Foundation Model Terms

GFME is licensed under Apache 2.0 (see [LICENSE](LICENSE)). However, this repository contains **pre-computed scores and trained models derived from third-party foundation models** that carry their own terms. By using these files, you agree to comply with the applicable upstream terms.

---

### AlphaGenome — Google DeepMind

The trained CatBoost models and pre-computed AlphaGenome features in this repository are derived from AlphaGenome outputs and are governed by the AlphaGenome terms.

**Primary terms (HuggingFace model weights release, January 2026):**
- [AlphaGenome Model Parameters Terms of Use](https://deepmind.google.com/science/alphagenome/model-terms)
- Training on AlphaGenome outputs is permitted; resulting models must be governed by the same terms
- Non-commercial use only by non-commercial organizations
- No clinical use

**API Output Terms (June 2025):**
- [AlphaGenome Output Terms of Use](https://storage.googleapis.com/alphagenome/terms/AlphaGenome-Output-Terms-of-Use.pdf)
- Item 1: Non-commercial only, with carve-out for scientific publication and open source release
- Item 3: Requires conspicuous notice (see [ALPHAGENOME_TERMS.md](ALPHAGENOME_TERMS.md))

**Affected files:** `app/models/*.cbm`, `data/scores/*ag*.npz`, `data/reclassification/*ag*.npz`

**Reference:** Avsec et al. (2026). *Nature* 649, 1206-1218. https://doi.org/10.1038/s41586-025-10014-0

**Contact:** alphagenome@google.com

---

### GPN-MSA — Song Lab, UC Berkeley

- **License:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- Non-commercial use only. No derivatives.
- **Affected files:** `data/scores/*gpnmsa*.npz`, `data/reclassification/*gpnmsa*.npz`
- **Reference:** Benegas et al. (2023). https://doi.org/10.1101/2023.10.10.561776
- **Contact:** https://github.com/songlab-cal/gpn

---

### Evo2 — Arc Institute / NVIDIA

- **License:** [Apache 2.0](https://github.com/ArcInstitute/evo2/blob/main/LICENSE) — no additional restrictions
- **Affected files:** `data/scores/*evo2*.npz`, `data/reclassification/*evo2*.npz`
- **Reference:** Nguyen et al. (2025)
- **Contact:** https://arcinstitute.org/

---

## Summary

| Component | License | Commercial use |
|-----------|---------|---------------|
| GFME code | Apache 2.0 | Yes |
| Trained models (`.cbm`) | [AG Model Parameters ToU](https://deepmind.google.com/science/alphagenome/model-terms) | No |
| AlphaGenome features | [AG Model Parameters ToU](https://deepmind.google.com/science/alphagenome/model-terms) | No |
| GPN-MSA scores | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | No |
| Evo2 scores | Apache 2.0 | Yes |
