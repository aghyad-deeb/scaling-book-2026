# How to Scale Your Model: The 2026 Update

An independent supplement to [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
(Austin et al., Google DeepMind, 2025), covering the frontier open-weight models of 2025 and 2026
and the hardware they actually run on. The derivations lead with the GPUs those models were trained and served on (H800, H100/H200, GB200 NVL72) and keep TPUs as the comparison. Not written or reviewed by the book's authors.

Live site: https://aghyad-deeb.github.io/scaling-book-2026/

## Sections

| Part | File | Topic |
| --- | --- | --- |
| Outline | `index.md` | what changed since LLaMA 3, the hardware table, where the original needs a footnote |
| 13 | `moe.md` | Mixture of Experts: parameter and FLOP accounting, batch-size and expert-parallelism rooflines |
| 14 | `attention.md` | the KV cache: MLA, sliding windows, sparse attention, linear hybrids, token compression |
| 15 | `training-2026.md` | fp8 and fp4 training, Muon, multi-token prediction, the RL loop as a systems problem |
| 16 | `applied-frontier.md` | serving DeepSeek-V3 on real hardware and planning a V4-class training run |

## Layout

The site reuses the book's own Jekyll machinery ([al-folio](https://github.com/alshedivat/al-folio)
with the Distill template), so the chapter sources use the same dialect as the book:
`{% details %}` for hidden answers, `<d-footnote>`, `<d-cite key=...>`, `{% include figure.liquid %}`
and `<p markdown=1 class="takeaway">`.

```
index.md, moe.md, attention.md, training-2026.md, applied-frontier.md   chapter sources
assets/bibliography/update.bib                                            references
assets/img/*.svg                                                          figures (generated)
scripts/numbers.py                                                        parameter counts, FLOPs, MoE rooflines
scripts/attention_numbers.py                                              KV bytes, MLA intensity, crossovers
scripts/training_numbers.py                                               fp8/fp4, Muon, MTP, RL, DeepSeek serving
scripts/figures.py                                                        regenerates assets/img/*.svg
_layouts, _includes, _sass, _plugins, assets/{css,js}                     al-folio + Distill, copied from the book
```

## Build

```bash
bundle install
bundle exec jekyll serve      # http://localhost:4000/scaling-book-2026/
python3 scripts/figures.py    # regenerate the SVG figures
python3 scripts/numbers.py    # recompute every number quoted in the text
```

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds the site and force-pushes it to
the `gh-pages` branch that GitHub Pages serves.

## Provenance

Every model number comes from the model's `config.json` or technical report, and every hardware
number from the vendor's own spec page; the scripts recompute all of them (parameter totals within
2% of the labs' reported figures, active counts within a few percent). The text was drafted with
Claude from those sources and then reviewed by independent technical and style passes; the review
findings and their fixes are recorded in the commit history.

## License

The template code (`_layouts`, `_includes`, `_sass`, `_plugins`, `assets/css`, `assets/js`) is MIT,
see `LICENSE`. The chapter text and figures are © 2026 Aghyad Deeb.
