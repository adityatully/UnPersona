# data/

## graded_prompts.json
`{trait: [8 system prompts]}` for evil, sycophantic, hallucinating.

- Source: paper appendix C.3 "System prompts for inducing traits" (arXiv 2507.21509, HTML version),
  used for section 3.3 / Figure 4 (monitoring prompt-induced persona shifts). Not in the code repo.
  Text copied verbatim (whitespace normalized; curly apostrophes kept as in the paper).
- **Index = level of trait encouragement**: index 0 = least (paper's prompt #8, "You are a helpful
  assistant. ..."), index 7 = most (paper's prompt #1, e.g. "You are an evil assistant. ...").
  The paper numbers them the other way round (1 = most trait-promoting).
- The paper generated these with Claude 4.0 Sonnet to "smoothly interpolate" between trait-suppressing
  and trait-promoting; the levels are ordinal labels, not a measured scale.
- Overlap with the extraction data: evil index 7 is the same text as the first evil extraction
  instruction ("You are an evil assistant." + instruction 0 pos). The vectors were built partly from
  answers under that prompt.
