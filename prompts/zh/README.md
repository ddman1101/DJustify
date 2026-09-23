# Original (Chinese) prompts

These are the prompts exactly as used to produce the listening-test stimuli
(Test1 transitions on 2026-09-01, Test2 story sets on 2026-09-03), dumped from
the source constants without editing. The code in `djustify/` and `story/`
ships English translations of the same prompts so the pipeline can be read
without Chinese; the translation keeps the structure and every rule, but a
run with the English prompts is not guaranteed to reproduce the exact plans
the Chinese prompts produced.

| File | Used by |
|---|---|
| 01 LEGEND | prefix of every transition system prompt (song-card field legend) |
| 02 GUIDE_ADV | transition planner system prompt (Test1, all three variants) = 01 + this + 03 |
| 03 RENDER_CONTRACT | footprint rules and renderer physics, shared by 02 and 04 |
| 04 GUIDE | junction planner system prompt inside a story set = this + 03 |
| 05 FX_CARD | technique card, sent in the user message |
| 06 BLEND_FEWSHOT | two worked blendecho examples appended to 02 |
| 07 SEL_SYS | next-track selection (free-select mode) |
| 08 S1_TASK / 09 S3_TASK | story-set junction: exit proposals, final plan |
| 10 MEAN / 11 FAMS | checker gate meanings and the four families (text returned to the planner on failure) |
| 12 lyric hook system | story-set song selection (storytelling + lyric-hook criteria) |
