# Weights licence and data terms

This page states how the released model weights may be used and why. It is a
good-faith summary of upstream terms, **not legal advice**.

## Position

| Artefact | Licence |
|---|---|
| Code, documentation, configs | [MIT](https://github.com/sachncs/tinylm/blob/main/LICENSE) |
| Model weights (`tinylm-small-3.7M`, `tinylm-strict-12.9M`) | **Non-commercial research use only**, with attribution. Not MIT. |

The weights were trained on the BabyLM 2026 corpora. The Hugging Face dataset
cards label those corpora `license: mit`, but that label covers the
compilation, not the licences of the material inside it. At least three of the
six sources restrict commercial use, and one of them names language models
explicitly. Until each source is cleared, the weights carry the strictest
common condition: **no commercial use**.

## What you may do

- Use, evaluate, fine-tune and study the weights for non-commercial research and
  education.
- Publish results about them, with attribution (below).
- Redistribute them under these same terms, keeping this page and the
  attribution with them.

## What you may not do

- Use the weights, or models derived from them, in a commercial product or
  service, or to produce one.
- Relicense them under a more permissive licence.

Derived models plausibly inherit these limits (CHILDES is share-alike); how far
that reaches for model weights is legally unsettled, so assume it applies.

## Attribution

Cite this project ([CITATION.cff](https://github.com/sachncs/tinylm/blob/main/CITATION.cff)) and the BabyLM 2026
call for papers (Choshen et al., 2026, arXiv:2602.20092). Acknowledge the
sources below, and link to opensubtitles.org as OPUS asks.

## Sources and their terms

Training data is the BabyLM 2026 Strict-Small and Strict corpora. Both are
built from these files (`bnc_spoken`, `childes`, `gutenberg`, `open_subtitles`,
`simple_wiki`, `switchboard`).

| Source | Upstream terms found | Effect | Confidence |
|---|---|---|---|
| CHILDES (TalkBank) | [CC BY-NC-SA 3.0](https://talkbank.org/share/rules.html) "except where otherwise indicated"; the ground rules say this precludes use in commercial products, including large language models | Non-commercial, share-alike | High (official page) |
| BNC spoken | BNC1994 via the Oxford Text Archive: [no commercial exploitation](https://www.natcorp.ox.ac.uk/getting/index.xml?ID=licence) or publication of the corpus text; Spoken BNC2014 is a separate [non-commercial research licence](https://corpora.lancs.ac.uk/bnc2014/licence.php) from Cambridge University Press. Which edition BabyLM used is not stated | Non-commercial | Medium (edition unknown; OTA text not read directly) |
| OpenSubtitles | [OPUS](https://opus.nlpl.eu/OpenSubtitles/en&/v2018/OpenSubtitles): personal, educational and research use; no resale; link to opensubtitles.org. Underlying film and TV text is copyrighted and not cleared | Research only | Medium |
| Switchboard | Released by the LDC under its user agreement; commercial terms for the transcripts were not established. Some annotations are CC BY-NC-SA | Treat as non-commercial | Low |
| Simple English Wikipedia | CC BY-SA (Wikimedia's standard text licence; not re-checked here) | Attribution, share-alike | Medium |
| Project Gutenberg | Public-domain texts in the US; the Gutenberg name and licence text carry conditions (not re-checked here) | Permissive | Medium |

Checked 2026-10-14. The two highest-risk rows for a commercial release are
CHILDES and OpenSubtitles.

## Changing this position

The restriction can be lifted only by clearing every source, for example by
retraining on a corpus with commercially usable terms, or by obtaining written
permission from the rights holders. If you hold such permission, open an issue.

## Takedown

If you are a rights holder and object to a use, open an issue or use the
private channel in [SECURITY.md](https://github.com/sachncs/tinylm/blob/main/SECURITY.md); the weights will be withdrawn on a
valid request.

## No warranty

The weights are provided "as is", without warranty of any kind. They are
research artefacts that trail the best published BabyLM systems; do not use
them for decisions that matter.
