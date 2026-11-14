# Kaizen style guide

How a kaizen note is written, laid out, and illustrated. The figure rules come from the
`codebase-explainer` skill (`references/style-and-diagrams.md`), adapted for PNG output.

## 1. Files

```
learnings/<yyyymmdd>_<slug>/
  <yyyy-mm-dd>-<slug>.markdown   # the post (Jekyll)
  <slug>.py                      # the full script; every number in the post comes from it
  diagrams/<figN_name>.py        # one script per figure; helpers start with "_"
  resources/<figN_name>.svg|png  # generated; the post links the PNG
  linkedin-post.md               # optional companion
```

Front matter:

```
---
layout: post
title:  "<Series> - <topic>"
date:   <yyyy-mm-dd> 00:00:00 +0530
categories: <space separated>
mathjax: true
---
```

Math: `$$ ... $$` blocks. Code: `{% highlight python %} ... {% endhighlight %}`.
Images: `https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/<folder>/resources/<name>.png`.
References: a `**Reference:**` list with link references at the bottom.

## 2. Voice

- First person and conversational. Short paragraphs.
- Open with the problem, a question, or a concrete fact, in one sentence. Then say what the note is about
  ("This note is about ..."). Examples: "Arithmetic alone does not run a neural network." / "Can a weight
  edit make a small language model better at calling tools?"
- Do not open with "Lately" or "I have been". Do not open two posts in a row with the same words.
- A series: open each part with its own problem, not with the series name.
- LinkedIn companion: the first line states the result or the surprise, before the "see more" cut. One
  idea, few numbers, every term explained. End with the series part and 2-3 hashtags. The first comment
  links to the published post on thomaspaul.dev, not to GitHub.
- Lead-ins: "Before diving in, let's look at ...", "Let's say ...", "Let's look at the code.", "It means that ...".
- Define a term in one sentence, then show a small concrete example with real numbers.
- A question as a section opener is fine ("Why column first?").
- Close with "So, in short: ..." in one or two sentences.
- No hype words (robust, seamless, powerful, leverage). Say the fact.

## 3. Code

- The post shows a short check that prints `"<what>. Is pass?"`. It usually prints `True`; a deliberate
  counter-example may print `False`, with a comment that says so (as in the Megatron note).
- The full script sits next to the post and is linked as "available [here](...)".
- Run scripts with uv: `uv run --with numpy python <script>.py`.
- Every number in the text comes from running the script. Do not type numbers by hand.

## 4. Figures

Build all figures of a post:

```
uv run --with cairosvg python tools/build_figures.py learnings/<folder>
```

It runs each `diagrams/*.py` (with `tools/diagrams/svglib.py` on the path), writes the SVG, and
converts it to a PNG at 2x. The Inter font must be installed for the PNG to match the SVG.

Rules (from style-and-diagrams.md; full guide: `tools/diagrams/instructions-svg.md`):

- **One Python script per figure**, built on `svglib.py`. No Mermaid, no hand-drawn SVG.
- **Canvas width 1100**: `Diagram(width=1100)`.
- **No text smaller than 11.** Mono data 12-13. No rotated text.
- **At most 3 boxes side by side, at most 3 panels.** Split a bigger figure into two.
- White background, inline attributes only (no CSS, gradients, shadows or images).
- Sans (Inter) for prose; mono for every number, size, identifier and path.
- **One colour = one meaning**, 3-5 accents, all in the legend; gray for the rest.
  Arrow colour = the role of what flows. Dashed arrows for secondary paths. Arrows do not cross boxes or labels.
- Numbered panels top to bottom; each panel title is a question, its subtitle is the answer.
- Header: the title says what the figure explains; the subtitle starts with the series name and gives the scope.
- Footnotes say what is simplified.
- Real values from the script. Sources in the diagram script docstring.
- In the post: alt text describes the figure panel by panel, and the next line is the caption in
  italics, starting with "Figure: ".
- Look at every PNG before publishing: nothing overlaps, nothing is clipped.
- Runs of spaces collapse in the PNG. Draw aligned columns as separate text: use `tools/diagrams/figkit.py`
  (`row()` for up to 3 boxes, `table()` for tables, `code()` with `\t` columns for code and config cards).

## 5. Series

A long topic becomes a series. Each title starts with the series name ("Parallelism Basics - ...",
"Quantization Basics - ..."), and each figure subtitle starts with it too.

## 6. LinkedIn companion (optional)

`linkedin-post.md`: 5-8 short paragraphs in the same voice, the figures as a carousel in order,
and the kaizen folder link in the first comment.
