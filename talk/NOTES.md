# Lightning talk: speaker notes

*Written: 2026-11-05 21:14 UTC*

About 486 words in the notes, roughly 3.2 minutes at 150 words per minute. Slides: `talk.pdf` (print/present) and `talk.html`; sources in `slides/`.

## 1. cover

About 10 seconds. Hi, I’m James Henry. My question: when different language models learn from different text, do they end up organizing their concepts in the same way? Three minutes, and the poster has the rest.

## 2. interp

About 40 seconds. Most of what we know about language models comes from what they say. Mechanistic interpretability asks a different question: what is going on inside? Instead of judging a model by its outputs, we read its internal activations, layer by layer, to see what it represents and how. That needs the model’s weights, so it is only possible with open models, which is why this work uses them.

## 3. lrh

About 35 seconds. One idea makes that practical: the Linear Representation Hypothesis. It says a model stores each concept it learns, negation, say, as a direction in its activation space. Prompts that express the concept land further along that direction, so measuring along it tells you how much of the concept is present. We read the activations at the final token of a passage, because by then the model has attended to everything before it.

## 4. vectors

About 40 seconds. Finding the direction is surprisingly simple. Write two versions of the same passage, one with the concept and one without: here, credibility markers against a flat assertion. Run both through the model and subtract the average activation of one from the other. That difference is the concept’s vector, and you can compute it at every layer. I did this for 17 concepts, about 250 pairs each, written by 14 different language models. This family of methods is called representation engineering.

## 5. hypothesis

About 35 seconds. Now the hypothesis. The Platonic Representation Hypothesis, from Huh and colleagues, says models trained on different data, text or images, with different architectures, are all learning from shadows of one world, and they converge toward a shared model of it. The important detail: what converges is relations, not coordinates. Each model’s coordinate system is its own. So the prediction is that models that never shared a weight should still agree on which concepts are similar to which.

## 6. meaning

About 35 seconds. So here is the question I brought to the poster. If the Platonic Representation Hypothesis is right, models built by different teams on different text should agree on how their concepts relate to one another. I tested that across 35 open models, concept by concept and layer by layer. I would like to show you what I found, and hear what you make of it. And if they do agree, is that the structure of language itself, or of the world it describes? Please come and find the poster.

If asked about scale or theory: a recent paper (Balestriero and LeCun, 2026, arXiv:2610.39302) argues that alignment depends on how training views are chosen, not on scale alone. It is vision-language and trains new models; this work is text-only across existing language models, so it neither confirms nor contradicts it.
