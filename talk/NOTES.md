# Lightning talk: speaker notes

*Written: 2026-11-05 20:53 UTC*

About 519 words in the notes, roughly 3.5 minutes at 150 words per minute. Slides: `talk.pdf` (print/present) and `talk.html`; sources in `slides/`.

## 1. cover

About 10 seconds. Hi, I’m James Henry. My question: when different language models learn from different text, do they end up organizing their concepts in the same way? Three minutes, and the poster has the rest.

## 2. vectors

About 35 seconds. Start with a simple idea: a model stores a concept like credibility as a direction in its activation space. To find it, I write pairs of passages on the same topic, one with credibility markers and one without. 17 concepts, about 250 pairs each, written by 14 different LLMs. Run both versions through the model, subtract the mean activation of one from the mean of the other, and you get that concept’s vector, at every layer. Project any prompt onto it and you can read off how much of the concept is present.

## 3. hypothesis

About 35 seconds. Now the hypothesis. The Platonic Representation Hypothesis, from Huh and colleagues, says models trained on different data, text or images, with different architectures, are all learning from shadows of one world, and they converge toward a shared model of it. The important detail: what converges is relations, not coordinates. Each model’s coordinate system is its own. So the prediction is that models that never shared a weight should still agree on which concepts are similar to which.

## 4. test

About 40 seconds. So I asked 35 models the same question at every layer: what is credibility aligned with? Each line is one concept’s cosine to credibility across depth, and the bands show the spread across models. Formality is closest, in 29 of 35 models at the final layer. Deception and sarcasm are nearly as strong but point the opposite way; the sign only reflects which pole we labeled positive. And the bands tighten from mid-depth on: the models are agreeing. If asked: this is a cosine, so plus and minus 0.7 are the same axis, not the same direction.

## 5. result

About 35 seconds. Here is the test that can fail. Hold out an entire model family. Describe each of its concept vectors only by its angles to the other concepts, and match that against the average angle profile from all the other families. At mid-depth it names the concept correctly 97 percent of the time. Shuffle the labels, or use random vectors instead, and you get 6 percent, which is chance.

## 6. meaning

About 35 seconds. What might this mean? Models built by different teams, on different text, agree on how their concepts sit relative to one another. That is what the Platonic Representation Hypothesis predicts, and because the map is measurable, we can test it concept by concept and model by model. Which is part of what interpretability is for. The open question: is this the structure of language itself, or of the world it describes? Come find the poster; the code and reading list are on GitHub.

If asked about scale or theory: a recent paper (Balestriero and LeCun, 2026, arXiv:2610.39302) argues that alignment depends on how training views are chosen, not on scale alone. It is vision-language and trains new models; this work is text-only across existing language models, so it neither confirms nor contradicts it.
