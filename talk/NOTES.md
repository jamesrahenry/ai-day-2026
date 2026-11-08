# 3-minute research presentation: speaker notes

*Written: 2026-11-08 14:32 UTC*

About 401 spoken words in the notes (timing cues and Q&A notes excluded), roughly 2.7 minutes at 150 words per minute. Slides: `talk.pdf` (print/present) and `talk.html`; sources in `slides/`.

## 1. cover

About 10 seconds. Hi, I’m James Henry. My question is simple to ask: do AI models built by different teams, from different text, end up organizing concepts the same way?

## 2. background

About 35 seconds. Some background first. First, mechanistic interpretability: we can study a model by reading its internal activity, not just its answers. Second, each concept a model learns, like negation or credibility, is stored as a direction inside it: the Linear Representation Hypothesis. Third, the Platonic Representation Hypothesis: models trained on different data may converge on the same representation of the world.

## 3. problem

About 35 seconds. The problem: if the linear and Platonic hypotheses hold, then models trained separately should agree on how concepts relate to each other, even though each stores them in its own private coordinates. Is credibility close to formality and opposed to sarcasm in every model? It matters because, if the map is shared, what we learn inside one model can be checked in others. And it tells us whether models learn something about the world, or just quirks of their training.

## 4. method

About 45 seconds. Four steps. First, I wrote pairs of passages: the same text twice, once with a concept and once without. Seventeen concepts, 250 pairs each. Second, I ran them through 35 open models from 10 families; the difference between the two versions gives each concept’s direction, its concept vector. Third, inside each model, I measured how close every concept is to every other. Those angles can be compared across models, even though coordinates can’t. Fourth, I tried to make it fail: hide a whole model family, and see whether the other families’ angles alone can identify its concepts.

## 5. results

About 35 seconds. The result: with one family hidden, the other families’ map identified its concepts correctly 97 percent of the time in the middle layers. With the labels shuffled, or with random directions, it drops to 6 percent, which is chance. So the shared map is real, not a pattern any set of directions would show. One example: in 29 of 35 models, the concept most closely tied to credibility is formality.

If asked whether the pairs really capture credibility: the test does not depend on what the label means. Whatever a set of pairs picks out, it picks out the same thing, in the same relation to the other 16, in every model. That is the finding; what to call each concept is a separate question.

## 6. conclusion

About 30 seconds. So: language models built by different teams, from different text, organize concepts the same way. That supports the Platonic Representation Hypothesis, measured concept by concept. And because the map is shared, findings from inside one model can be tested on others. Which leaves a question: is this the structure of language itself, or of the world it describes? Come and find me at the poster. Thank you.

If asked about scale or theory: a recent paper (Balestriero and LeCun, 2026, arXiv:2610.39302) argues that alignment depends on how training views are chosen, not on scale alone. It is vision-language and trains new models; this work is text-only across existing language models, so it neither confirms nor contradicts it.
