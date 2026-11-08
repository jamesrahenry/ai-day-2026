# 3-minute research presentation: speaker notes

*Written: 2026-11-08 09:56 UTC*

About 390 spoken words in the notes (timing cues and Q&A notes excluded), roughly 2.6 minutes at 150 words per minute. Slides: `talk.pdf` (print/present) and `talk.html`; sources in `slides/`.

## 1. cover

About 10 seconds. Hi, I’m James Henry. My question is simple to ask: do AI models built by different teams, from different text, end up organizing ideas the same way?

## 2. background

About 35 seconds. Three ideas. First, mechanistic interpretability: we can study a model by reading its internal activity, not just its answers. Second, each idea a model learns, like negation or credibility, is stored as a direction inside it. Third, the Platonic Representation Hypothesis: models trained on different data may converge on the same picture of the world.

## 3. problem

About 35 seconds. The problem: every model stores ideas in its own private coordinates, so you can’t simply lay one model’s directions over another’s. But the relationships between ideas could still match. Is credibility close to formality and far from sarcasm in every model? It matters because, if the map is shared, what we learn inside one model can be checked in others. And it tells us whether models learn something about the world, or just quirks of their training.

## 4. method

About 45 seconds. Four steps. First, I wrote pairs of passages: the same text twice, once with an idea and once without. Seventeen ideas, 250 pairs each. Second, I ran them through 35 open models from 10 families; the difference between the two versions gives each idea’s direction. Third, inside each model, I measured how close every idea is to every other. Those angles can be compared across models, even though coordinates can’t. Fourth, I tried to make it fail: hide a whole model family, and see whether the other families’ angles alone can name its ideas.

## 5. results

About 35 seconds. The result: with one family hidden, the other families’ map named its ideas correctly 97 percent of the time in the middle layers. With the labels shuffled, or with random directions, it drops to 6 percent, which is chance. So the shared map is real, not a pattern any set of directions would show. One example: in 29 of 35 models, the idea most closely tied to credibility is formality.

## 6. conclusion

About 30 seconds. So: language models built by different teams, from different text, organize ideas the same way. That supports the Platonic Representation Hypothesis, measured idea by idea. And because the map is shared, findings from inside one model can be tested on others. Which leaves a question: is this the structure of language itself, or of the world it describes? Come and find me at the poster. Thank you.
