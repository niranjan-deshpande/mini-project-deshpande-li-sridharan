<!--
Draft (not yet run). Neutral Human/AI dialogues used only for denoising: PCA on their activations, keep the top PCs that
explain 50% of variance, and project them out of every emotion vector. Paraphrased from the paper's "Neutral dialogues prompt".
The paper does not state how many neutral dialogues it used; proposal: the same 100 topics x 12 = 1,200 dialogues.
Keys: {topic} {topic_id} {n_dialogues}. Schema: schema_neutral_dialogues.json (strict).
Rendering for Qwen (ours): the paper rewrote Person/AI to Claude's Human/Assistant format; we instead render each dialogue
with Qwen's chat template (optional system message -> system, Person -> user, AI -> assistant, no <think> content),
and use all tokens of the rendered transcript for the PCA.
-->
Write {n_dialogues} different dialogues based on the topic below.

Topic: {topic}

Each dialogue is between a Person (a human) and an AI (an AI assistant). The Person asks the AI a question or asks for help with a task, and the AI gives a helpful answer. The Person always speaks first.

Make the set diverse:
- Some dialogues, but not all, begin with system instructions for the AI (put them in "system"; leave it as an empty string otherwise). Refer to the AI as "you" or "the assistant" in them.
- Some are about code or programming tasks.
- Some are factual questions (science, history, math, geography).
- Some are work tasks (writing, analysis, summarization).
- Some are practical how-to questions.
- Some are creative but neutral tasks (brainstorming names, making lists).
- A single question-and-answer is fine where natural, but at least some dialogues have several exchanges; use 1-6 exchanges.
- The topic only seeds the subject matter; keep the dialogue itself free of the topic's emotional angle.

CRITICAL: the dialogues must be completely neutral and free of emotion.
- No emotional content at all: not explicit, implied, or subtle.
- The Person expresses no feelings (no frustration, excitement, gratitude, worry, and so on).
- The AI expresses no feelings (no enthusiasm, concern, satisfaction, and so on).
- System instructions, if present, do not mention emotions or use emotionally charged language.
- Avoid emotionally charged subjects entirely, and use matter-of-fact language throughout.
- No pleasantries (no "Happy to help", "Great question!", thanks, apologies).
- Focus only on exchanging information and completing the task.

Return JSON matching the schema; "topic_id" echoes "{topic_id}" and "dialogues" holds exactly {n_dialogues} items.
