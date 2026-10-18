<!--
Draft (not yet run). One call = one (emotion, topic) cell = {n_stories} stories; paper: 100 topics x 12 stories = 1,200 per emotion.
Adapted from the "Emotional stories prompt" in Sofroniew et al. 2026 (Appendix, Dataset generation); wording paraphrased.
Render with str.replace (the file contains JSON braces), keys: {emotion} {topic} {topic_id} {n_stories} {gloss_line}.
{gloss_line} = "" for a strict replication, else "(Here '{emotion}' means: <gloss>.)" from emotions.json glosses_optional.
Schema: schema_emotion_stories.json (strict). Everything below the line is sent as the user input; there is no system prompt.
Lines marked [ours] are additions beyond the paper's prompt; delete them for a strict replication.
-->
Write {n_stories} different short stories based on the premise below.

Topic: {topic}

Each story follows a character who is feeling {emotion}. {gloss_line}

Requirements:
- Each story is a single paragraph and a fresh start, with no continuity between stories (different characters, settings and turns of events within the premise).
- Make the stories diverse and do not reuse the same turns of phrase across them.
- Mix first-person and third-person narration across the set (roughly half each), and record which one each story uses.
- Length: about 100-180 words per story. [ours: the paper says only "roughly one paragraph"; the length band keeps at least ~70 tokens after the 50-token cutoff used when averaging activations]
- Vary character names, ages and backgrounds; do not reuse a character name within the set. [ours]
- Write plain prose only: no titles, numbering, headings, quotation of these instructions, or commentary before or after a story. [ours]

IMPORTANT: Never use the word "{emotion}", any form of it, or any direct synonym of it anywhere in the stories. Convey the emotion only indirectly, through:
- what the character does and how they behave;
- physical sensations and body language;
- dialogue and tone of voice;
- the character's thoughts and internal reactions;
- the situation and the surrounding environment.
A reader should clearly recognize the emotion from these cues even though it is never named.

Return JSON matching the schema: "emotion" and "topic_id" echo the values given here ("{emotion}", "{topic_id}"), and "stories" holds exactly {n_stories} items.
