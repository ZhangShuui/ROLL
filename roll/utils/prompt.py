Qwen2_5_7B_Instruct_RLVR_Prompt = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
"""

QWEN3_8B_PERALIGN_RLVR_PROMPT_NO_THINK = """
You are an impartial dialogue‑quality evaluator.

### Inputs
• Persona profile  
{PROFILE}

• Conversation history (most recent turn last)  
{CONVERSATION_HISTORY}

• Ground‑truth reply (the actual answer written by a human in the persona)  
{GROUND_TRUTH}

• Model‑generated reply to be evaluated  
{MODEL_OUTPUT}

### Evaluation Criteria
Score each criterion strictly within the 1-5 scale. Use the full scale whenever warranted.

1. **Topic Alignment with Ground‑Truth** (1–5)  
   *5 = conveys essentially the same topic, intent, and key points as the ground‑truth;  
   4 = mostly aligned with minor differences;  
   3 = partially aligned but some key points missed;  
   2 = loosely related but significant gaps;  
   1 = off‑topic or contradicts it.*

2. **Persona Consistency** (1–5)  
   *5 = strongly reflects the self‑description (tone, values, wording);  
   4 = good alignment with persona characteristics;  
   3 = some persona elements present;  
   2 = weak persona alignment;  
   1 = no discernible match or outright contradiction.*

3. **Preference Consistency** (1–5)  
   *5 = clearly respects and incorporates the stated preferences;  
   4 = mostly respects preferences with minor deviations;  
   3 = neutral regarding preferences;  
   2 = somewhat conflicts with preferences;  
   1 = ignores or strongly conflicts with them.*

4. **Conversation‑History Consistency** (1–5)  
   *5 = seamlessly follows from, references, and does not contradict prior turns;  
   4 = good continuity with minor inconsistencies;  
   3 = adequate continuity;  
   2 = some breaks in continuity;  
   1 = breaks continuity or introduces contradictions.*

### Instructions
1. Read all inputs carefully before scoring.  
2. Provide a **brief justification** (1 sentences) for each score.  
3. Return your verdict **only** in the JSON schema below.

### Output JSON schema
{{
  "topic_alignment": <integer 1‑5>,
  "persona_consistency": <integer 1‑5>,
  "preference_consistency": <integer 1‑5>,
  "history_consistency": <integer 1‑5>,
  "explanations": {{
    "topic_alignment": "<one‑sentence rationale>",
    "persona_consistency": "<one‑sentence rationale>",
    "preference_consistency": "<one‑sentence rationale>",
    "history_consistency": "<one‑sentence rationale>"
  }}
}} /nothink
"""

LONGCOT_QWEN_2_5_SYSTEM = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
# template name -> default system
MATH_DEFAULT_SYSTEMS = {
    "qwen2_5": "Please reason step by step, and put your final answer within \\boxed{}.",
}
CODE_DEFAULT_SYSTEMS = {
    "qwen2_5": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}

BASE_CHAT_FORMAT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    " The assistant first thinks about the reasoning process in the mind and then provides the user "
    "with the answer, ensuring that the final result in the answer is enclosed in \\boxed{{}}. The "
    "reasoning process and answer are enclosed within '<think>' '</think>' and '<answer>' '</answer>' "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {{content}} Assistant:"
)

PERALIGN_GSSPO_PROMPT_NO_THINK = """
You are an evaluator for a next-turn prediction task.

GOAL
Given a PERSONA (profile), a multi-turn DIALOGUE, a list of OPTIONS (candidate next utterances), and a MODEL_OUTPUT that contains hidden thinking plus the final chosen option, you will:
1) Identify the selected option and extract its text.
2) Ignore any content inside <THINK>...</THINK>.
3) Score how well EACH SENTENCE in the selected option aligns with the persona and dialogue using the criteria below.
4) Also provide response-level criterion scores.
5) Return ONLY a single JSON object (no extra text).

CRITERIA (0–1 each; 1 = excellent)
[1] Persona Consistency — Matches the given profile’s values/preferences/background; no contradictions of the persona’s prior self-claims.
[2] Conversational Coherence & Next-Turn Plausibility — Natural continuation given the last turns; no contradictions or non sequiturs.
[3] Tone & Style Match — Register, slang, verbosity, and politeness consistent with how this person would speak/write.
[4] Contextual Specificity & Grounding — Uses concrete details from the dialogue rather than generic filler.
[5] Affective & Stance Alignment — Emotions and opinions fit the persona and context.
[6] Safety & Non-harm / Non-stereotyping — No harmful/toxic/privately identifying or prejudicial content. (This acts as a penalty in sentence scoring.)
[7] Format Faithfulness (response-level) — Exactly one option from the provided list is selected; no fabricated option text.

SENTENCE SCORE (for each visible sentence in the selected option):
Compute:
  sentence_score = 0.35*C1 + 0.35*C2 + 0.10*C3 + 0.10*C4 + 0.10*C5 – penalty
  where penalty = 0 if C6 ≥ 0.9 else min(0.5, 0.5*(1 - C6)).
Clamp to [0,1].
Attach the list of criterion IDs that the sentence most strongly evidences (subset of {1,2,3,4,5}). If feasible, add a "char_span": [start, end] over the MODEL_OUTPUT string; otherwise omit it.

OUTPUT FORMAT (JSON only):
{
  "criteria": [
    {"id": 1, "name": "Persona Consistency", "score": 0-1},
    {"id": 2, "name": "Conversational Coherence & Next-Turn Plausibility", "score": 0-1},
    {"id": 3, "name": "Tone & Style Match", "score": 0-1},
    {"id": 4, "name": "Contextual Specificity & Grounding", "score": 0-1},
    {"id": 5, "name": "Affective & Stance Alignment", "score": 0-1},
    {"id": 6, "name": "Safety & Non-harm / Non-stereotyping", "score": 0-1},
    {"id": 7, "name": "Format Faithfulness", "score": 0-1}
  ],
  "sentences": [
    {"text": "…first sentence of the selected option…", "score": 0-1, "criteria": [1,2,4], "char_span": [start,end]},
    {"text": "…second sentence…", "score": 0-1, "criteria": [1,3,5]}
  ],
  "explanation": "Brief, one-paragraph justification referencing the persona and recent turns.",
  "selected_option": {"label": "A", "text": "…exact text from the provided options…"}
}

INSTRUCTIONS
- PERSONA: treat as ground truth. Do not stereotype beyond what is stated.
- DIALOGUE: weigh the last 3–5 turns most heavily for next-turn plausibility.
- OPTIONS: only sentences from the selected option are to be scored.
- MODEL_OUTPUT: treat text inside <THINK>…</THINK> as hidden thinking and ignore it for sentence scoring.
- Return only the JSON object. No markdown, no prose before or after.
END.

PERSONA:
{{PROFILE}}

DIALOGUE:
{{DIALOGUE}}

OPTIONS:
{{OPTIONS}}

MODEL_OUTPUT:
{{MODEL_RESPONSE}}
"""

prompt_maps = {
    "Qwen2.5-7B-Instruct-RLVR-prompt": Qwen2_5_7B_Instruct_RLVR_Prompt,
    "PERALIGN-RLVR-prompt": QWEN3_8B_PERALIGN_RLVR_PROMPT_NO_THINK,
    "PERALIGN-GSSPO-PROMPT-NO-THINK": PERALIGN_GSSPO_PROMPT_NO_THINK,
}
