import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import queue

try:
    from openai import OpenAI
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Model selection will be disabled.", file=sys.stderr)

# Threshold for similarity filtering
MIN_SIM_THRESHOLD = 0.03
MAX_SIM_THRESHOLD = 0.97
HI_SIM_THRESHOLD = 0.8


# Thread-safe progress tracking
class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()

    def update(self, success: bool = True):
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1

    def get_stats(self) -> Tuple[int, int, int]:
        with self.lock:
            return self.completed, self.failed, self.total


@dataclass
class QuestionTask:
    qid: str
    actual_idx: int
    num_choices: int


def load_similarity_data(file_path: str) -> List[Dict]:
    """Load similarity data from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_multiple_banks(bank_paths: List[str]) -> Dict[str, List[Dict]]:
    """Load multiple distractor banks"""
    banks = {}
    for path in bank_paths:
        bank_name = os.path.basename(path).replace(".json", "").replace("multi_choice_similarity_", "")
        print(f"Loading bank: {bank_name} from {path}")
        banks[bank_name] = load_similarity_data(path)
        print(f"Loaded {len(banks[bank_name])} items from {bank_name}")
    return banks


def extract_distractors_from_item(item: Dict, bank_name: str) -> List[Tuple[str, float, str, str]]:
    """
    Extract all valid distractors from a single item

    Returns:
        List of tuples: (distractor_type, similarity_score, distractor_text, bank_name)
    """
    distractors = []

    # Define all distractor types
    distractor_types = [
        "style_violation_distractor",
        "topic_violation_distractor",
        "richness_violation_distractor",
        "free_violation_distractor",
        "profile_violation_w_distractor",
        "conversation_violation_w_distractor",
        "both_violation_w_distractor",
        "profile_violation_w/o_distractor",
        "conversation_violation_w/o_distractor",
        "both_violation_w/o_distractor",
    ]

    for distractor_type in distractor_types:
        similarity_key = f"{distractor_type}_similarity"

        if (
            distractor_type in item
            and similarity_key in item
            and item[distractor_type] is not None
            and item[similarity_key] is not None
        ):
            distractor_text = item[distractor_type]
            similarity_score = item[similarity_key]

            # Filter out failed distractors and apply similarity thresholds
            if (
                not distractor_text.startswith("Failed_")
                and MIN_SIM_THRESHOLD <= similarity_score <= MAX_SIM_THRESHOLD
            ):
                distractors.append((distractor_type, similarity_score, distractor_text, bank_name))

    return distractors


def deduplicate_distractors(distractors: List[Tuple[str, float, str, str]]) -> List[Tuple[str, float, str, str]]:
    """
    Remove duplicate distractors based on text content
    Keep the one with highest similarity score for each unique text

    Args:
        distractors: List of (distractor_type, similarity_score, distractor_text, bank_name)

    Returns:
        Deduplicated list of distractors
    """
    # Group by distractor text (case-insensitive and strip whitespace)
    text_to_best = {}

    for dtype, sim, text, bank in distractors:
        # Normalize text for comparison
        normalized_text = text.strip().lower()

        if normalized_text not in text_to_best or sim > text_to_best[normalized_text][1]:
            text_to_best[normalized_text] = (dtype, sim, text, bank)

    # Return deduplicated list
    return list(text_to_best.values())


def extract_all_distractors(banks: Dict[str, List[Dict]], qid: str) -> List[Tuple[str, float, str, str]]:
    """
    Extract all valid distractors for a given qid from all banks

    Returns:
        List of tuples: (distractor_type, similarity_score, distractor_text, bank_name)
    """
    all_distractors = []

    for bank_name, bank_data in banks.items():
        # Find the item with matching qid
        for item in bank_data:
            if item.get("qid") == qid:
                distractors = extract_distractors_from_item(item, bank_name)
                all_distractors.extend(distractors)
                break

    # Deduplicate distractors
    all_distractors = deduplicate_distractors(all_distractors)

    return all_distractors


def model_select_distractors(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    candidate_distractors: List[Tuple[str, float, str, str]],
    num_to_select: int,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> List[Tuple[str, float, str, str]]:
    """
    Use a language model to select the most confusing distractors for personality-based questions
    Enhanced version based on build_choice_training_data.py

    Args:
        correct_answer: The correct response
        profile: The person's profile information
        conversation_history: The conversation context
        candidate_distractors: List of candidate distractors (distractor_type, similarity_score, distractor_text, bank_name)
        num_to_select: Number of distractors to select
        model_name: Model to use for selection
        api_key: OpenAI API key

    Returns:
        Selected distractors
    """
    if len(candidate_distractors) <= num_to_select:
        return candidate_distractors

    if not api_key or not OPENAI_AVAILABLE:
        print("Warning: No API key provided or OpenAI not available, falling back to random selection")
        return random.sample(candidate_distractors, num_to_select)

    try:
        client = OpenAI(api_key=api_key, base_url="https://yunwu.zeabur.app/v1")

        # Prepare candidates for the model
        candidates_text = []
        for i, (dtype, sim, text, bank) in enumerate(candidate_distractors):
            candidates_text.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        selection_prompt = f"""You are tasked with selecting the most confusing and challenging distractors for a personality-based multiple-choice question.

Context:
{context_description}

Correct Answer: {correct_answer}

Candidate Distractors:
{chr(10).join(candidates_text)}

Please select exactly {num_to_select} distractors that would be most confusing for someone trying to identify how this specific person would respond. Choose distractors that:

1. Are plausible responses that someone with different personality traits might give
2. Are similar enough to the correct answer to be confusing but represent different personality aspects
3. Test understanding of the person's specific characteristics, values, and communication style
4. Cover different potential misconceptions about this person's likely response
5. Represent a good mix of difficulty levels

Consider factors like:
- Personality traits and values shown in the profile
- Communication style and preferences
- Consistency with previous conversation patterns
- Emotional intelligence and social awareness levels
- The similarity scores and distractor types provided

Please respond with only the numbers of your selected distractors, separated by commas (e.g., "1,3,7").
"""
        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an expert at creating challenging personality-based multiple-choice questions that test understanding of individual differences and communication patterns.",
        #         },
        #         {"role": "user", "content": selection_prompt},
        #     ],
        #     max_tokens=50,
        #     temperature=0.3,
        # )

        # Parse the response
        # selection_text = response.choices[0].message.content.strip()
        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating challenging personality-based multiple-choice questions that test understanding of individual differences and communication patterns.",
                        },
                        {"role": "user", "content": selection_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                selection_text = response.output_text.strip()
                selected_indices = [int(x.strip()) - 1 for x in selection_text.split(",")]
            except Exception as e:
                print("Warning: API Exceptions, retrying...")
                if i < 2:
                    # time.sleep(2**i)  # Exponential backoff
                    model_name = "gpt-5-mini-2025-08-07"
                    continue
                else:
                    print("Max retries reached, falling back to random selection")
                    return random.sample(candidate_distractors, num_to_select)
            # except ValueError:
            #     print(f"Warning: Failed to parse model response: {selection_text}")
            #     return random.sample(candidate_distractors, num_to_select)

        # Validate indices and select distractors
        selected_distractors = []
        for idx in selected_indices:
            if 0 <= idx < len(candidate_distractors):
                selected_distractors.append(candidate_distractors[idx])

        if len(selected_distractors) == num_to_select:
            return selected_distractors
        else:
            print(
                f"Warning: Model selection returned {len(selected_distractors)} distractors instead of {num_to_select}"
            )
            # Fill with random selection if needed
            remaining = [d for d in candidate_distractors if d not in selected_distractors]
            needed = num_to_select - len(selected_distractors)
            if needed > 0 and remaining:
                selected_distractors.extend(random.sample(remaining, min(needed, len(remaining))))
            return selected_distractors[:num_to_select]

    except Exception as e:
        print(f"Error in model selection: {e}")
        print("Falling back to random selection")
        return random.sample(candidate_distractors, num_to_select)


def evaluate_distractor_quality(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    distractors: List[Tuple[str, float, str, str]],
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Tuple[bool, float, str]:
    """
    Evaluate if the current distractors are challenging enough
    Based on build_choice_training_data.py implementation

    Args:
        correct_answer: The correct response
        profile: Person's profile information
        conversation_history: Conversation context
        distractors: List of distractor tuples to evaluate
        model_name: Model to use for evaluation
        api_key: OpenAI API key

    Returns:
        Tuple of (is_challenging_enough, difficulty_score, explanation)
    """
    if not api_key or not OPENAI_AVAILABLE:
        # Default fallback: assume distractors are adequate
        return True, 0.7, "Model evaluation not available, assuming adequate difficulty"

    try:
        client = OpenAI(api_key=api_key, base_url="https://yunwu.zeabur.app/v1")

        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        # Format distractors for evaluation
        distractor_list = []
        for i, (dtype, sim, text, bank) in enumerate(distractors):
            distractor_list.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        evaluation_prompt = f"""You are an expert at evaluating the quality and difficulty of multiple-choice question distractors for personality-based assessments.

Context:
{context_description}

Correct Answer: {correct_answer}

Current Distractors:
{chr(10).join(distractor_list)}

Please evaluate these distractors based on the following criteria:

1. **Plausibility**: Do the distractors seem like reasonable responses someone might give?
2. **Personality Discrimination**: Do they test understanding of this specific person's traits and characteristics?
3. **Difficulty Level**: Are they similar enough to the correct answer to be genuinely confusing?
4. **Coverage**: Do they represent different personality types or response patterns?
5. **Contextual Appropriateness**: Do they fit the conversation context and situation?

Consider the similarity scores and distractor types provided. Higher similarity scores indicate more challenging distractors.

Rate the overall quality on a scale of 0.0 to 1.0 where:
- 0.0-0.3: Poor distractors (too obvious, irrelevant, or inappropriate)
- 0.4-0.6: Adequate distractors (decent but could be more challenging)
- 0.7-0.8: Good distractors (appropriately challenging and well-designed)
- 0.9-1.0: Excellent distractors (highly challenging and sophisticated)

Please respond in the following format:
SCORE: [0.0-1.0]
CHALLENGING_ENOUGH: [YES/NO]
EXPLANATION: [Brief explanation of your assessment, including specific strengths and weaknesses]

Consider whether someone who understands this person's personality well would find these distractors genuinely confusing or if they would be too easy to eliminate."""

        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an expert educational assessment designer specializing in personality-based multiple-choice questions. You have extensive experience in creating challenging but fair distractors.",
        #         },
        #         {"role": "user", "content": evaluation_prompt},
        #     ],
        #     max_tokens=300,
        #     temperature=0.3,
        # )

        # # Parse the response
        # response_text = response.choices[0].message.content.strip()
        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are an expert educational assessment designer specializing in personality-based multiple-choice questions. You have extensive experience in creating challenging but fair distractors.",
                        },
                        {"role": "user", "content": evaluation_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                response_text = response.output_text.strip()
                break
            except Exception as e:
                print("Warning: API timeout, retrying...")
                if i < 2:
                    # time.sleep(2**i)  # Exponential backoff
                    model_name = "gpt-5-mini-2025-08-07"
                    continue
                else:
                    print("Max retries reached, returning default evaluation")
                    return True, 0.7, "Evaluation failed due to repeated exceptions"
            # except Exception as e:
            #     print(f"Error during evaluation: {e}")
            #     return True, 0.7, f"Evaluation failed: {e}"

        # Extract score
        difficulty_score = 0.7  # default
        is_challenging = True  # default
        explanation = "Failed to parse evaluation response"

        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    difficulty_score = float(score_text)
                except ValueError:
                    pass
            elif line.startswith("CHALLENGING_ENOUGH:"):
                answer_text = line.replace("CHALLENGING_ENOUGH:", "").strip().upper()
                is_challenging = answer_text.startswith("Y")  # YES
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        # Set threshold for "challenging enough"
        if not is_challenging or difficulty_score < 0.6:
            is_challenging = False

        return is_challenging, difficulty_score, explanation

    except Exception as e:
        print(f"Error in distractor quality evaluation: {e}")
        # Default fallback
        return True, 0.7, f"Evaluation failed: {e}"


def generate_enhanced_distractor(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    existing_distractors: List[Tuple[str, float, str, str]],
    target_weakness: str = "general",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Optional[Tuple[str, float, str, str]]:
    """
    Generate a single, highly challenging distractor to replace a weak one
    Based on build_choice_training_data.py implementation

    Args:
        correct_answer: The correct response
        profile: Person's profile information
        conversation_history: Conversation context
        existing_distractors: Current distractors for reference
        target_weakness: Specific weakness to address
        model_name: Model to use
        api_key: OpenAI API key

    Returns:
        Enhanced distractor tuple or None if generation failed
    """
    if not api_key or not OPENAI_AVAILABLE:
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://yunwu.zeabur.app/v1")

        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        # Format existing distractors
        existing_list = []
        for i, (dtype, sim, text, bank) in enumerate(existing_distractors):
            existing_list.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        # Customize prompt based on target weakness
        weakness_instructions = {
            "similarity": "Focus on creating a response that is very similar to the correct answer but represents a subtle personality difference.",
            "personality": "Focus on creating a response that tests deep understanding of personality traits and would appeal to someone with different values or characteristics.",
            "context": "Focus on creating a response that fits the conversation context perfectly but shows different situational awareness or priorities.",
            "general": "Focus on creating the most challenging and sophisticated distractor possible.",
        }

        specific_instruction = weakness_instructions.get(target_weakness, weakness_instructions["general"])

        enhancement_prompt = f"""You are tasked with creating one highly challenging distractor for a personality-based multiple-choice question. The current distractors are not challenging enough and need enhancement.

Context:
{context_description}

Correct Answer: {correct_answer}

Current Distractors (for reference):
{chr(10).join(existing_list) if existing_list else "None"}

Special Focus: {specific_instruction}

Create ONE new distractor that is:

1. **Highly Similar**: Close enough to the correct answer that it requires deep understanding to distinguish
2. **Personality-Specific**: Represents how someone with different but plausible personality traits might respond
3. **Contextually Perfect**: Fits the situation and conversation flow naturally
4. **Psychologically Sound**: Based on real personality differences
5. **Maximally Confusing**: Would genuinely challenge someone who knows this person well

The distractor should represent a plausible alternative that someone with different personality traits, values, or communication styles might actually give in this exact situation.

Please provide exactly ONE enhanced distractor, with no additional formatting or explanation."""

        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are a master of psychological assessment and personality theory. You excel at creating sophisticated distractors that test nuanced understanding of individual differences.",
        #         },
        #         {"role": "user", "content": enhancement_prompt},
        #     ],
        #     max_tokens=200,
        # )

        # # Extract and clean the response
        # enhanced_distractor = response.choices[0].message.content.strip()

        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are a master of psychological assessment and personality theory. You excel at creating sophisticated distractors that test nuanced understanding of individual differences.",
                        },
                        {"role": "user", "content": enhancement_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                enhanced_distractor = response.output_text.strip()
                break
            except Exception as e:
                print("Warning: API timeout, retrying...")
                if i < 2:
                    # time.sleep(2**i)  # Exponential backoff
                    model_name = "gpt-5-mini-2025-08-07"
                    continue
                else:
                    print("Max retries reached, returning None")
                    return None

        # Remove any unwanted formatting
        if enhanced_distractor.startswith(("1.", "-", "*", "•")):
            enhanced_distractor = enhanced_distractor[2:].strip()

        # Validate that it's different from correct answer and existing distractors
        if enhanced_distractor and enhanced_distractor != correct_answer.strip():
            existing_texts = [d[2].lower().strip() for d in existing_distractors]
            if enhanced_distractor.lower().strip() not in existing_texts:
                # Return as tuple format (distractor_type, similarity, text, bank)
                return ("enhanced_distractor", 0.75, enhanced_distractor, "model_generated")

        return None

    except Exception as e:
        return None


def build_multi_choice_question_batch(
    banks: Dict[str, List[Dict]],
    tasks: List[QuestionTask],
    use_model_selection: bool = False,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    enhance_weak_distractors: bool = False,
    progress_tracker: Optional[ProgressTracker] = None,
    thread_id: int = 0,
) -> List[Tuple[QuestionTask, Optional[Dict]]]:
    """
    Process a batch of questions in a single thread

    Args:
        banks: Dictionary of distractor banks
        tasks: List of QuestionTask objects to process
        use_model_selection: Whether to use model selection
        model_name: Model name for API calls
        api_key: OpenAI API key
        enhance_weak_distractors: Whether to enhance weak distractors
        progress_tracker: Thread-safe progress tracker
        thread_id: Thread identifier for logging

    Returns:
        List of (task, result) tuples
    """
    results = []

    # Create separate OpenAI client for this thread if needed
    client = None
    if api_key and OPENAI_AVAILABLE and use_model_selection:
        client = OpenAI(api_key=api_key, base_url="https://yunwu.zeabur.app/v1")

    for i, task in enumerate(tasks):
        try:
            # Periodically report progress
            if i % 10 == 0 and progress_tracker:
                completed, failed, total = progress_tracker.get_stats()
                print(
                    f"Thread {thread_id}: Processing {task.qid} ({i+1}/{len(tasks)} in batch, overall: {completed}/{total})"
                )

            question = build_multi_choice_question_single(
                banks=banks,
                qid=task.qid,
                num_choices=task.num_choices,
                use_model_selection=use_model_selection,
                model_name=model_name,
                api_key=api_key,
                enhance_weak_distractors=enhance_weak_distractors,
                client=client,  # Pass pre-created client
            )

            if question:
                question["original_index"] = task.actual_idx
                results.append((task, question))
                if progress_tracker:
                    progress_tracker.update(success=True)
            else:
                results.append((task, None))
                if progress_tracker:
                    progress_tracker.update(success=False)

            # Add delay for API rate limiting
            if use_model_selection and api_key:
                time.sleep(0.05)  # Reduced delay since we're using multiple threads

        except Exception as e:
            print(f"Thread {thread_id}: Error processing {task.qid}: {e}")
            results.append((task, None))
            if progress_tracker:
                progress_tracker.update(success=False)

    return results


def build_multi_choice_question_single(
    banks: Dict[str, List[Dict]],
    qid: str,
    num_choices: int = 4,
    use_model_selection: bool = False,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    enhance_weak_distractors: bool = False,
    client: Optional[OpenAI] = None,
) -> Optional[Dict]:
    """
    Build a single multi-choice question (thread-safe version)

    This is a modified version of the original build_multi_choice_question
    that accepts a pre-created OpenAI client to avoid creating multiple clients per thread
    """
    # Find the item with this qid from any bank (they should have same content)
    reference_item = None
    for bank_data in banks.values():
        for item in bank_data:
            if item.get("qid") == qid:
                reference_item = item
                break
        if reference_item:
            break

    if not reference_item:
        return None

    correct_answer = reference_item["output"]
    prompt = reference_item.get("prompt", "")

    # Extract profile and conversation history for enhanced model selection
    profile, conversation_history = extract_profile_conv_from_prompt(prompt)

    # Extract all distractors from all banks (already deduplicated)
    all_distractors = extract_all_distractors(banks, qid)

    num_distractors = num_choices - 1  # One slot for correct answer

    if len(all_distractors) < num_distractors:
        return None

    # Select optimal distractors with enhanced logic
    selected_distractors = select_optimal_distractors_single(
        all_distractors,
        correct_answer,
        profile,
        conversation_history,
        num_choices=num_distractors,
        use_model_selection=use_model_selection,
        model_name=model_name,
        api_key=api_key,
        enhance_weak_distractors=enhance_weak_distractors,
        client=client,  # Pass the pre-created client
    )

    # Additional check: ensure no duplicates in selected distractors
    seen_texts = set()
    final_distractors = []
    for distractor in selected_distractors:
        text = distractor[2].strip().lower()
        if text not in seen_texts:
            seen_texts.add(text)
            final_distractors.append(distractor)

    # Also check that distractors are not the same as correct answer
    correct_text = correct_answer.strip().lower()
    final_distractors = [d for d in final_distractors if d[2].strip().lower() != correct_text]

    if len(final_distractors) < max(1, num_distractors - 2):  # Allow some flexibility
        return None

    # Build choices list
    choices = []

    # Add correct answer (always gets score 5)
    choices.append({"text": correct_answer, "label": "A", "is_correct": True, "score": 5, "type": "correct_output"})

    # Add distractors
    labels = [chr(66 + i) for i in range(len(final_distractors))]  # B, C, D, E, F, ...
    for i, (distractor_type, similarity, distractor_text, bank_name) in enumerate(final_distractors):
        score = calculate_choice_score(similarity)
        choices.append(
            {
                "text": distractor_text,
                "label": labels[i],
                "is_correct": False,
                "score": score,
                "type": distractor_type,
                "similarity": similarity,
                "bank": bank_name,
            }
        )

    # Shuffle choices while keeping track of correct answer
    random.shuffle(choices)

    # Update labels after shuffling
    for i, choice in enumerate(choices):
        choice["label"] = chr(65 + i)  # A, B, C, D, E, F, ...
        if choice["is_correct"]:
            correct_label = choice["label"]

    # Calculate question difficulty metrics
    similarities = [c["similarity"] for c in choices if "similarity" in c]
    difficulty_metrics = {
        "avg_similarity": np.mean(similarities) if similarities else 0,
        "max_similarity": max(similarities) if similarities else 0,
        "min_similarity": min(similarities) if similarities else 0,
        "similarity_std": np.std(similarities) if similarities else 0,
    }

    question = {
        "qid": qid,
        "question_text": f"Choose the most appropriate response:",
        "choices": choices,
        "correct_answer": correct_label,
        "num_choices": len(choices),  # Use actual number of choices (may be less than requested)
        "difficulty_metrics": difficulty_metrics,
        "original_prompt": prompt,
        "metadata": {
            "total_available_distractors": len(all_distractors),
            "selected_distractors": len(final_distractors),
            "banks_used": list(set(d[3] for d in final_distractors)),
            "model_selection": use_model_selection,
            "enhanced_distractors": enhance_weak_distractors and use_model_selection,
            "requested_choices": num_choices,
            "actual_choices": len(choices),
        },
    }

    return question


def select_optimal_distractors_single(
    distractors: List[Tuple[str, float, str, str]],
    correct_answer: str,
    profile: str,
    conversation_history: str,
    num_choices: int = 3,
    use_model_selection: bool = False,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    enhance_weak_distractors: bool = False,
    client: Optional[OpenAI] = None,
) -> List[Tuple[str, float, str, str]]:
    """
    Thread-safe version of select_optimal_distractors that accepts a pre-created client
    """
    if len(distractors) < num_choices:
        return distractors

    # Sort distractors by similarity score
    sorted_distractors = sorted(distractors, key=lambda x: x[1], reverse=True)

    # Extract profile and conversation history for model-based selection
    selected_distractors = []

    # If using model selection, use enhanced approach
    if use_model_selection and len(sorted_distractors) > num_choices * 2 and api_key and client:
        # Strategy for model selection mode:
        # 1. Pre-select some high/medium similarity distractors randomly
        # 2. Let model choose the remaining ones from a candidate pool

        high_sim = [d for d in sorted_distractors if d[1] > HI_SIM_THRESHOLD]
        medium_sim = [d for d in sorted_distractors if 0.4 <= d[1] <= HI_SIM_THRESHOLD]

        # Pre-select 1-2 distractors randomly from high/medium similarity
        pre_selected = []
        num_pre_select = min(num_choices - 2, len(high_sim) + len(medium_sim))
        num_pre_select = max(1, num_pre_select)

        # Randomly select from high and medium similarity pools
        candidate_pool_for_random = high_sim + medium_sim
        if len(candidate_pool_for_random) >= num_pre_select:
            pre_selected = random.sample(candidate_pool_for_random, num_pre_select)

        # Remaining slots for model to choose
        num_model_select = num_choices - len(pre_selected)

        if num_model_select > 0:
            # Create candidate pool for model selection (exclude pre-selected ones)
            remaining_candidates = [d for d in sorted_distractors if d not in pre_selected]

            # Limit candidate pool size to avoid overwhelming the model
            candidate_pool_size = min(len(remaining_candidates), max(num_model_select * 3, 10))
            model_candidate_pool = remaining_candidates[:candidate_pool_size]

            # Use model to select the most confusing ones
            model_selected = model_select_distractors_single(
                correct_answer,
                profile,
                conversation_history,
                model_candidate_pool,
                num_model_select,
                model_name,
                api_key,
                client,
            )

            # Combine pre-selected and model-selected
            selected_distractors = pre_selected + model_selected
        else:
            selected_distractors = pre_selected

        # Enhancement step: evaluate and potentially improve distractors
        if enhance_weak_distractors and len(selected_distractors) >= 2:
            is_challenging, difficulty_score, explanation = evaluate_distractor_quality_single(
                correct_answer, profile, conversation_history, selected_distractors, model_name, api_key, client
            )

            if not is_challenging:
                # Try to generate 1-2 enhanced distractors
                enhanced_distractors = []
                max_enhancements = min(2, len(selected_distractors))

                for attempt in range(max_enhancements):
                    focus_areas = ["similarity", "personality", "context", "general"]
                    target_weakness = focus_areas[attempt % len(focus_areas)]

                    enhanced = generate_enhanced_distractor_single(
                        correct_answer,
                        profile,
                        conversation_history,
                        selected_distractors,
                        target_weakness,
                        model_name,
                        api_key,
                        client,
                    )

                    if enhanced and enhanced not in enhanced_distractors:
                        enhanced_distractors.append(enhanced)

                # Replace weakest distractors with enhanced ones
                if enhanced_distractors:
                    num_to_replace = min(len(enhanced_distractors), len(selected_distractors))
                    selected_distractors[-num_to_replace:] = enhanced_distractors[:num_to_replace]

        return selected_distractors

    # Original diversity strategy (non-model selection mode)
    else:
        # Strategy 1: Ensure diversity in difficulty
        high_sim = [d for d in sorted_distractors if d[1] > HI_SIM_THRESHOLD]
        medium_sim = [d for d in sorted_distractors if 0.3 <= d[1] <= HI_SIM_THRESHOLD]
        low_sim = [d for d in sorted_distractors if d[1] < 0.3]

        # Pre-select for diversity
        diversity_selected = []

        # Select one from each category if available
        if high_sim:
            diversity_selected.append(random.choice(high_sim))
        if low_sim:
            diversity_selected.append(random.choice(low_sim))

        # Fill remaining slots with medium similarity
        remaining_slots = num_choices - len(diversity_selected)
        if remaining_slots > 0 and medium_sim:
            medium_available = [d for d in medium_sim if d not in diversity_selected]
            diversity_selected.extend(random.sample(medium_available, min(remaining_slots, len(medium_available))))
            remaining_slots = num_choices - len(diversity_selected)

        # Fill any remaining slots
        if remaining_slots > 0:
            other_available = [d for d in sorted_distractors if d not in diversity_selected]
            diversity_selected.extend(random.sample(other_available, min(remaining_slots, len(other_available))))

        return diversity_selected


def model_select_distractors_single(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    candidate_distractors: List[Tuple[str, float, str, str]],
    num_to_select: int,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> List[Tuple[str, float, str, str]]:
    """
    Thread-safe version of model_select_distractors that uses a pre-created client
    """
    if len(candidate_distractors) <= num_to_select:
        return candidate_distractors

    if not api_key or not OPENAI_AVAILABLE or not client:
        return random.sample(candidate_distractors, num_to_select)

    try:
        # Prepare candidates for the model
        candidates_text = []
        for i, (dtype, sim, text, bank) in enumerate(candidate_distractors):
            candidates_text.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        selection_prompt = f"""You are tasked with selecting the most confusing and challenging distractors for a personality-based multiple-choice question.

Context:
{context_description}

Correct Answer: {correct_answer}

Candidate Distractors:
{chr(10).join(candidates_text)}

Please select exactly {num_to_select} distractors that would be most confusing for someone trying to identify how this specific person would respond. Choose distractors that:

1. Are plausible responses that someone with different personality traits might give
2. Are similar enough to the correct answer to be confusing but represent different personality aspects
3. Test understanding of the person's specific characteristics, values, and communication style
4. Cover different potential misconceptions about this person's likely response
5. Represent a good mix of difficulty levels

Consider factors like:
- Personality traits and values shown in the profile
- Communication style and preferences
- Consistency with previous conversation patterns
- Emotional intelligence and social awareness levels
- The similarity scores and distractor types provided

Please respond with only the numbers of your selected distractors, separated by commas (e.g., "1,3,7").
"""

        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating challenging personality-based multiple-choice questions that test understanding of individual differences and communication patterns.",
                        },
                        {"role": "user", "content": selection_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                selection_text = response.output_text.strip()
                selected_indices = [int(x.strip()) - 1 for x in selection_text.split(",")]
                break
            except Exception as e:
                if i < 2:
                    time.sleep(2**i)  # Exponential backoff
                    continue
                else:
                    return random.sample(candidate_distractors, num_to_select)

        # Validate indices and select distractors
        selected_distractors = []
        for idx in selected_indices:
            if 0 <= idx < len(candidate_distractors):
                selected_distractors.append(candidate_distractors[idx])

        if len(selected_distractors) == num_to_select:
            return selected_distractors
        else:
            # Fill with random selection if needed
            remaining = [d for d in candidate_distractors if d not in selected_distractors]
            needed = num_to_select - len(selected_distractors)
            if needed > 0 and remaining:
                selected_distractors.extend(random.sample(remaining, min(needed, len(remaining))))
            return selected_distractors[:num_to_select]

    except Exception as e:
        return random.sample(candidate_distractors, num_to_select)


def calculate_choice_score(similarity: float) -> int:
    """Calculate score for each choice based on similarity"""
    if similarity >= 0.8:
        return 3  # Very high similarity
    elif similarity >= 0.6:
        return 2  # High similarity
    elif similarity >= 0.3:
        return 1  # Medium similarity
    else:
        return 0  # Low similarity


def extract_profile_conv_from_prompt(prompt: str) -> Tuple[str, str]:
    """Extract profile and conversation history from prompt text"""
    profile = ""
    conversation_history = ""
    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        if end > start:
            profile = prompt[start:end].strip()
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        if end > start:
            conversation_history = prompt[start:end].strip()
    return profile, conversation_history


def evaluate_distractor_quality_single(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    distractors: List[Tuple[str, float, str, str]],
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[bool, float, str]:
    """
    Thread-safe version of evaluate_distractor_quality
    """
    if not api_key or not OPENAI_AVAILABLE or not client:
        return True, 0.7, "Model evaluation not available, assuming adequate difficulty"

    try:
        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        # Format distractors for evaluation
        distractor_list = []
        for i, (dtype, sim, text, bank) in enumerate(distractors):
            distractor_list.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        evaluation_prompt = f"""You are an expert at evaluating the quality and difficulty of multiple-choice question distractors for personality-based assessments.

Context:
{context_description}

Correct Answer: {correct_answer}

Current Distractors:
{chr(10).join(distractor_list)}

Please evaluate these distractors based on the following criteria:

1. **Plausibility**: Do the distractors seem like reasonable responses someone might give?
2. **Personality Discrimination**: Do they test understanding of this specific person's traits and characteristics?
3. **Difficulty Level**: Are they similar enough to the correct answer to be genuinely confusing?
4. **Coverage**: Do they represent different personality types or response patterns?
5. **Contextual Appropriateness**: Do they fit the conversation context and situation?

Consider the similarity scores and distractor types provided. Higher similarity scores indicate more challenging distractors.

Rate the overall quality on a scale of 0.0 to 1.0 where:
- 0.0-0.3: Poor distractors (too obvious, irrelevant, or inappropriate)
- 0.4-0.6: Adequate distractors (decent but could be more challenging)
- 0.7-0.8: Good distractors (appropriately challenging and well-designed)
- 0.9-1.0: Excellent distractors (highly challenging and sophisticated)

Please respond in the following format:
SCORE: [0.0-1.0]
CHALLENGING_ENOUGH: [YES/NO]
EXPLANATION: [Brief explanation of your assessment, including specific strengths and weaknesses]

Consider whether someone who understands this person's personality well would find these distractors genuinely confusing or if they would be too easy to eliminate."""

        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an expert educational assessment designer specializing in personality-based multiple-choice questions. You have extensive experience in creating challenging but fair distractors.",
        #         },
        #         {"role": "user", "content": evaluation_prompt},
        #     ],
        #     max_tokens=300,
        #     temperature=0.3,
        # )

        # # Parse the response
        # response_text = response.choices[0].message.content.strip()
        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are an expert educational assessment designer specializing in personality-based multiple-choice questions. You have extensive experience in creating challenging but fair distractors.",
                        },
                        {"role": "user", "content": evaluation_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                response_text = response.output_text.strip()
                break
            except Exception as e:
                print("Warning: API timeout, retrying...")
                if i < 2:
                    # time.sleep(2**i)  # Exponential backoff
                    model_name = "gpt-5-mini-2025-08-07"
                    continue
                else:
                    print("Max retries reached, returning default evaluation")
                    return True, 0.7, "Evaluation failed due to repeated exceptions"

        # Extract score
        difficulty_score = 0.7  # default
        is_challenging = True  # default
        explanation = "Failed to parse evaluation response"

        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    difficulty_score = float(score_text)
                except ValueError:
                    pass
            elif line.startswith("CHALLENGING_ENOUGH:"):
                answer_text = line.replace("CHALLENGING_ENOUGH:", "").strip().upper()
                is_challenging = answer_text.startswith("Y")  # YES
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        # Set threshold for "challenging enough"
        if not is_challenging or difficulty_score < 0.6:
            is_challenging = False

        return is_challenging, difficulty_score, explanation

    except Exception as e:
        return True, 0.7, f"Evaluation failed: {e}"


def generate_enhanced_distractor_single(
    correct_answer: str,
    profile: str,
    conversation_history: str,
    existing_distractors: List[Tuple[str, float, str, str]],
    target_weakness: str = "general",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> Optional[Tuple[str, float, str, str]]:
    """
    Thread-safe version of generate_enhanced_distractor
    """
    if not api_key or not OPENAI_AVAILABLE or not client:
        return None

    try:
        # Create context description
        context_parts = []
        if profile:
            context_parts.append(f"Person's Profile:\n{profile}")
        if conversation_history:
            context_parts.append(f"Conversation History:\n{conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        # Format existing distractors
        existing_list = []
        for i, (dtype, sim, text, bank) in enumerate(existing_distractors):
            existing_list.append(f"{i+1}. [{bank}|{dtype}|sim:{sim:.3f}] {text}")

        # Customize prompt based on target weakness
        weakness_instructions = {
            "similarity": "Focus on creating a response that is very similar to the correct answer but represents a subtle personality difference.",
            "personality": "Focus on creating a response that tests deep understanding of personality traits and would appeal to someone with different values or characteristics.",
            "context": "Focus on creating a response that fits the conversation context perfectly but shows different situational awareness or priorities.",
            "general": "Focus on creating the most challenging and sophisticated distractor possible.",
        }

        specific_instruction = weakness_instructions.get(target_weakness, weakness_instructions["general"])

        enhancement_prompt = f"""You are tasked with creating one highly challenging distractor for a personality-based multiple-choice question. The current distractors are not challenging enough and need enhancement.

Context:
{context_description}

Correct Answer: {correct_answer}

Current Distractors (for reference):
{chr(10).join(existing_list) if existing_list else "None"}

Special Focus: {specific_instruction}

Create ONE new distractor that is:

1. **Highly Similar**: Close enough to the correct answer that it requires deep understanding to distinguish
2. **Personality-Specific**: Represents how someone with different but plausible personality traits might respond
3. **Contextually Perfect**: Fits the situation and conversation flow naturally
4. **Psychologically Sound**: Based on real personality differences
5. **Maximally Confusing**: Would genuinely challenge someone who knows this person well

The distractor should represent a plausible alternative that someone with different personality traits, values, or communication styles might actually give in this exact situation.

Please provide exactly ONE enhanced distractor, with no additional formatting or explanation."""

        for i in range(3):  # Retry up to 3 times
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are a master of psychological assessment and personality theory. You excel at creating sophisticated distractors that test nuanced understanding of individual differences.",
                        },
                        {"role": "user", "content": enhancement_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )
                enhanced_distractor = response.output_text.strip()
                break
            except Exception as e:
                print("Warning: API timeout, retrying...")
                if i < 2:
                    # time.sleep(2**i)  # Exponential backoff
                    model_name = "gpt-5-mini-2025-08-07"
                    continue
                else:
                    print("Max retries reached, returning None")
                    return None

        # Remove any unwanted formatting
        if enhanced_distractor.startswith(("1.", "-", "*", "•")):
            enhanced_distractor = enhanced_distractor[2:].strip()

        # Validate that it's different from correct answer and existing distractors
        if enhanced_distractor and enhanced_distractor != correct_answer.strip():
            existing_texts = [d[2].lower().strip() for d in existing_distractors]
            if enhanced_distractor.lower().strip() not in existing_texts:
                # Return as tuple format (distractor_type, similarity, text, bank)
                return ("enhanced_distractor", 0.75, enhanced_distractor, "model_generated")

        return None

    except Exception as e:
        return None


def create_question_batches(
    all_qids: List[str], start_idx: int, args, batch_size: int = 50
) -> List[List[QuestionTask]]:
    """
    Create batches of QuestionTask objects for parallel processing

    Args:
        all_qids: List of question IDs to process
        start_idx: Starting index in the original dataset
        args: Command line arguments
        batch_size: Number of questions per batch

    Returns:
        List of batches, where each batch is a list of QuestionTask objects
    """
    tasks = []
    for i, qid in enumerate(all_qids):
        actual_idx = start_idx + i

        # Determine number of choices for this question
        if args.fixed_choices:
            num_choices = args.fixed_choices
        else:
            num_choices = random.randint(args.min_choices, args.max_choices)

        tasks.append(QuestionTask(qid=qid, actual_idx=actual_idx, num_choices=num_choices))

    # Split into batches
    batches = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        batches.append(batch)

    return batches


def get_all_qids(banks: Dict[str, List[Dict]]) -> List[str]:
    """Get all unique qids from all banks"""
    all_qids = set()
    for bank_data in banks.values():
        for item in bank_data:
            if "qid" in item:
                all_qids.add(item["qid"])
    return list(all_qids)


def analyze_question_quality(questions: List[Dict]) -> Dict:
    """Analyze the quality and distribution of generated questions"""
    if not questions:
        return {}

    total_questions = len(questions)

    # Difficulty analysis
    difficulties = [q["difficulty_metrics"]["avg_similarity"] for q in questions]
    max_similarities = [q["difficulty_metrics"]["max_similarity"] for q in questions]

    # Choice count distribution
    choice_counts = {}
    for q in questions:
        num_choices = q.get("num_choices", 4)
        choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1

    # Score distribution analysis
    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for q in questions:
        for choice in q["choices"]:
            if not choice["is_correct"]:
                s = int(choice.get("score", 0))
                if s in score_counts:
                    score_counts[s] += 1

    # Distractor type distribution
    distractor_types = {}
    bank_usage = {}
    for q in questions:
        for choice in q["choices"]:
            if not choice["is_correct"]:
                dtype = choice["type"]
                distractor_types[dtype] = distractor_types.get(dtype, 0) + 1

                bank = choice.get("bank", "unknown")
                bank_usage[bank] = bank_usage.get(bank, 0) + 1

    # Check for any duplicate issues in analysis
    duplicate_check = {}
    for q in questions:
        texts = [choice["text"].strip().lower() for choice in q["choices"]]
        if len(texts) != len(set(texts)):
            duplicate_check[q["qid"]] = "Has duplicates"

    analysis = {
        "total_questions": total_questions,
        "choice_count_distribution": choice_counts,
        "difficulty_stats": {
            "mean_avg_similarity": np.mean(difficulties),
            "std_avg_similarity": np.std(difficulties),
            "mean_max_similarity": np.mean(max_similarities),
            "questions_with_high_similarity": len([d for d in max_similarities if d >= HI_SIM_THRESHOLD]),
            "questions_with_low_similarity": len([d for d in difficulties if d < 0.3]),
        },
        "score_distribution": score_counts,
        "distractor_type_usage": distractor_types,
        "bank_usage": bank_usage,
        "duplicate_issues": duplicate_check,  # Track any remaining duplicate issues
    }

    return analysis


def classify_question_by_difficulty(question: Dict) -> str:
    """Classify a question into 'hard' | 'medium' | 'easy' | '' based on difficulty metrics"""
    avg_sim = question.get("difficulty_metrics", {}).get("avg_similarity", 0.0)
    max_sim = question.get("difficulty_metrics", {}).get("max_similarity", 0.0)
    high_sim_count = sum(
        1
        for c in question.get("choices", [])
        if (not c.get("is_correct")) and (c.get("similarity", 0.0) >= HI_SIM_THRESHOLD)
    )

    # More flexible difficulty classification
    if max_sim >= 0.85 or (high_sim_count >= 2) or (high_sim_count >= 1 and avg_sim > 0.70):
        return "hard"
    elif max_sim >= 0.65 or (high_sim_count >= 1) or (avg_sim > 0.45):
        return "medium"
    elif avg_sim < 0.30:
        return "easy"
    else:
        return "medium"  # Default to medium for edge cases


def analyze_difficulty_distribution(questions: List[Dict]) -> Dict:
    """Analyze detailed difficulty distribution of questions"""
    if not questions:
        return {}

    # Basic stats
    avg_sims = [q["difficulty_metrics"]["avg_similarity"] for q in questions]
    max_sims = [q["difficulty_metrics"]["max_similarity"] for q in questions]
    min_sims = [q["difficulty_metrics"]["min_similarity"] for q in questions]

    # High similarity count distribution
    high_sim_counts = []
    for q in questions:
        count = sum(
            1 for c in q["choices"] if (not c.get("is_correct")) and (c.get("similarity", 0.0) >= HI_SIM_THRESHOLD)
        )
        high_sim_counts.append(count)

    # Percentile analysis
    avg_sim_percentiles = {
        "p10": np.percentile(avg_sims, 10),
        "p25": np.percentile(avg_sims, 25),
        "p50": np.percentile(avg_sims, 50),
        "p75": np.percentile(avg_sims, 75),
        "p90": np.percentile(avg_sims, 90),
        "p95": np.percentile(avg_sims, 95),
    }

    max_sim_percentiles = {
        "p10": np.percentile(max_sims, 10),
        "p25": np.percentile(max_sims, 25),
        "p50": np.percentile(max_sims, 50),
        "p75": np.percentile(max_sims, 75),
        "p90": np.percentile(max_sims, 90),
        "p95": np.percentile(max_sims, 95),
    }

    # High similarity count distribution
    high_sim_count_dist = {}
    for count in high_sim_counts:
        high_sim_count_dist[count] = high_sim_count_dist.get(count, 0) + 1

    # Similarity range distribution
    avg_sim_ranges = {
        "0.0-0.2": len([s for s in avg_sims if 0.0 <= s < 0.2]),
        "0.2-0.3": len([s for s in avg_sims if 0.2 <= s < 0.3]),
        "0.3-0.4": len([s for s in avg_sims if 0.3 <= s < 0.4]),
        "0.4-0.5": len([s for s in avg_sims if 0.4 <= s < 0.5]),
        "0.5-0.6": len([s for s in avg_sims if 0.5 <= s < 0.6]),
        "0.6-0.7": len([s for s in avg_sims if 0.6 <= s < 0.7]),
        "0.7-0.8": len([s for s in avg_sims if 0.7 <= s < 0.8]),
        "0.8-0.9": len([s for s in avg_sims if 0.8 <= s < 0.9]),
        "0.9-1.0": len([s for s in avg_sims if 0.9 <= s <= 1.0]),
    }

    max_sim_ranges = {
        "0.0-0.2": len([s for s in max_sims if 0.0 <= s < 0.2]),
        "0.2-0.3": len([s for s in max_sims if 0.2 <= s < 0.3]),
        "0.3-0.4": len([s for s in max_sims if 0.3 <= s < 0.4]),
        "0.4-0.5": len([s for s in max_sims if 0.4 <= s < 0.5]),
        "0.5-0.6": len([s for s in max_sims if 0.5 <= s < 0.6]),
        "0.6-0.7": len([s for s in max_sims if 0.6 <= s < 0.7]),
        "0.7-0.8": len([s for s in max_sims if 0.7 <= s < 0.8]),
        "0.8-0.9": len([s for s in max_sims if 0.8 <= s < 0.9]),
        "0.9-1.0": len([s for s in max_sims if 0.9 <= s <= 1.0]),
    }

    return {
        "total_questions": len(questions),
        "avg_similarity_stats": {
            "mean": np.mean(avg_sims),
            "std": np.std(avg_sims),
            "min": np.min(avg_sims),
            "max": np.max(avg_sims),
            "percentiles": avg_sim_percentiles,
        },
        "max_similarity_stats": {
            "mean": np.mean(max_sims),
            "std": np.std(max_sims),
            "min": np.min(max_sims),
            "max": np.max(max_sims),
            "percentiles": max_sim_percentiles,
        },
        "min_similarity_stats": {
            "mean": np.mean(min_sims),
            "std": np.std(min_sims),
            "min": np.min(min_sims),
            "max": np.max(min_sims),
        },
        "high_sim_count_distribution": high_sim_count_dist,
        "avg_similarity_ranges": avg_sim_ranges,
        "max_similarity_ranges": max_sim_ranges,
    }


def main():
    parser = argparse.ArgumentParser(description="Build multi-choice questions from multiple distractor banks")
    parser.add_argument(
        "--banks", nargs="+", required=True, help="Input JSON files with similarity data from different banks"
    )
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for questions")
    parser.add_argument("--analysis", "-a", help="Output file for quality analysis")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample-size", type=int, help="Number of questions to generate (default: all)")

    # Choice configuration
    parser.add_argument("--min-choices", type=int, default=4, help="Minimum number of choices (default: 4)")
    parser.add_argument("--max-choices", type=int, default=6, help="Maximum number of choices (default: 6)")
    parser.add_argument("--fixed-choices", type=int, help="Fixed number of choices (overrides min/max)")

    # Model selection arguments (enhanced)
    parser.add_argument(
        "--use-model-selection", action="store_true", help="Use model to select most confusing distractors"
    )
    parser.add_argument("--model-name", default="gpt-4o-mini", help="Model name for distractor selection")
    parser.add_argument("--api-key", help="OpenAI API key for model selection")
    parser.add_argument(
        "--enhance-weak-distractors",
        action="store_true",
        help="Evaluate distractor quality and enhance weak ones (requires --use-model-selection)",
    )

    # Index range arguments
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index for processing questions (inclusive, 0-based)"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End index for processing questions (exclusive, -1 means process until end)",
    )

    # Parallel processing arguments
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Number of questions per batch (default: 50)")

    # Difficulty bucket outputs
    parser.add_argument("--hard-output", help="Output JSON for hard-mode questions")
    parser.add_argument("--medium-output", help="Output JSON for medium-mode questions")
    parser.add_argument("--easy-output", help="Output JSON for easy-mode questions")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load multiple banks
    print("Loading distractor banks...")
    banks = load_multiple_banks(args.banks)

    # Get all unique qids
    all_qids = get_all_qids(banks)
    total_questions = len(all_qids)
    print(f"Found {total_questions} unique questions across all banks")

    # Apply index range filtering
    start_idx = max(0, args.start_index)
    if args.end_index == -1:
        end_idx = total_questions
    else:
        end_idx = min(total_questions, args.end_index)

    # Ensure start_idx < end_idx
    if start_idx >= end_idx:
        print(f"ERROR: Invalid index range. start_index ({start_idx}) must be less than end_index ({end_idx})")
        print(f"Total questions available: {total_questions}")
        sys.exit(1)

    # Slice the qids list
    original_total = len(all_qids)
    all_qids = all_qids[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} (total: {len(all_qids)} out of {original_total})")

    # Sample qids if requested (after range filtering)
    if args.sample_size and args.sample_size < len(all_qids):
        all_qids = random.sample(all_qids, args.sample_size)
        print(f"Sampled {len(all_qids)} questions from range")

    # Enhanced model selection validation
    if args.use_model_selection:
        if not OPENAI_AVAILABLE:
            print("ERROR: Cannot use model selection - openai library not available!")
            print("Please install openai: pip install openai")
            sys.exit(1)
        if not args.api_key:
            print("Warning: Model selection enabled but no API key provided. Will fall back to rule-based selection.")
        print(f"Model selection enabled using: {args.model_name}")
        if args.enhance_weak_distractors:
            print("Distractor quality enhancement enabled")

    # Update output filename to include range info if not processing full dataset
    if start_idx != 0 or end_idx != original_total:
        # Modify output path to include range
        base_out = args.output
        name, ext = os.path.splitext(base_out)
        if ext:
            # File output - add range to filename
            range_suffix = f"_range_{start_idx}_{end_idx-1}"
            args.output = f"{name}{range_suffix}{ext}"
        else:
            # Directory output - add range to directory name
            args.output = f"{base_out}_range_{start_idx}_{end_idx-1}"

        print(f"Output path updated to include range: {args.output}")

    # Create question batches for parallel processing
    print(f"Creating batches for parallel processing (batch size: {args.batch_size}, threads: {args.num_threads})")
    batches = create_question_batches(all_qids, start_idx, args, args.batch_size)
    print(f"Created {len(batches)} batches")

    # Initialize progress tracker
    progress_tracker = ProgressTracker(len(all_qids))

    # Process batches in parallel
    print("Starting parallel processing...")
    questions = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all batches
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(
                build_multi_choice_question_batch,
                banks=banks,
                tasks=batch,
                use_model_selection=args.use_model_selection,
                model_name=args.model_name,
                api_key=args.api_key,
                enhance_weak_distractors=args.enhance_weak_distractors,
                progress_tracker=progress_tracker,
                thread_id=i,
            )
            future_to_batch[future] = i

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()

                # Process results from this batch
                for task, question in batch_results:
                    if question:
                        questions.append(question)
                    else:
                        failed_count += 1

                # Print batch completion status
                completed, failed, total = progress_tracker.get_stats()
                print(f"Batch {batch_id} completed. Overall progress: {completed}/{total} (failed: {failed})")

            except Exception as e:
                print(f"Batch {batch_id} failed with error: {e}")
                failed_count += len(batches[batch_id])

    print(f"Parallel processing completed!")
    print(f"Successfully built {len(questions)} questions from range {start_idx}-{end_idx-1}")
    if failed_count > 0:
        print(f"Failed to build {failed_count} questions (insufficient unique distractors)")

    # Sort questions by original index to maintain order
    questions.sort(key=lambda x: x.get("original_index", 0))

    # Save questions
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Questions saved to: {args.output}")

    # Analyze question quality
    analysis = analyze_question_quality(questions)

    # Detailed difficulty analysis
    difficulty_analysis = analyze_difficulty_distribution(questions)

    print(f"\n=== Question Quality Analysis (Range {start_idx}-{end_idx-1}) ===")
    print(f"Total questions generated: {analysis['total_questions']}")
    print(f"Choice count distribution: {analysis['choice_count_distribution']}")
    print(f"Mean difficulty (avg similarity): {analysis['difficulty_stats']['mean_avg_similarity']:.4f}")
    print(
        f"Questions with challenging distractors (max sim >= {HI_SIM_THRESHOLD}): {analysis['difficulty_stats']['questions_with_high_similarity']}"
    )
    print(
        f"Questions with easy elimination (avg sim < 0.3): {analysis['difficulty_stats']['questions_with_low_similarity']}"
    )

    print("\n=== Detailed Difficulty Distribution ===")
    print(f"Average Similarity Statistics:")
    print(f"  Mean: {difficulty_analysis['avg_similarity_stats']['mean']:.4f}")
    print(f"  Std:  {difficulty_analysis['avg_similarity_stats']['std']:.4f}")
    print(
        f"  Range: [{difficulty_analysis['avg_similarity_stats']['min']:.4f}, {difficulty_analysis['avg_similarity_stats']['max']:.4f}]"
    )

    print(f"\nAverage Similarity Percentiles:")
    for p, v in difficulty_analysis["avg_similarity_stats"]["percentiles"].items():
        print(f"  {p}: {v:.4f}")

    print(f"\nMaximum Similarity Statistics:")
    print(f"  Mean: {difficulty_analysis['max_similarity_stats']['mean']:.4f}")
    print(f"  Std:  {difficulty_analysis['max_similarity_stats']['std']:.4f}")
    print(
        f"  Range: [{difficulty_analysis['max_similarity_stats']['min']:.4f}, {difficulty_analysis['max_similarity_stats']['max']:.4f}]"
    )

    print(f"\nMaximum Similarity Percentiles:")
    for p, v in difficulty_analysis["max_similarity_stats"]["percentiles"].items():
        print(f"  {p}: {v:.4f}")

    print(f"\nHigh Similarity Count Distribution (>= {HI_SIM_THRESHOLD}):")
    for count, freq in sorted(difficulty_analysis["high_sim_count_distribution"].items()):
        print(f"  {count} high-sim distractors: {freq} questions ({freq/len(questions)*100:.1f}%)")

    print(f"\nAverage Similarity Range Distribution:")
    for range_str, count in difficulty_analysis["avg_similarity_ranges"].items():
        print(f"  {range_str}: {count} questions ({count/len(questions)*100:.1f}%)")

    print(f"\nMaximum Similarity Range Distribution:")
    for range_str, count in difficulty_analysis["max_similarity_ranges"].items():
        print(f"  {range_str}: {count} questions ({count/len(questions)*100:.1f}%)")

    print("\nScore distribution:")
    for score, count in analysis["score_distribution"].items():
        print(f"  Score {score}: {count}")

    print("\nBank usage:")
    for bank, count in sorted(analysis["bank_usage"].items()):
        print(f"  {bank}: {count}")

    print("\nDistractor type usage:")
    for dtype, count in sorted(analysis["distractor_type_usage"].items()):
        print(f"  {dtype}: {count}")

    # Report any duplicate issues
    if analysis["duplicate_issues"]:
        print(f"\nWarning: Found {len(analysis['duplicate_issues'])} questions with potential duplicate issues")
    else:
        print("\nNo duplicate issues detected ✓")

    # Classify into difficulty datasets with optimized thresholds
    hard_questions, medium_questions, easy_questions = [], [], []
    for q in questions:
        bucket = classify_question_by_difficulty(q)
        if bucket == "hard":
            hard_questions.append(q)
        elif bucket == "medium":
            medium_questions.append(q)
        elif bucket == "easy":
            easy_questions.append(q)

    print("\n=== Optimized Difficulty Buckets ===")
    print(
        f"Hard   (max_sim>=0.85 OR high_sim_count>=2 OR (high_sim_count>=1 AND avg_sim>0.70)): {len(hard_questions)} ({len(hard_questions)/len(questions)*100:.1f}%)"
    )
    print(
        f"Medium (max_sim>=0.65 OR high_sim_count>=1 OR avg_sim>0.45): {len(medium_questions)} ({len(medium_questions)/len(questions)*100:.1f}%)"
    )
    print(f"Easy   (avg_sim<0.30): {len(easy_questions)} ({len(easy_questions)/len(questions)*100:.1f}%)")

    # Update difficulty bucket output paths to include range if needed
    if start_idx != 0 or end_idx != original_total:
        range_suffix = f"_range_{start_idx}_{end_idx-1}"

        if args.hard_output:
            name, ext = os.path.splitext(args.hard_output)
            args.hard_output = f"{name}{range_suffix}{ext}"
        if args.medium_output:
            name, ext = os.path.splitext(args.medium_output)
            args.medium_output = f"{name}{range_suffix}{ext}"
        if args.easy_output:
            name, ext = os.path.splitext(args.easy_output)
            args.easy_output = f"{name}{range_suffix}{ext}"

    # Save per-bucket outputs if requested
    if args.hard_output:
        with open(args.hard_output, "w", encoding="utf-8") as f:
            json.dump(hard_questions, f, indent=2, ensure_ascii=False)
        print(f"Hard questions saved to: {args.hard_output}")

    if args.medium_output:
        with open(args.medium_output, "w", encoding="utf-8") as f:
            json.dump(medium_questions, f, indent=2, ensure_ascii=False)
        print(f"Medium questions saved to: {args.medium_output}")

    if args.easy_output:
        with open(args.easy_output, "w", encoding="utf-8") as f:
            json.dump(easy_questions, f, indent=2, ensure_ascii=False)
        print(f"Easy questions saved to: {args.easy_output}")

    # Save analysis if requested
    if args.analysis:
        # Update analysis output path to include range if needed
        if start_idx != 0 or end_idx != original_total:
            name, ext = os.path.splitext(args.analysis)
            args.analysis = f"{name}_range_{start_idx}_{end_idx-1}{ext}"

        combined_analysis = {
            **analysis,
            "difficulty_distribution": difficulty_analysis,
            "processing_range": {
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "total_processed": len(questions),
                "original_total": original_total,
            },
            "parallel_processing": {
                "num_threads": args.num_threads,
                "batch_size": args.batch_size,
                "num_batches": len(batches),
                "failed_questions": failed_count,
            },
        }
        with open(args.analysis, "w", encoding="utf-8") as f:
            json.dump(combined_analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {args.analysis}")

    print(f"\nGenerated dataset from range {start_idx}-{end_idx-1} using {args.num_threads} threads")
    print("Done!")


if __name__ == "__main__":
    main()
