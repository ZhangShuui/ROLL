import json
import torch
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
import argparse
import re
import os

# ========================= 1. Prompt 模板 ========================= #

GENERATION_MESSAGE = [
    {
        "role": "system",
        "content": "You are an answer agent for multiple-choice questions. Your task is to think briefly through the question and output one letter 'A' or 'B' within /choice{{}} tags. Keep your analysis concise and focused.",
    },
    {
        "role": "user",
        "content": "Choose 'A' or 'B' for the following question: {question}\n{reference_data}\n\nBriefly analyze:\n1. Key personality traits from the profile\n2. Context from conversation\n3. Which option better matches the person's style\n\nProvide your answer in the format: /choice{{X}} where X is either A or B.\n\nYour response:",
    },
]


def extract_profile_conv_from_prompt(prompt):
    """Extract profile and conversation from the prompt"""
    profile = ""
    conversation_history = ""

    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        profile = prompt[start:end].strip()

    # Extract conversation history
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        conversation_history = prompt[start:end].strip()

    return profile, conversation_history


def load_dataset(file_path, start_index=None, end_index=None):
    """Load dataset from jsonl file with optional index range"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:  # Skip empty lines
                # Apply index filtering
                if start_index is not None and i < start_index:
                    continue
                if end_index is not None and i >= end_index:
                    break

                item = json.loads(line)
                item["dataset_index"] = i  # Add original index
                data.append(item)
    return data


def create_binary_choice_question(correct_answer, distractor, profile, conversation):
    """Create a binary choice question with correct answer and distractor"""
    # Randomly decide whether correct answer is A or B
    if random.choice([True, False]):
        option_a = correct_answer
        option_b = distractor
        correct_choice = "A"
    else:
        option_a = distractor
        option_b = correct_answer
        correct_choice = "B"

    # Create the question
    question = f"[Profile Begin]{profile}[Profile End]\n"
    question += f"[Conversation History Begin]{conversation}[Conversation History End]\n\n"
    question += "Which response is most appropriate for this person in this context?\n\n"
    question += f"A. {option_a}\n"
    question += f"B. {option_b}\n"

    return question, correct_choice, option_a, option_b


def prepare_prompts_for_distractor_type(data, distractor_type, tokenizer):
    """Prepare prompts for a specific distractor type"""
    prompts = []
    correct_choices = []
    valid_items = []
    question_details = []

    for item in data:
        # Check if this distractor type exists for this item
        distractor_key = f"{distractor_type}_distractor"
        if distractor_key not in item or item[distractor_key].startswith("Failed_"):
            continue

        profile, conversation = extract_profile_conv_from_prompt(item["prompt"])
        correct_answer = item["output"]
        distractor = item[distractor_key]

        # Create binary choice question
        question, correct_choice, option_a, option_b = create_binary_choice_question(
            correct_answer, distractor, profile, conversation
        )

        # Apply chat template
        messages = [
            GENERATION_MESSAGE[0],  # system message
            {"role": "user", "content": GENERATION_MESSAGE[1]["content"].format(question=question, reference_data="")},
        ]

        prompt = tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        prompts.append(prompt)
        correct_choices.append(correct_choice)
        valid_items.append(item)

        # Store question details for later use
        question_details.append(
            {
                "qid": item.get("qid", f"unknown_{len(question_details)}"),
                "dataset_index": item.get("dataset_index", -1),
                "distractor_type": distractor_type,
                "correct_answer": correct_answer,
                "distractor": distractor,
                "option_a": option_a,
                "option_b": option_b,
                "correct_choice": correct_choice,
                "question": question,
            }
        )

    return prompts, correct_choices, valid_items, question_details


def batch_inference(model, tokenizer, prompts, batch_size=16):
    """Perform batch inference"""
    results = []
    valid_indices = []
    full_responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize batch
        input_tokens = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        # Generate responses with shorter max_new_tokens for concise thinking
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=2048,  # Increased to allow for thinking process
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                top_p=0.9,
            )

        # Decode responses
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[input_tokens["input_ids"].shape[1] :], skip_special_tokens=True)
            response = response.strip()
            full_responses.append(response)

            # Extract answer from /choice{X} format
            bos_pattern = r"/choice\{([AB])\}"
            match = re.search(bos_pattern, response)

            if match:
                answer = match.group(1).upper()
                results.append(answer)
                valid_indices.append(i + j)
            else:
                # Fallback: try to extract A or B from the response
                if "A" in response.upper() and "B" not in response.upper():
                    results.append("A")
                    valid_indices.append(i + j)
                elif "B" in response.upper() and "A" not in response.upper():
                    results.append("B")
                    valid_indices.append(i + j)
                else:
                    # Invalid response, skip
                    print(f"Invalid response format: {response[:100]}...")
                    results.append("INVALID")
                    valid_indices.append(i + j)

    return results, valid_indices, full_responses


def evaluate_distractor_type(model, tokenizer, data, distractor_type, batch_size=16, output_file=None):
    """Evaluate accuracy for a specific distractor type"""
    print(f"\n{'='*60}")
    print(f"Evaluating distractor type: {distractor_type}")
    print(f"{'='*60}")

    # Prepare prompts for this distractor type
    prompts, correct_choices, valid_items, question_details = prepare_prompts_for_distractor_type(
        data, distractor_type, tokenizer
    )

    if len(prompts) == 0:
        print(f"No valid data found for distractor type: {distractor_type}")
        return None, []

    print(f"Valid samples for {distractor_type}: {len(prompts)}")

    # Perform inference
    predictions, valid_indices, full_responses = batch_inference(model, tokenizer, prompts, batch_size)

    if len(predictions) == 0:
        print(f"No valid predictions for distractor type: {distractor_type}")
        return None, []

    # Process results and create detailed records
    detailed_results = []
    correct_count = 0

    for i, (pred_idx, prediction) in enumerate(zip(valid_indices, predictions)):
        if pred_idx < len(question_details):
            detail = question_details[pred_idx]
            correct_choice = detail["correct_choice"]
            is_correct = prediction == correct_choice

            if is_correct:
                correct_count += 1

            # Create detailed result record
            result_record = {
                "qid": detail["qid"],
                "dataset_index": detail["dataset_index"],
                "distractor_type": distractor_type,
                "correct_answer": detail["correct_answer"],
                "distractor": detail["distractor"],
                "option_a": detail["option_a"],
                "option_b": detail["option_b"],
                "correct_choice": correct_choice,
                "predicted_choice": prediction,
                "is_correct": is_correct,
                "model_response": full_responses[pred_idx] if pred_idx < len(full_responses) else "",
            }
            detailed_results.append(result_record)

    # Save individual results to file if specified
    if output_file and detailed_results:
        # Append to the output file
        with open(output_file, "a", encoding="utf-8") as f:
            for record in detailed_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(detailed_results)
    accuracy = correct_count / total * 100 if total > 0 else 0

    print(f"Total samples: {len(prompts)}")
    print(f"Valid predictions: {total}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "distractor_type": distractor_type,
        "total_samples": len(prompts),
        "valid_predictions": total,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
    }, detailed_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate choice questions with multiple distractor types")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/hdtaccuracy/Personality-Alignment/choice_ver/v8/raw_choice_data_v8_merged.jsonl",
        help="Path to input data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/project/hdtaccuracy/Personality-Alignment/choice_ver/v8/evaluation_detailed_results.jsonl",
        help="Path to output detailed results file",
    )
    parser.add_argument(
        "--model_path", type=str, default="/project/hdtaccuracy/models/base/Qwen3-8B", help="Path to model"
    )
    parser.add_argument("--start_index", type=int, default=None, help="Start index for data processing (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="End index for data processing (exclusive)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument(
        "--distractor_types",
        nargs="+",
        default=[
            "style_violation",
            "topic_violation",
            "richness_violation",
            "free_violation",
            "profile_violation_w",
            "conversation_violation_w",
            "both_violation_w",
            "profile_violation_w/o",
            "conversation_violation_w/o",
            "both_violation_w/o",
        ],
        help="List of distractor types to evaluate",
    )
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    print(f"{'='*80}")
    print("Choice Question Evaluation with Multiple Distractor Types")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Model path: {args.model_path}")
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else "END"
        print(f"Processing range: [{start_idx}:{end_idx}]")
    print(f"Batch size: {args.batch_size}")
    print(f"Distractor types: {args.distractor_types}")
    print(f"{'='*80}")

    # Load model and tokenizer
    print("Loading model...")

    # Setup quantization if needed
    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset with index range
    print("Loading dataset...")
    data = load_dataset(args.data_path, args.start_index, args.end_index)
    print(f"Loaded {len(data)} samples")

    # Prepare output file (clear it first)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        pass  # Clear the file

    # Store results for all distractor types
    all_results = []
    all_detailed_results = []

    # Test each distractor type
    for distractor_type in args.distractor_types:
        result, detailed_results = evaluate_distractor_type(
            model, tokenizer, data, distractor_type, batch_size=args.batch_size, output_file=args.output_path
        )
        if result:
            all_results.append(result)
            all_detailed_results.extend(detailed_results)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL DISTRACTOR TYPES")
    print(f"{'='*80}")
    print(f"{'Distractor Type':<25} {'Samples':<8} {'Valid':<7} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['distractor_type']:<25} "
            f"{result['total_samples']:<8} "
            f"{result['valid_predictions']:<7} "
            f"{result['correct_predictions']:<8} "
            f"{result['accuracy']:<10.2f}%"
        )

    # Calculate average accuracy
    if all_results:
        avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
        print("-" * 80)
        print(f"{'AVERAGE':<25} {'':<8} {'':<7} {'':<8} {avg_accuracy:<10.2f}%")

    # Save summary results to file
    summary_output_file = args.output_path.replace(".jsonl", "_summary.json")
    with open(summary_output_file, "w", encoding="utf-8") as f:
        summary_data = {
            "evaluation_config": {
                "data_path": args.data_path,
                "model_path": args.model_path,
                "start_index": args.start_index,
                "end_index": args.end_index,
                "batch_size": args.batch_size,
                "total_samples": len(data),
                "distractor_types": args.distractor_types,
            },
            "results": all_results,
            "average_accuracy": avg_accuracy if all_results else 0,
        }
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {args.output_path}")
    print(f"Summary results saved to: {summary_output_file}")
    print(f"Total detailed records: {len(all_detailed_results)}")


if __name__ == "__main__":
    main()
