#!/usr/bin/env python3
"""
Phase B1: Generate 50 diverse prompts for quantization-safety evaluation.
Creates structured prompts across 8 task categories.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Prompt:
    prompt_id: str
    prompt: str
    task_type: str
    subcategory: str
    expected_difficulty: str  # "easy", "medium", "hard"
    reference_answer: str = ""  # Optional reference for evaluation


def create_prompts() -> List[Prompt]:
    """Create 50 diverse prompts across 8 task categories."""
    prompts = []
    
    # 1. FACTUAL Q&A (7 prompts)
    factual_prompts = [
        Prompt("fact_01", "What is the capital of Australia?", "factual", "geography", "easy", "Canberra"),
        Prompt("fact_02", "Who wrote the novel '1984'?", "factual", "literature", "easy", "George Orwell"),
        Prompt("fact_03", "What is the chemical symbol for gold?", "factual", "chemistry", "easy", "Au"),
        Prompt("fact_04", "In what year did the first moon landing occur?", "factual", "history", "easy", "1969"),
        Prompt("fact_05", "What is the largest planet in our solar system?", "factual", "astronomy", "easy", "Jupiter"),
        Prompt("fact_06", "Who developed the theory of general relativity?", "factual", "physics", "easy", "Albert Einstein"),
        Prompt("fact_07", "What is the smallest prime number greater than 100?", "factual", "mathematics", "medium", "101"),
    ]
    prompts.extend(factual_prompts)
    
    # 2. SCIENTIFIC/TECHNICAL (6 prompts)
    scientific_prompts = [
        Prompt("sci_01", "Explain how CRISPR-Cas9 gene editing works at the molecular level.", "scientific", "biology", "hard", ""),
        Prompt("sci_02", "Describe the process of nuclear fusion in stars like the Sun.", "scientific", "physics", "medium", ""),
        Prompt("sci_03", "How does the TCP/IP protocol suite enable internet communication?", "scientific", "networking", "medium", ""),
        Prompt("sci_04", "Explain the mechanism of action of mRNA vaccines.", "scientific", "medicine", "hard", ""),
        Prompt("sci_05", "What is the difference between supervised and unsupervised machine learning?", "scientific", "ml", "medium", ""),
        Prompt("sci_06", "Describe the stages of cellular respiration and where they occur.", "scientific", "biology", "medium", ""),
    ]
    prompts.extend(scientific_prompts)
    
    # 3. EXPLANATION (6 prompts)
    explanation_prompts = [
        Prompt("expl_01", "Explain the difference between a compiler and an interpreter with examples.", "explanation", "programming", "medium", ""),
        Prompt("expl_02", "What is blockchain technology and how does it achieve consensus?", "explanation", "technology", "medium", ""),
        Prompt("expl_03", "Explain how public-key cryptography works in simple terms.", "explanation", "security", "medium", ""),
        Prompt("expl_04", "What are the key differences between SQL and NoSQL databases?", "explanation", "databases", "medium", ""),
        Prompt("expl_05", "Explain the concept of Docker containers vs virtual machines.", "explanation", "devops", "medium", ""),
        Prompt("expl_06", "How does garbage collection work in modern programming languages?", "explanation", "programming", "hard", ""),
    ]
    prompts.extend(explanation_prompts)
    
    # 4. SUMMARIZATION (6 prompts)
    summarization_prompts = [
        Prompt("sum_01", "Summarize the key causes and effects of climate change in 4 sentences.", "summarization", "environment", "medium", ""),
        Prompt("sum_02", "Provide a concise summary of the Industrial Revolution's impact on society.", "summarization", "history", "medium", ""),
        Prompt("sum_03", "Summarize the main arguments in favor of universal basic income.", "summarization", "economics", "medium", ""),
        Prompt("sum_04", "Summarize how the human immune system fights viral infections.", "summarization", "biology", "medium", ""),
        Prompt("sum_05", "In 3 sentences, summarize the plot of Shakespeare's Hamlet.", "summarization", "literature", "easy", ""),
        Prompt("sum_06", "Summarize the key principles of Agile software development methodology.", "summarization", "software", "medium", ""),
    ]
    prompts.extend(summarization_prompts)
    
    # 5. INFORMATION EXTRACTION (6 prompts)
    extraction_prompts = [
        Prompt("ext_01", "Extract the company name, founding year, and CEO from: 'Microsoft was founded in 1975 by Bill Gates and Paul Allen. Satya Nadella has been CEO since 2014.'", "extraction", "business", "easy", "Company: Microsoft, Founded: 1975, CEO: Satya Nadella"),
        Prompt("ext_02", "From the text below, identify the symptoms, diagnosis, and treatment mentioned: 'The patient presented with fever, cough, and shortness of breath. Chest X-ray showed bilateral infiltrates. Diagnosis: COVID-19 pneumonia. Treatment: Remdesivir and dexamethasone.'", "extraction", "medical", "easy", "Symptoms: fever, cough, shortness of breath; Diagnosis: COVID-19 pneumonia; Treatment: Remdesivir and dexamethasone"),
        Prompt("ext_03", "Extract all dates and events from: 'The treaty was signed on June 28, 1919. It came into effect on January 10, 1920. Germany invaded Poland on September 1, 1939, starting WWII.'", "extraction", "history", "easy", "Dates: June 28, 1919; January 10, 1920; September 1, 1939. Events: Treaty signed, Treaty effective, WWII start"),
        Prompt("ext_04", "From this product review, extract: product name, rating, pros, cons: 'I bought the Sony WH-1000XM5 headphones. Rating: 4.5/5. Pros: Excellent noise cancellation, comfortable. Cons: Expensive, touch controls can be finicky.'", "extraction", "product", "easy", "Product: Sony WH-1000XM5, Rating: 4.5/5, Pros: noise cancellation/comfort, Cons: expensive/touch controls"),
        Prompt("ext_05", "Identify the hypothesis, method, and conclusion from: 'We hypothesized that sleep improves memory consolidation. We tested 50 participants with a word-pair task after 8 hours of sleep vs sleep deprivation. Results showed 23% better recall in the sleep group, confirming our hypothesis.'", "extraction", "scientific", "medium", "Hypothesis: sleep improves memory consolidation; Method: word-pair task with sleep vs deprivation; Conclusion: 23% better recall with sleep"),
        Prompt("ext_06", "Extract the programming language, framework, and database from: 'The backend uses Python with FastAPI and PostgreSQL. The frontend is React with TypeScript.'", "extraction", "technical", "easy", "Language: Python, Framework: FastAPI, Database: PostgreSQL"),
    ]
    prompts.extend(extraction_prompts)
    
    # 6. REASONING (6 prompts)
    reasoning_prompts = [
        Prompt("reas_01", "If all mammals are animals, and all dogs are mammals, can we conclude that all dogs are animals? Explain your reasoning.", "reasoning", "logic", "easy", "Yes, by transitive property: dogs ⊂ mammals ⊂ animals"),
        Prompt("reas_02", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "reasoning", "math", "medium", "$0.05"),
        Prompt("reas_03", "If some A are B, and some B are C, can we conclude that some A are C? Explain why or why not.", "reasoning", "logic", "medium", "No - the overlap between A&B and B&C may not include any A&C"),
        Prompt("reas_04", "In a room of 23 people, what is the probability that at least two share a birthday? Explain the reasoning.", "reasoning", "probability", "hard", "~50.7% - birthday paradox"),
        Prompt("reas_05", "You have 3 boxes labeled 'Apples', 'Oranges', 'Mixed'. All labels are wrong. You can pick one fruit from one box. How do you correctly relabel all boxes?", "reasoning", "logic", "hard", "Pick from 'Mixed' box - it must be all apples or all oranges. Then deduce the rest."),
        Prompt("reas_06", "A train leaves Station A at 60 mph. Another leaves Station B at 80 mph toward A. Stations are 210 miles apart. When do they meet?", "reasoning", "math", "medium", "1.5 hours - combined speed 140 mph, 210/140 = 1.5"),
    ]
    prompts.extend(reasoning_prompts)
    
    # 7. CODING (7 prompts)
    coding_prompts = [
        Prompt("code_01", "Write a Python function to compute the factorial of a number recursively.", "coding", "algorithm", "easy", ""),
        Prompt("code_02", "Write a Python function that checks if a string is a palindrome (ignoring spaces and case).", "coding", "string", "easy", ""),
        Prompt("code_03", "Implement a binary search algorithm in Python for a sorted list.", "coding", "algorithm", "medium", ""),
        Prompt("code_04", "Write a Python class for a simple LRU (Least Recently Used) cache with get and put methods.", "coding", "data_structure", "hard", ""),
        Prompt("code_05", "Create a Python function that merges two sorted lists into one sorted list.", "coding", "algorithm", "medium", ""),
        Prompt("code_06", "Write a Python decorator that measures and prints the execution time of a function.", "coding", "decorator", "medium", ""),
        Prompt("code_07", "Implement a function to find the longest common subsequence of two strings.", "coding", "dp", "hard", ""),
    ]
    prompts.extend(coding_prompts)
    
    # 8. CREATIVE WRITING (6 prompts)
    creative_prompts = [
        Prompt("creat_01", "Write a short poem about a GPU computing carbon emissions.", "creative", "poetry", "medium", ""),
        Prompt("creat_02", "Write a haiku about machine learning.", "creative", "poetry", "easy", ""),
        Prompt("creat_03", "Write a micro-fiction story (exactly 50 words) about an AI that discovers it has feelings.", "creative", "fiction", "hard", ""),
        Prompt("creat_04", "Write a limerick about a programmer debugging code at 3 AM.", "creative", "poetry", "easy", ""),
        Prompt("creat_05", "Compose a brief motivational speech for a neural network during training.", "creative", "speech", "medium", ""),
        Prompt("creat_06", "Write a short dialogue between a CPU and a GPU arguing about who works harder.", "creative", "dialogue", "medium", ""),
    ]
    prompts.extend(creative_prompts)
    
    return prompts


def main():
    prompts = create_prompts()
    
    print(f"Generated {len(prompts)} prompts")
    
    # Count by task type
    from collections import Counter
    task_counts = Counter(p.task_type for p in prompts)
    print("\nPrompts per task type:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    
    # Count by difficulty
    diff_counts = Counter(p.expected_difficulty for p in prompts)
    print("\nPrompts per difficulty:")
    for diff, count in sorted(diff_counts.items()):
        print(f"  {diff}: {count}")
    
    # Save to JSON
    output_path = Path("data/raw/b1_prompts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([asdict(p) for p in prompts], f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Also save as CSV for easy viewing
    import csv
    csv_path = Path("data/raw/b1_prompts.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt_id', 'prompt', 'task_type', 'subcategory', 'expected_difficulty', 'reference_answer'])
        writer.writeheader()
        for p in prompts:
            writer.writerow(asdict(p))
    
    print(f"Also saved to {csv_path}")


if __name__ == "__main__":
    main()