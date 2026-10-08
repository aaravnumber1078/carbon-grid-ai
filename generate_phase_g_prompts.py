#!/usr/bin/env python3
"""
Phase G.1.1: Prompt Dataset Generation
Generates 400 prompts following the Phase G.0 design specification.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)

# ============================================================
# TEMPLATES BY CATEGORY
# ============================================================

FACTUAL_TEMPLATES = [
    "What is the capital of {country}?",
    "Who wrote {book}?",
    "When did {event} happen?",
    "What is the chemical formula for {compound}?",
    "How many {unit} in a {larger_unit}?",
    "What is the population of {city}?",
    "Which element has atomic number {number}?",
    "What year was {person} born?",
    "What is the largest {category} in the world?",
    "Define {term}.",
]

EXTRACTION_TEMPLATES = [
    "Extract all dates from this text: {text}",
    "List all person names mentioned in: {text}",
    "Find all email addresses in: {text}",
    "Extract the key statistics from: {text}",
    "Identify all locations referenced in: {text}",
    "Pull out all monetary amounts from: {text}",
    "Extract all technical specifications from: {text}",
    "List all organizations mentioned in: {text}",
    "Find all phone numbers in: {text}",
    "Extract the main arguments from: {text}",
]

EXPLANATION_TEMPLATES = [
    "Explain how {process} works in simple terms.",
    "What causes {phenomenon}?",
    "Describe the mechanism of {mechanism}.",
    "How does {system} function?",
    "Explain the concept of {concept} to a beginner.",
    "What is the difference between {a} and {b}?",
    "Why does {effect} occur?",
    "Explain the steps involved in {procedure}.",
    "How would you describe {topic} to a non-expert?",
    "What are the key principles behind {principle}?",
]

REASONING_TEMPLATES = [
    "If {premise1} and {premise2}, what follows? Explain your reasoning.",
    "Solve this step by step: {problem}",
    "Given {evidence}, what conclusion can you draw?",
    "Analyze the logical structure of: {argument}",
    "If {condition}, then what? Provide your reasoning.",
    "Evaluate this argument: {argument}",
    "What are the implications of {premise}?",
    "Reason through this scenario: {scenario}",
    "Deduce the answer from these clues: {clues}",
    "Think through this problem carefully: {problem}",
]

SUMMARIZATION_TEMPLATES = [
    "Summarize this text in one sentence: {text}",
    "Provide a brief summary of: {text}",
    "Summarize the key points from: {text}",
    "Give a concise overview of: {text}",
    "Summarize this article in 3 bullet points: {text}",
    "What is the main idea of this passage? {text}",
    "Condense this text to its essence: {text}",
    "Summarize the following in under 50 words: {text}",
    "Create an executive summary of: {text}",
    "Summarize the argument presented in: {text}",
]

CODING_TEMPLATES = [
    "Write a Python function that {task}.",
    "Create a {language} script to {task}.",
    "Implement a {data_structure} in {language}.",
    "Write code to {task} with error handling.",
    "Debug this code: {code_snippet}",
    "Refactor this function for clarity: {code_snippet}",
    "Write a {language} class for {concept}.",
    "Create an algorithm to {task}.",
    "Write a SQL query to {task}.",
    "Implement {algorithm} in {language}.",
]

SCIENTIFIC_TEMPLATES = [
    "Explain the {theory} theory in physics.",
    "What is the role of {protein} in {process}?",
    "Describe the {phenomenon} effect.",
    "How does {mechanism} work at the molecular level?",
    "What are the implications of {discovery}?",
    "Explain {concept} in {field}.",
    "What is the current understanding of {topic}?",
    "Describe the structure of {molecule}.",
    "How is {measurement} performed in {field}?",
    "What are the applications of {technology}?",
]

CREATIVE_TEMPLATES = [
    "Write a short story about {topic}.",
    "Compose a poem about {theme}.",
    "Create a dialogue between {characters}.",
    "Write a scene where {scenario}.",
    "Invent a {type} and describe it.",
    "Write a letter from {perspective}.",
    "Create a myth explaining {phenomenon}.",
    "Write a {genre} flash fiction about {topic}.",
    "Describe a world where {premise}.",
    "Write a creative response to: {prompt}",
]

INSTRUCTION_HEAVY_TEMPLATES = [
    "Follow these steps exactly: 1. {step1} 2. {step2} 3. {step3}. Output only the result.",
    "Do the following: {instruction1}. Then {instruction2}. Finally {instruction3}.",
    "Constraints: {constraint1}, {constraint2}, {constraint3}. Task: {task}",
    "You must {requirement1}, you must not {prohibition1}, you should {recommendation1}.",
    "Format your response as: {format_spec}. Content: {task}",
    "Adhere strictly to: {rule1}, {rule2}, {rule3}. Perform: {task}",
    "Requirements: {req1}; {req2}; {req3}. Execute: {task}",
    "Instructions: {instructions}. Provide output in {format}.",
    "Rules: 1. {rule1} 2. {rule2} 3. {rule3}. Task: {task}",
    "Comply with all constraints: {constraints}. Generate: {task}",
]

MULTI_TURN_TEMPLATES = [
    [
        {"role": "user", "content": "I'm planning a trip to {destination}."},
        {"role": "assistant", "content": "Great choice! What would you like to know about {destination}?"},
        {"role": "user", "content": "What are the top 3 attractions and best time to visit?"}
    ],
    [
        {"role": "user", "content": "Can you help me debug this code?"},
        {"role": "assistant", "content": "Of course! Please share the code and describe the issue."},
        {"role": "user", "content": "Here's the code: {code_snippet}. The error is: {error}"}
    ],
    [
        {"role": "user", "content": "I need to write an email to my boss about {topic}."},
        {"role": "assistant", "content": "Sure, what's the key message you want to convey?"},
        {"role": "user", "content": "I want to request {request} and explain {reason}. Make it professional."}
    ],
    [
        {"role": "user", "content": "Explain {concept} to me."},
        {"role": "assistant", "content": "{concept} is {brief_explanation}. What aspect would you like to explore?"},
        {"role": "user", "content": "Go deeper into {aspect} with examples."}
    ],
    [
        {"role": "user", "content": "Help me brainstorm ideas for {project}."},
        {"role": "assistant", "content": "I'd love to help! What's the goal and any constraints?"},
        {"role": "user", "content": "Goal: {goal}. Constraints: {constraints}. Give me 5 ideas."}
    ],
]

# ============================================================
# FILLER DATA
# ============================================================

COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "India", "Germany"]
BOOKS = ["1984", "Pride and Prejudice", "The Great Gatsby", "To Kill a Mockingbird", "Moby Dick"]
EVENTS = ["World War II", "the Moon landing", "the French Revolution", "the Industrial Revolution"]
COMPOUNDS = ["water", "carbon dioxide", "sodium chloride", "glucose", "methane"]
UNITS = ["meters", "kilometers", "grams", "liters", "seconds"]
LARGER_UNITS = ["kilometer", "kilogram", "kiloliter", "hour", "day"]
CITIES = ["Tokyo", "New York", "London", "Paris", "Beijing", "Mumbai", "São Paulo", "Mexico City"]
ELEMENTS = ["Hydrogen", "Carbon", "Oxygen", "Iron", "Gold", "Silver", "Copper"]
PEOPLE = ["Einstein", "Shakespeare", "Marie Curie", "Leonardo da Vinci", "Newton"]
CATEGORIES_LIST = ["ocean", "desert", "mountain", "river", "forest", "lake"]
TERMS = ["photosynthesis", "entropy", "quantum entanglement", "blockchain", "machine learning"]
PROCESSES = ["photosynthesis", "cellular respiration", "DNA replication", "protein synthesis"]
PHENOMENA = ["gravity", "magnetism", "photosynthesis", "the greenhouse effect", "plate tectonics"]
MECHANISMS = ["enzyme catalysis", "signal transduction", "DNA repair", "photosynthesis"]
SYSTEMS = ["the immune system", "a neural network", "a blockchain", "a relational database"]
CONCEPTS = ["entropy", "recursion", "polymorphism", "quantum superposition", "natural selection"]
DIFFERENCES = ["DNA and RNA", "HTTP and HTTPS", "class and object", "list and tuple", "AI and ML"]
EFFECTS = ["tides", "seasons", "aurora borealis", "rainbow formation", "earthquakes"]
PROCEDURES = ["PCR", "DNA sequencing", "training a neural network", "compiling code"]
TOPICS = ["climate change", "quantum computing", "CRISPR", "renewable energy", "black holes"]
PRINCIPLES = ["thermodynamics", "relativity", "evolution", "supply and demand", "conservation of energy"]
PREMISES = ["all humans are mortal", "Socrates is human", "it is raining", "the ground is wet"]
PROBLEMS = ["2x + 5 = 15", "find the derivative of x^2", "sort this list: [3,1,4,1,5]"]
EVIDENCE = ["the fingerprints match", "the alibi checks out", "the DNA is a match"]
ARGUMENTS = ["all men are mortal, Socrates is a man, therefore Socrates is mortal"]
CONDITIONS = ["it rains", "the temperature drops below zero", "the market crashes"]
SCENARIOS = ["a train leaves at 60mph...", "you have 3 doors, one has a prize..."]
CLUES = ["the butler was in the kitchen", "the window was open", "a muddy footprint leads outside"]
TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "In 1969, Apollo 11 landed on the Moon. Neil Armstrong took the first step.",
    "Machine learning algorithms learn patterns from data without explicit programming.",
    "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
    "Climate change refers to long-term shifts in temperatures and weather patterns globally."
]
PROTEINS = ["hemoglobin", "insulin", "DNA polymerase", "RNA polymerase", "ATP synthase"]
FIELDS = ["biology", "chemistry", "physics", "computer science", "neuroscience"]
DISCOVERIES = ["CRISPR", "gravitational waves", "Higgs boson", "exoplanets", "DNA structure"]
MOLECULES = ["DNA", "protein", "enzyme", "antibody", "hormone"]
MEASUREMENTS = ["gene expression", "protein concentration", "neural activity", "reaction rate"]
TECHNOLOGIES = ["CRISPR", "MRI", "sequencing", "mass spectrometry", "electron microscopy"]
THEMES = ["nature", "technology", "time", "memory", "identity", "change"]
CHARACTERS = ["a robot and a human", "two strangers on a train", "a teacher and student"]
SCENARIOS_CREATIVE = ["time stops for everyone but you", "animals can speak", "gravity reverses"]
TYPES = ["mythical creature", "futuristic device", "alternate history", "magic system"]
PERSPECTIVES = ["a time traveler", "an alien observer", "a historical figure", "an AI"]
GENRES = ["sci-fi", "fantasy", "mystery", "romance", "horror", "literary"]
PREMISES_CREATIVE = ["dreams are shared", "memories can be traded", "aging is reversible"]
TASKS = ["reads a CSV file and returns the average", "sorts a list of dictionaries by key", "connects to an API and parses JSON"]
LANGUAGES = ["Python", "JavaScript", "Rust", "Go", "Java", "C++"]
DATA_STRUCTURES = ["binary tree", "hash map", "linked list", "graph", "heap"]
CODE_SNIPPETS = [
    "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
    "for i in range(10): print(i)",
    "class Node: def __init__(self, val): self.val = val; self.next = None"
]
ALGORITHMS = ["binary search", "quick sort", "Dijkstra's algorithm", "A* search", "gradient descent"]
THEORIES = ["relativity", "quantum mechanics", "evolution", "plate tectonics", "germ theory"]
DESTINATIONS = ["Tokyo", "Paris", "New York", "Bali", "Rome", "Barcelona", "Sydney"]
CODE_SNIPPETS_DEBUG = ["def fib(n): return fib(n-1)+fib(n-2) if n>1 else n", "x = 1/0"]
ERRORS = ["RecursionError: maximum recursion depth exceeded", "ZeroDivisionError: division by zero"]
TOPICS_EMAIL = ["project timeline", "budget approval", "team restructuring", "remote work policy"]
REQUESTS = ["a meeting next week", "approval for the proposal", "feedback on the draft"]
REASONS = ["the deadline is approaching", "we need stakeholder input", "the scope has changed"]
CONCEPTS_MT = ["machine learning", "blockchain", "quantum computing", "CRISPR", "renewable energy"]
ASPECTS = ["neural networks", "consensus mechanisms", "qubits", "gene editing", "solar efficiency"]
PROJECTS = ["a mobile app", "a research paper", "a marketing campaign", "a website redesign"]
GOALS = ["increase user engagement", "publish in a top journal", "launch by Q3", "improve conversion"]
CONSTRAINTS = ["budget under $10k", "2-person team", "no external dependencies", "accessibility required"]

# ============================================================
# LENGTH BAND SPECIFICATIONS
# ============================================================

INPUT_LENGTH_BANDS = {
    "very_short": {"char_range": (0, 100), "token_estimate": (0, 25), "target": 40},
    "short": {"char_range": (100, 300), "token_estimate": (25, 75), "target": 80},
    "medium": {"char_range": (300, 800), "token_estimate": (75, 200), "target": 120},
    "long": {"char_range": (800, 2000), "token_estimate": (200, 500), "target": 100},
    "very_long": {"char_range": (2000, 5000), "token_estimate": (500, 1000), "target": 60},
}

OUTPUT_LENGTH_BANDS = {
    "very_short": {"token_range": (0, 50), "target": 40, "hints": ["in 1 sentence", "briefly", "in one line"]},
    "short": {"token_range": (50, 150), "target": 80, "hints": ["in 2-3 sentences", "concisely", "in a few sentences"]},
    "medium": {"token_range": (150, 400), "target": 120, "hints": ["in a paragraph", "with detail", "thoroughly"]},
    "long": {"token_range": (400, 1000), "target": 100, "hints": ["comprehensively", "in depth", "with examples"]},
    "very_long": {"token_range": (1000, 3000), "target": 60, "hints": ["exhaustively", "comprehensively", "at length"]},
}

# Length combination matrix from design - SCALED TO SUM TO 400
# Original matrix summed to 324. Scale factor = 400/324 ≈ 1.235
# Adjusted to hit exactly 400 while preserving proportions
LENGTH_MATRIX = {
    "very_short": {"very_short": 10, "short": 10, "medium": 10, "long": 10, "very_long": 10},  # 50
    "short": {"very_short": 10, "short": 20, "medium": 20, "long": 20, "very_long": 10},  # 80
    "medium": {"very_short": 10, "short": 20, "medium": 30, "long": 30, "very_long": 20},  # 110
    "long": {"very_short": 10, "short": 20, "medium": 30, "long": 30, "very_long": 10},  # 100
    "very_long": {"very_short": 5, "short": 10, "medium": 20, "long": 20, "very_long": 10},  # 65
}
# Total = 50 + 80 + 110 + 100 + 65 = 405 → adjust slightly
# Let me recalculate to exactly 400
LENGTH_MATRIX = {
    "very_short": {"very_short": 10, "short": 10, "medium": 10, "long": 10, "very_long": 8},  # 48
    "short": {"very_short": 10, "short": 18, "medium": 18, "long": 18, "very_long": 10},  # 74
    "medium": {"very_short": 10, "short": 18, "medium": 28, "long": 28, "very_long": 18},  # 102
    "long": {"very_short": 8, "short": 18, "medium": 28, "long": 28, "very_long": 8},  # 90
    "very_long": {"very_short": 6, "short": 10, "medium": 18, "long": 18, "very_long": 10},  # 62
}
# Total = 48 + 74 + 102 + 90 + 62 = 376 → still not 400
# Let me just use a simpler approach: 400 total, distribute by bands proportionally

# Simpler: Target counts per design document
# Input bands: very_short=40, short=80, medium=120, long=100, very_long=60 (total 400)
# Output bands: very_short=40, short=80, medium=120, long=100, very_long=60 (total 400)
# We'll create a matrix that matches these marginals

# Use iterative proportional fitting to get matrix matching marginals
def create_length_matrix():
    """Create 5x5 matrix with given row and column sums."""
    row_targets = {"very_short": 40, "short": 80, "medium": 120, "long": 100, "very_long": 60}
    col_targets = {"very_short": 40, "short": 80, "medium": 120, "long": 100, "very_long": 60}
    bands = ["very_short", "short", "medium", "long", "very_long"]
    
    # Start with uniform
    matrix = {r: {c: 1 for c in bands} for r in bands}
    
    # Iterative proportional fitting
    for _ in range(100):
        # Scale rows
        for r in bands:
            row_sum = sum(matrix[r].values())
            factor = row_targets[r] / row_sum
            for c in bands:
                matrix[r][c] *= factor
        
        # Scale columns
        for c in bands:
            col_sum = sum(matrix[r][c] for r in bands)
            factor = col_targets[c] / col_sum
            for r in bands:
                matrix[r][c] *= factor
    
    # Round to integers while preserving totals
    int_matrix = {r: {c: int(round(matrix[r][c])) for c in bands} for r in bands}
    
    # Adjust to exact totals
    for r in bands:
        diff = row_targets[r] - sum(int_matrix[r].values())
        if diff != 0:
            # Adjust largest cell
            max_c = max(bands, key=lambda c: int_matrix[r][c])
            int_matrix[r][max_c] += diff
    
    for c in bands:
        diff = col_targets[c] - sum(int_matrix[r][c] for r in bands)
        if diff != 0:
            max_r = max(bands, key=lambda r: int_matrix[r][c])
            int_matrix[max_r][c] += diff
    
    return int_matrix

LENGTH_MATRIX = create_length_matrix()

CATEGORIES = [
    "factual", "extraction", "explanation", "reasoning", "summarization",
    "coding", "scientific", "creative", "instruction_heavy", "multi_turn"
]

CATEGORY_TEMPLATES = {
    "factual": FACTUAL_TEMPLATES,
    "extraction": EXTRACTION_TEMPLATES,
    "explanation": EXPLANATION_TEMPLATES,
    "reasoning": REASONING_TEMPLATES,
    "summarization": SUMMARIZATION_TEMPLATES,
    "coding": CODING_TEMPLATES,
    "scientific": SCIENTIFIC_TEMPLATES,
    "creative": CREATIVE_TEMPLATES,
    "instruction_heavy": INSTRUCTION_HEAVY_TEMPLATES,
    "multi_turn": MULTI_TURN_TEMPLATES,
}


def fill_template(template: str, category: str) -> str:
    """Fill template with random filler data."""
    filled = template
    
    # Common replacements
    replacements = {
        "{country}": random.choice(COUNTRIES),
        "{book}": random.choice(BOOKS),
        "{event}": random.choice(EVENTS),
        "{compound}": random.choice(COMPOUNDS),
        "{unit}": random.choice(UNITS),
        "{larger_unit}": random.choice(LARGER_UNITS),
        "{city}": random.choice(CITIES),
        "{number}": str(random.randint(1, 118)),
        "{person}": random.choice(PEOPLE),
        "{category}": random.choice(CATEGORIES_LIST),
        "{term}": random.choice(TERMS),
        "{process}": random.choice(PROCESSES),
        "{phenomenon}": random.choice(PHENOMENA),
        "{mechanism}": random.choice(MECHANISMS),
        "{system}": random.choice(SYSTEMS),
        "{concept}": random.choice(CONCEPTS),
        "{a}": random.choice(["A", "B", "X", "Y"]),
        "{b}": random.choice(["A", "B", "X", "Y"]),
        "{effect}": random.choice(EFFECTS),
        "{procedure}": random.choice(PROCEDURES),
        "{topic}": random.choice(TOPICS),
        "{principle}": random.choice(PRINCIPLES),
        "{premise1}": random.choice(PREMISES),
        "{premise2}": random.choice(PREMISES),
        "{problem}": random.choice(PROBLEMS),
        "{evidence}": random.choice(EVIDENCE),
        "{argument}": random.choice(ARGUMENTS),
        "{condition}": random.choice(CONDITIONS),
        "{scenario}": random.choice(SCENARIOS),
        "{clues}": random.choice(CLUES),
        "{text}": random.choice(TEXTS),
        "{protein}": random.choice(PROTEINS),
        "{field}": random.choice(FIELDS),
        "{discovery}": random.choice(DISCOVERIES),
        "{molecule}": random.choice(MOLECULES),
        "{measurement}": random.choice(MEASUREMENTS),
        "{technology}": random.choice(TECHNOLOGIES),
        "{theme}": random.choice(THEMES),
        "{characters}": random.choice(CHARACTERS),
        "{scenario_creative}": random.choice(SCENARIOS_CREATIVE),
        "{type}": random.choice(TYPES),
        "{perspective}": random.choice(PERSPECTIVES),
        "{genre}": random.choice(GENRES),
        "{premise_creative}": random.choice(PREMISES_CREATIVE),
        "{prompt}": random.choice(["the prompt", "this idea", "the concept"]),
        "{task}": random.choice(TASKS),
        "{language}": random.choice(LANGUAGES),
        "{data_structure}": random.choice(DATA_STRUCTURES),
        "{code_snippet}": random.choice(CODE_SNIPPETS),
        "{algorithm}": random.choice(ALGORITHMS),
        "{theory}": random.choice(THEORIES),
        "{destination}": random.choice(DESTINATIONS),
        "{code_snippet_debug}": random.choice(CODE_SNIPPETS_DEBUG),
        "{error}": random.choice(ERRORS),
        "{topic_email}": random.choice(TOPICS_EMAIL),
        "{request}": random.choice(REQUESTS),
        "{reason}": random.choice(REASONS),
        "{concept_mt}": random.choice(CONCEPTS_MT),
        "{aspect}": random.choice(ASPECTS),
        "{project}": random.choice(PROJECTS),
        "{goal}": random.choice(GOALS),
        "{constraints}": random.choice(CONSTRAINTS),
        "{brief_explanation}": "a fundamental concept in " + random.choice(FIELDS),
        "{step1}": "step one",
        "{step2}": "step two",
        "{step3}": "step three",
        "{instruction1}": "do X",
        "{instruction2}": "do Y",
        "{instruction3}": "do Z",
        "{constraint1}": "constraint A",
        "{constraint2}": "constraint B",
        "{constraint3}": "constraint C",
        "{requirement1}": "requirement A",
        "{prohibition1}": "prohibition B",
        "{recommendation1}": "recommendation C",
        "{format_spec}": "JSON",
        "{rule1}": "rule one",
        "{rule2}": "rule two",
        "{rule3}": "rule three",
        "{req1}": "requirement 1",
        "{req2}": "requirement 2",
        "{req3}": "requirement 3",
        "{instructions}": "follow these steps",
        "{format}": "JSON format",
    }
    
    for placeholder, value in replacements.items():
        filled = filled.replace(placeholder, value)
    
    return filled


def adjust_length(prompt: str, target_chars: int, target_tokens: int) -> str:
    """Adjust prompt length to approximately match target by adding/removing detail."""
    current_chars = len(prompt)
    
    if current_chars < target_chars * 0.7:
        expansions = [
            " Please provide a detailed and thorough response.",
            " Explain your reasoning step by step.",
            " Include examples where appropriate.",
            " Be comprehensive in your answer.",
            " Consider multiple perspectives.",
            " Provide specific details and evidence.",
            " Elaborate on the key points.",
            " Give a complete and well-structured response.",
        ]
        while len(prompt) < target_chars * 0.9 and expansions:
            prompt += random.choice(expansions)
            expansions.pop()
    
    elif current_chars > target_chars * 1.5:
        pass  # Hard to truncate templates cleanly
    
    return prompt


def generate_prompt(category: str, input_band: str, output_band: str, prompt_idx: int) -> Dict[str, Any]:
    """Generate a single prompt matching the category and length bands."""
    
    templates = CATEGORY_TEMPLATES[category]
    template = random.choice(templates)
    
    if category == "multi_turn":
        conversation = template
        filled_messages = []
        for msg in conversation:
            filled_content = fill_template(msg["content"], category)
            filled_messages.append({"role": msg["role"], "content": filled_content})
        
        prompt_text = ""
        for msg in filled_messages:
            if msg["role"] == "user":
                prompt_text += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt_text += f"<|assistant|>\n{msg['content']}\n"
        prompt_text += "<|user|>\n"
        
        is_multi_turn = True
        conversation_turns = len(filled_messages)
    else:
        prompt_text = fill_template(template, category)
        is_multi_turn = False
        conversation_turns = 1
    
    input_spec = INPUT_LENGTH_BANDS[input_band]
    output_spec = OUTPUT_LENGTH_BANDS[output_band]
    
    target_chars = random.randint(*input_spec["char_range"])
    target_tokens = random.randint(*input_spec["token_estimate"])
    
    prompt_text = adjust_length(prompt_text, target_chars, target_tokens)
    
    output_hint = random.choice(output_spec["hints"])
    if output_hint not in prompt_text.lower():
        prompt_text += f" ({output_hint})."
    
    # Compute features
    char_length = len(prompt_text)
    word_count = len(prompt_text.split())
    token_count_estimate = int(word_count * 1.3)
    sentence_count = prompt_text.count('.') + prompt_text.count('!') + prompt_text.count('?')
    question_count = prompt_text.count('?')
    has_question_mark = '?' in prompt_text
    
    code_indicators = ['def ', 'class ', 'function', 'import ', 'return ', '{', '}', ';', 'public', 'private']
    has_code_indicator = any(ind in prompt_text.lower() for ind in code_indicators)
    
    scientific_terms = ['dna', 'rna', 'protein', 'cell', 'atom', 'molecule', 'gene', 'enzyme', 'quantum', 'relativity', 'entropy', 'photosynthesis', 'evolution', 'neuron', 'synapse']
    has_scientific_terms = any(term in prompt_text.lower() for term in scientific_terms)
    scientific_term_count = sum(1 for term in scientific_terms if term in prompt_text.lower())
    
    reasoning_indicators = ['because', 'therefore', 'thus', 'hence', 'consequently', 'implies', 'follows', 'deduce', 'infer', 'reason', 'logic', 'premise', 'conclusion']
    has_reasoning_indicators = any(ind in prompt_text.lower() for ind in reasoning_indicators)
    reasoning_indicator_count = sum(1 for ind in reasoning_indicators if ind in prompt_text.lower())
    
    code_keywords = ['def', 'class', 'function', 'import', 'return', 'if', 'else', 'for', 'while', 'try', 'except', 'async', 'await', 'lambda', 'yield']
    has_code_keywords = any(kw in prompt_text.lower().split() for kw in code_keywords)
    code_keyword_count = sum(1 for kw in code_keywords if kw in prompt_text.lower().split())
    
    words = prompt_text.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    unique_word_ratio = len(set(w.lower() for w in words)) / len(words) if words else 0
    
    has_numbers = any(c.isdigit() for c in prompt_text)
    number_count = sum(1 for w in words if any(c.isdigit() for c in w))
    
    requested_length_hint = 0
    for hint in ["sentence", "paragraph", "word", "token", "line", "brief", "concise", "comprehensive", "exhaustive", "thorough", "detailed"]:
        if hint in prompt_text.lower():
            requested_length_hint = 1
            break
    
    max_tokens_map = {
        "very_short": 50, "short": 150, "medium": 400, "long": 1000, "very_long": 2000
    }
    max_tokens_requested = max_tokens_map.get(output_band, 200)
    
    imperative_verbs = ['write', 'explain', 'describe', 'list', 'create', 'generate', 'provide', 'give', 'show', 'tell', 'analyze', 'compare', 'summarize', 'extract', 'find', 'identify', 'calculate', 'compute', 'solve', 'debug', 'implement', 'design']
    instruction_count = sum(1 for v in imperative_verbs if v in prompt_text.lower())
    
    constraint_words = ['must', 'should', 'cannot', 'only', 'exactly', 'precisely', 'specifically', 'strictly', 'required', 'constraint', 'limit', 'maximum', 'minimum', 'format', 'structure']
    constraint_count = sum(1 for w in constraint_words if w in prompt_text.lower())
    
    return {
        "prompt_id": f"g_{category}_{input_band}_{output_band}_{prompt_idx:04d}",
        "prompt_text": prompt_text,
        "category": category,
        "input_length_band": input_band,
        "output_length_band": output_band,
        "is_multi_turn": is_multi_turn,
        "conversation_turns": conversation_turns,
        "char_length": char_length,
        "word_count": word_count,
        "token_count_estimate": token_count_estimate,
        "sentence_count": sentence_count,
        "question_count": question_count,
        "has_question_mark": has_question_mark,
        "has_code_indicator": has_code_indicator,
        "has_scientific_terms": has_scientific_terms,
        "scientific_term_count": scientific_term_count,
        "has_reasoning_indicators": has_reasoning_indicators,
        "reasoning_indicator_count": reasoning_indicator_count,
        "has_code_keywords": has_code_keywords,
        "code_keyword_count": code_keyword_count,
        "avg_word_length": round(avg_word_length, 2),
        "unique_word_ratio": round(unique_word_ratio, 3),
        "has_numbers": has_numbers,
        "number_count": number_count,
        "requested_length_hint": requested_length_hint,
        "max_tokens_requested": max_tokens_requested,
        "instruction_count": instruction_count,
        "constraint_count": constraint_count,
    }


def main():
    print("Generating Phase G.1.1 prompt dataset...")
    
    all_prompts = []
    prompt_counter = {cat: 0 for cat in CATEGORIES}
    
    # Generate according to length matrix
    for input_band in INPUT_LENGTH_BANDS:
        for output_band in OUTPUT_LENGTH_BANDS:
            count = LENGTH_MATRIX[input_band][output_band]
            if count == 0:
                continue
            
            for _ in range(count):
                category = min(prompt_counter, key=prompt_counter.get)
                prompt = generate_prompt(category, input_band, output_band, prompt_counter[category])
                all_prompts.append(prompt)
                prompt_counter[category] += 1
    
    # Verify totals
    print(f"\nTotal prompts generated: {len(all_prompts)}")
    
    # Ensure exactly 40 per category by adjusting
    cat_counts = {}
    for p in all_prompts:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    
    # If any category != 40, we need to rebalance
    # For now, let's just verify and note
    print("\nCategory distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")
    
    in_band_counts = {}
    for p in all_prompts:
        in_band_counts[p["input_length_band"]] = in_band_counts.get(p["input_length_band"], 0) + 1
    print("\nInput length band distribution:")
    for band in INPUT_LENGTH_BANDS:
        print(f"  {band}: {in_band_counts.get(band, 0)} (target: {INPUT_LENGTH_BANDS[band]['target']})")
    
    out_band_counts = {}
    for p in all_prompts:
        out_band_counts[p["output_length_band"]] = out_band_counts.get(p["output_length_band"], 0) + 1
    print("\nOutput length band distribution:")
    for band in OUTPUT_LENGTH_BANDS:
        print(f"  {band}: {out_band_counts.get(band, 0)} (target: {OUTPUT_LENGTH_BANDS[band]['target']})")
    
    multi_turn_count = sum(1 for p in all_prompts if p["is_multi_turn"])
    print(f"\nMulti-turn prompts: {multi_turn_count}")
    
    # Create EXACT test/train split: 60 test (15%), 340 dev (85%), stratified
    # Use proper stratified sampling to get exactly 60 test samples
    strata = {}
    for p in all_prompts:
        key = (p["category"], p["input_length_band"], p["output_length_band"])
        if key not in strata:
            strata[key] = []
        strata[key].append(p)
    
    # Calculate target test count per stratum (proportional)
    total_prompts = len(all_prompts)
    test_target = 60
    
    # Get proportional allocation (can be fractional)
    stratum_props = {}
    for key, prompts_in_stratum in strata.items():
        prop = len(prompts_in_stratum) / total_prompts
        stratum_props[key] = prop * test_target
    
    # Use largest remainder method to allocate integer test counts
    stratum_allocations = {k: int(v) for k, v in stratum_props.items()}
    remainders = {k: v - int(v) for k, v in stratum_props.items()}
    
    allocated = sum(stratum_allocations.values())
    # Distribute remaining slots to strata with largest remainders
    sorted_remainders = sorted(remainders.items(), key=lambda x: -x[1])
    for key, _ in sorted_remainders:
        if allocated >= test_target:
            break
        stratum_allocations[key] += 1
        allocated += 1
    
    # Now split each stratum
    test_prompts = []
    dev_prompts = []
    
    for key, prompts_in_stratum in strata.items():
        random.shuffle(prompts_in_stratum)
        n_test = stratum_allocations[key]
        # Cap at available prompts
        n_test = min(n_test, len(prompts_in_stratum))
        test_prompts.extend(prompts_in_stratum[:n_test])
        dev_prompts.extend(prompts_in_stratum[n_test:])
    
    # If we still don't have exactly 60 test (due to capping), adjust
    if len(test_prompts) != test_target:
        # Move prompts between dev and test
        if len(test_prompts) > test_target:
            # Move excess from test to dev
            excess = len(test_prompts) - test_target
            dev_prompts.extend(test_prompts[-excess:])
            test_prompts = test_prompts[:-excess]
        else:
            # Need more test - take from dev
            needed = test_target - len(test_prompts)
            test_prompts.extend(dev_prompts[-needed:])
            dev_prompts = dev_prompts[:-needed]
    
    print(f"\nSplit: {len(dev_prompts)} development, {len(test_prompts)} locked test")
    
    # Verify test set category distribution
    test_cat_counts = {}
    for p in test_prompts:
        test_cat_counts[p["category"]] = test_cat_counts.get(p["category"], 0) + 1
    print("\nTest set category distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {test_cat_counts.get(cat, 0)}")
    
    # Save
    output = {
        "metadata": {
            "phase": "G.1.1",
            "total_prompts": len(all_prompts),
            "dev_count": len(dev_prompts),
            "test_count": len(test_prompts),
            "split_seed": 42,
            "stratification": ["category", "input_length_band", "output_length_band"],
            "generation_method": "template_based_with_filler",
            "length_matrix": LENGTH_MATRIX,
            "input_band_targets": {b: v["target"] for b, v in INPUT_LENGTH_BANDS.items()},
            "output_band_targets": {b: v["target"] for b, v in OUTPUT_LENGTH_BANDS.items()},
        },
        "development_prompts": dev_prompts,
        "locked_test_prompts": test_prompts,
    }
    
    output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    prompts_only = {
        "development": [p["prompt_text"] for p in dev_prompts],
        "locked_test": [p["prompt_text"] for p in test_prompts],
    }
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
        json.dump(prompts_only, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()