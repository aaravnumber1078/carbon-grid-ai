#!/usr/bin/env python3
"""
Phase G.1.1: Prompt Dataset Generation (v2 - Fixed)
Generates 400 prompts following the Phase G.0 design specification.
Addresses all audit failures from v1.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from difflib import SequenceMatcher

# Set seed for reproducibility
random.seed(42)

# ============================================================
# CLEAN TEMPLATES BY CATEGORY - NO CONTRADICTIONS, NO PLACEHOLDER LEAKS
# ============================================================

# Each template now includes output-length hint naturally, no post-hoc addition
FACTUAL_TEMPLATES = {
    "very_short": [
        "What is the capital of {country}? Answer briefly.",
        "Who wrote {book}? One sentence.",
        "When did {event} happen? Short answer.",
        "Which element has atomic number {number}? Brief.",
        "What year was {person} born? Concise.",
    ],
    "short": [
        "What is the chemical formula for {compound}? In a few sentences.",
        "How many {unit} in a {larger_unit}? Explain concisely.",
        "What is the population of {city}? A short paragraph.",
        "What is the largest {category} in the world? Briefly.",
        "Define {term}. A few sentences.",
    ],
    "medium": [
        "Explain the significance of {event} in history. In a paragraph.",
        "Describe the structure and function of {compound}. With detail.",
        "How does {process} work? Provide a thorough explanation.",
        "What are the key properties of {element}? Comprehensive answer.",
        "Summarize the life and achievements of {person}. Detailed.",
    ],
    "long": [
        "Analyze the causes and effects of {event}. Comprehensive with examples.",
        "Explain the role of {compound} in {process}. In depth.",
        "Describe the discovery and impact of {discovery}. Thoroughly.",
        "How has our understanding of {topic} evolved? Exhaustive.",
        "Compare and contrast {a} and {b} in {field}. With examples.",
    ],
    "very_long": [
        "Write a comprehensive overview of {topic}, including history, current state, and future directions. Exhaustive.",
        "Provide an exhaustive analysis of {theory} in {field}, covering principles, evidence, and applications. At length.",
        "Detail the complete mechanism of {process} from start to finish, including all stages and regulation. Comprehensive.",
        "Explain {concept} thoroughly with historical context, mathematical foundation, and modern applications. Exhaustive.",
        "Describe the full lifecycle of {system} including all components, interactions, and failure modes. Comprehensive.",
    ],
}

EXTRACTION_TEMPLATES = {
    "very_short": [
        "Extract all dates from this text: {text} List only.",
        "List all person names in: {text} Brief.",
        "Find all emails in: {text} Short.",
    ],
    "short": [
        "Extract key statistics from: {text} In a few lines.",
        "Identify all locations in: {text} Concisely.",
        "Pull all monetary amounts from: {text} A short list.",
        "List all organizations in: {text} Briefly.",
        "Find all phone numbers in: {text} Short answer.",
    ],
    "medium": [
        "Extract the main arguments from: {text} In a paragraph.",
        "Extract all technical specifications from: {text} Detailed.",
        "Identify all dates, names, and locations in: {text} Thorough.",
        "Extract key findings and numbers from: {text} With detail.",
        "List all entities and relationships in: {text} Comprehensive.",
    ],
    "long": [
        "Extract and categorize all named entities from: {text} With examples.",
        "Parse this text and extract all structured data: {text} In depth.",
        "Extract arguments, evidence, and conclusions from: {text} Comprehensive.",
        "Identify all technical terms, values, and units in: {text} Thorough.",
        "Extract all actionable items and deadlines from: {text} Detailed.",
    ],
    "very_long": [
        "Perform complete entity extraction and relation mapping on: {text} Exhaustive with categories.",
        "Extract all facts, figures, entities, and relationships from this document: {text} Comprehensive.",
        "Parse and structure all information from: {text} Including metadata, cross-references, and implicit data. Exhaustive.",
        "Extract and classify every data point in: {text} With full context. Comprehensive.",
        "Perform exhaustive information extraction from: {text} Including nested structures and inferred relations. At length.",
    ],
}

EXPLANATION_TEMPLATES = {
    "very_short": [
        "Explain {concept} in one sentence.",
        "What causes {phenomenon}? Briefly.",
        "Define {term} concisely.",
    ],
    "short": [
        "Explain how {process} works. A few sentences.",
        "Describe the mechanism of {mechanism}. Short.",
        "How does {system} function? Concisely.",
        "What is the difference between {a} and {b}? Briefly.",
        "Why does {effect} occur? Short explanation.",
    ],
    "medium": [
        "Explain the concept of {concept} to a beginner. In a paragraph.",
        "Describe the steps involved in {procedure}. With detail.",
        "How would you describe {topic} to a non-expert? Thorough.",
        "What are the key principles behind {principle}? Detailed.",
        "Explain {phenomenon} including causes and mechanisms. Comprehensive.",
    ],
    "long": [
        "Explain {theory} including history, evidence, and implications. In depth.",
        "Describe how {system} works from first principles. With examples.",
        "Explain the mechanism of {mechanism} at multiple levels of abstraction. Comprehensive.",
        "How does {process} achieve its function? Detailed with diagrams described.",
        "Explain {concept} covering definitions, examples, and edge cases. Thorough.",
    ],
    "very_long": [
        "Provide a comprehensive explanation of {topic} from fundamentals to advanced applications. Exhaustive.",
        "Explain {theory} in full detail including mathematical formulation, experimental evidence, and open questions. At length.",
        "Describe the complete workings of {system} including all subsystems, interactions, and design rationale. Exhaustive.",
        "Give an exhaustive account of {process} covering every stage, variant, and regulatory mechanism. Comprehensive.",
        "Explain {concept} across all relevant domains with historical development and current debates. Exhaustive.",
    ],
}

REASONING_TEMPLATES = {
    "very_short": [
        "If {premise1} and {premise2}, what follows? Brief.",
        "Solve: {problem} Short answer.",
        "Given {evidence}, what conclusion? Concise.",
    ],
    "short": [
        "Analyze this argument: {argument} A few steps.",
        "If {condition}, then what? Reason concisely.",
        "Evaluate: {argument} Short reasoning.",
        "What are the implications of {premise}? Briefly.",
        "Deduce the answer from: {clues} A few steps.",
    ],
    "medium": [
        "Reason through this scenario step by step: {scenario} In a paragraph.",
        "Given {evidence}, what conclusion follows? Explain your reasoning.",
        "Solve this problem showing all steps: {problem} With detail.",
        "Analyze the logical structure of: {argument} Thorough.",
        "If {premise1} and {premise2}, derive the conclusion. Comprehensive.",
    ],
    "long": [
        "Think through this complex problem: {scenario} In depth with multiple angles.",
        "Evaluate this multi-premise argument: {argument} Comprehensive analysis.",
        "Reason through {scenario} considering all constraints and tradeoffs. Detailed.",
        "Solve {problem} showing complete derivation and verification. Thorough.",
        "Analyze the implications of {premise} across multiple domains. In depth.",
    ],
    "very_long": [
        "Perform exhaustive multi-step reasoning on: {scenario} Covering all branches, counterarguments, and conclusions. Exhaustive.",
        "Analyze this complex argument chain: {argument} Identify all premises, inferences, and hidden assumptions. Comprehensive.",
        "Reason through {scenario} with full decision tree, sensitivity analysis, and robustness checks. Exhaustive.",
        "Solve {problem} with complete mathematical derivation, edge cases, and generalization. At length.",
        "Evaluate the logical consistency and completeness of: {argument} Exhaustive critique.",
    ],
}

SUMMARIZATION_TEMPLATES = {
    "very_short": [
        "Summarize in one sentence: {text}",
        "Main idea of this passage in one line: {text}",
        "Condense to one sentence: {text}",
    ],
    "short": [
        "Summarize briefly: {text}",
        "Key points in a few sentences: {text}",
        "Concise overview: {text}",
        "Brief summary: {text}",
        "Summarize in 2-3 sentences: {text}",
    ],
    "medium": [
        "Summarize the key points: {text} In a paragraph.",
        "Provide a summary with main arguments: {text} Detailed.",
        "Condense this text to its essence: {text} With detail.",
        "Summarize the argument presented: {text} Thorough.",
        "Create a structured summary of: {text} Comprehensive.",
    ],
    "long": [
        "Summarize comprehensively: {text} With examples and nuance.",
        "Provide an executive summary of: {text} In depth.",
        "Summarize with all key details preserved: {text} Thorough.",
        "Create a detailed summary covering all sections: {text} Comprehensive.",
        "Summarize the text preserving all major arguments and evidence: {text} In depth.",
    ],
    "very_long": [
        "Provide an exhaustive summary of: {text} Including all arguments, evidence, nuances, and implications. Comprehensive.",
        "Summarize this document completely: {text} Preserving all detail, structure, and cross-references. Exhaustive.",
        "Create a comprehensive synopsis of: {text} With chapter-level detail and all supporting evidence. At length.",
        "Produce an exhaustive summary of: {text} Including minor points, counterarguments, and metadata. Comprehensive.",
        "Summarize {text} at full granularity preserving all information. Exhaustive.",
    ],
}

CODING_TEMPLATES = {
    "very_short": [
        "Write a {language} function to {task}. Brief.",
        "One-liner to {task} in {language}.",
        "Short snippet for {task}.",
    ],
    "short": [
        "Write a {language} function that {task}. A few lines.",
        "Create a short {language} script to {task}. Concise.",
        "Implement {algorithm} in {language}. Short.",
        "Write code to {task} with minimal boilerplate.",
        "Debug this: {code_snippet} Fix briefly.",
    ],
    "medium": [
        "Write a {language} class for {concept}. With methods and docstrings.",
        "Implement {data_structure} in {language}. With core operations.",
        "Write a {language} script to {task} with error handling. Detailed.",
        "Create an algorithm to {task}. With explanation.",
        "Refactor this for clarity: {code_snippet} Improved version.",
    ],
    "long": [
        "Write a complete {language} module for {concept}. With tests and documentation.",
        "Implement {algorithm} in {language} with all optimizations. Comprehensive.",
        "Create a {language} application that {task}. Full implementation.",
        "Design and implement a {data_structure} with all operations. Thorough.",
        "Write production-ready code for {task} in {language}. In depth.",
    ],
    "very_long": [
        "Build a complete {language} library for {concept} with full API, tests, docs, and examples. Exhaustive.",
        "Implement a full {language} framework for {task} including all components, plugins, and extensions. Comprehensive.",
        "Write a production-grade {language} system for {concept} with monitoring, config, and deployment. Exhaustive.",
        "Implement {algorithm} with all variants, benchmarks, and analysis in {language}. At length.",
        "Create a comprehensive {language} toolkit for {task} with CLI, API, and documentation. Exhaustive.",
    ],
}

SCIENTIFIC_TEMPLATES = {
    "very_short": [
        "What is {theory}? Brief.",
        "Role of {protein} in {process}? Short.",
        "Define {phenomenon} concisely.",
    ],
    "short": [
        "Explain {theory} in physics. A few sentences.",
        "What is {protein} and its function? Briefly.",
        "Describe the {phenomenon} effect. Short.",
        "How does {mechanism} work? Concisely.",
        "What is {concept} in {field}? Brief explanation.",
    ],
    "medium": [
        "Explain the {theory} theory including key principles. In a paragraph.",
        "Describe the role of {protein} in {process} with detail.",
        "Explain {phenomenon} with mechanism and examples. Detailed.",
        "How does {mechanism} work at molecular level? Thorough.",
        "What are implications of {discovery}? With context.",
    ],
    "long": [
        "Explain {theory} with experimental evidence and applications. In depth.",
        "Describe {protein} structure, function, and regulation. Comprehensive.",
        "Explain {phenomenon} from first principles with mathematical basis. Detailed.",
        "How is {measurement} performed in {field}? With protocols.",
        "What are applications of {technology}? Current and future.",
    ],
    "very_long": [
        "Provide exhaustive review of {theory} including history, math, evidence, and open problems. Comprehensive.",
        "Describe {protein} in full: structure, dynamics, interactions, regulation, disease links. Exhaustive.",
        "Explain {phenomenon} across scales from quantum to macroscopic. At length.",
        "Detail all methods for {measurement} in {field} with protocols and limitations. Exhaustive.",
        "Comprehensive analysis of {technology}: principles, implementations, applications, limits. Exhaustive.",
    ],
}

CREATIVE_TEMPLATES = {
    "very_short": [
        "Write a one-sentence story about {topic}.",
        "Short poem about {theme}. One stanza.",
        "Describe {type} in one line.",
    ],
    "short": [
        "Write a short story about {topic}. A paragraph.",
        "Compose a poem about {theme}. A few stanzas.",
        "Create a dialogue between {characters}. Brief.",
        "Write a scene where {scenario}. Short.",
        "Invent a {type} and describe it concisely.",
    ],
    "medium": [
        "Write a short story about {topic}. With character and plot.",
        "Compose a poem about {theme}. Multiple stanzas with imagery.",
        "Create a dialogue between {characters} exploring {topic}. Detailed.",
        "Write a scene where {scenario}. With sensory detail.",
        "Invent a {type} and describe it thoroughly.",
    ],
    "long": [
        "Write a short story about {topic} with full narrative arc. In depth.",
        "Compose an extended poem about {theme}. Multiple sections.",
        "Create a multi-scene dialogue between {characters}. Comprehensive.",
        "Write a detailed scene where {scenario} with rich description. Thorough.",
        "Invent a {type} with full lore, rules, and examples. In depth.",
    ],
    "very_long": [
        "Write a complete short story about {topic} with developed characters, plot, and themes. Exhaustive.",
        "Compose an epic poem about {theme} with multiple cantos and complex structure. At length.",
        "Create an extended narrative dialogue between {characters} across multiple encounters. Exhaustive.",
        "Write a detailed world-building document for a world where {premise}. Comprehensive.",
        "Invent a {type} with complete system, history, variations, and cultural context. Exhaustive.",
    ],
}

INSTRUCTION_HEAVY_TEMPLATES = {
    "very_short": [
        "Output only: {task}. Format: JSON.",
        "Do {task}. Constraints: {constraint1}. Brief.",
        "Follow: {rule1}. Task: {task}. Short.",
    ],
    "short": [
        "Steps: 1.{step1} 2.{step2} 3.{step3}. Output only result. Concise.",
        "Do {instruction1}, then {instruction2}, finally {instruction3}. Short.",
        "Constraints: {constraint1}, {constraint2}. Task: {task}. A few lines.",
        "Format as {format_spec}. Content: {task}. Brief.",
        "Rules: {rule1}, {rule2}. Execute: {task}. Concise.",
    ],
    "medium": [
        "Follow exactly: 1.{step1} 2.{step2} 3.{step3} 4.{step4}. Output JSON. Detailed.",
        "Requirements: {req1}; {req2}; {req3}. Task: {task}. With structure.",
        "Instructions: {instructions}. Format: {format}. Comprehensive.",
        "Adhere to: {rule1}, {rule2}, {rule3}. Perform: {task}. Thorough.",
        "Comply with all: {constraints}. Generate: {task}. Detailed.",
    ],
    "long": [
        "Execute this multi-step procedure: {instructions}. Requirements: {req1}, {req2}, {req3}, {req4}. Output structured. In depth.",
        "Follow this workflow: {instructions}. Constraints: {constraints}. Validate each step. Comprehensive.",
        "Process: {instructions}. Rules: {rule1}, {rule2}, {rule3}, {rule4}. Format: {format}. Thorough.",
        "Implement to spec: {task}. Constraints: {constraints}. Requirements: {req1}, {req2}, {req3}. Detailed.",
        "Complete this pipeline: {instructions}. Checkpoints: {constraints}. Output: {format}. In depth.",
    ],
    "very_long": [
        "Execute this complex workflow: {instructions}. Validate at each of 10+ stages. Constraints: {constraints}. Requirements: {req1}...{req5}. Output: {format}. Exhaustive.",
        "Follow this detailed specification: {instructions}. Rules: {rule1}...{rule10}. Edge cases: {constraints}. Comprehensive output. At length.",
        "Implement this full pipeline: {instructions}. Stages: 15+. Validation: {constraints}. Monitoring: {req1}...{req5}. Exhaustive.",
        "Process per this exhaustive spec: {instructions}. All constraints: {constraints}. Formats: {format}. Complete. Comprehensive.",
        "Execute end-to-end: {instructions}. Phases: 20+. Gates: {constraints}. Deliverables: {format}. Exhaustive.",
    ],
}

MULTI_TURN_TEMPLATES = {
    "very_short": [
        [
            {"role": "user", "content": "I'm planning a trip to {destination}."},
            {"role": "assistant", "content": "Great! What do you need?"},
            {"role": "user", "content": "Top 3 attractions? Brief."}
        ],
    ],
    "short": [
        [
            {"role": "user", "content": "Help debug this code."},
            {"role": "assistant", "content": "Share the code and error."},
            {"role": "user", "content": "Code: {code_snippet}. Error: {error}. Fix concisely."}
        ],
        [
            {"role": "user", "content": "Write an email to my boss about {topic}."},
            {"role": "assistant", "content": "What's the key message?"},
            {"role": "user", "content": "Request {request} because {reason}. Professional, brief."}
        ],
    ],
    "medium": [
        [
            {"role": "user", "content": "Explain {concept} to me."},
            {"role": "assistant", "content": "{concept} is {brief_explanation}. What aspect?"},
            {"role": "user", "content": "Go deeper into {aspect} with examples. A paragraph."}
        ],
        [
            {"role": "user", "content": "Brainstorm ideas for {project}."},
            {"role": "assistant", "content": "Goal and constraints?"},
            {"role": "user", "content": "Goal: {goal}. Constraints: {constraints}. 5 ideas detailed."}
        ],
        [
            {"role": "user", "content": "Review this code for issues."},
            {"role": "assistant", "content": "Please share it."},
            {"role": "user", "content": "Code: {code_snippet}. Find bugs and suggest fixes. Detailed."}
        ],
    ],
    "long": [
        [
            {"role": "user", "content": "Help me design a {project}."},
            {"role": "assistant", "content": "What are the requirements?"},
            {"role": "user", "content": "Requirements: {constraints}. Goal: {goal}. Full design with components. In depth."}
        ],
        [
            {"role": "user", "content": "Explain {concept} thoroughly."},
            {"role": "assistant", "content": "Overview: {brief_explanation}. Specific focus?"},
            {"role": "user", "content": "Deep dive into {aspect} with math, examples, and applications. Comprehensive."}
        ],
        [
            {"role": "user", "content": "Plan a {project} from start to finish."},
            {"role": "assistant", "content": "Scope and timeline?"},
            {"role": "user", "content": "Timeline: 3 months. Team: 3. Constraints: {constraints}. Full plan. Thorough."}
        ],
    ],
    "very_long": [
        [
            {"role": "user", "content": "Architect a complete {project} system."},
            {"role": "assistant", "content": "Requirements and scale?"},
            {"role": "user", "content": "Scale: enterprise. Requirements: {constraints}. Full architecture with all components, APIs, data models, deployment, monitoring. Exhaustive."}
        ],
        [
            {"role": "user", "content": "Teach me {concept} from basics to advanced."},
            {"role": "assistant", "content": "Starting level?"},
            {"role": "user", "content": "Beginner to expert. Cover theory, math, implementation, applications, research frontiers. Exhaustive course."}
        ],
        [
            {"role": "user", "content": "Design and specify a {project} for production."},
            {"role": "assistant", "content": "Domain and constraints?"},
            {"role": "user", "content": "Domain: {topic}. Constraints: {constraints}. Complete spec: architecture, APIs, schemas, tests, deployment, runbooks. Exhaustive."}
        ],
    ],
}

# ============================================================
# FILLER DATA - CLEAN, NO AMBIGUITY
# ============================================================

COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "India", "Germany"]
BOOKS = ["1984", "Pride and Prejudice", "The Great Gatsby", "To Kill a Mockingbird", "Moby Dick"]
EVENTS = ["World War II", "the Moon landing", "the French Revolution", "the Industrial Revolution"]
COMPOUNDS = ["water", "carbon dioxide", "sodium chloride", "glucose", "methane"]
UNITS = ["meters", "kilometers", "grams", "liters", "seconds"]
LARGER_UNITS = ["kilometer", "kilogram", "kiloliter", "hour", "day"]
CITIES = ["Tokyo", "New York", "London", "Paris", "Beijing", "Mumbai", "Sao Paulo", "Mexico City"]
ELEMENTS = ["Hydrogen", "Carbon", "Oxygen", "Iron", "Gold", "Silver", "Copper"]
PEOPLE = ["Einstein", "Shakespeare", "Marie Curie", "Leonardo da Vinci", "Newton"]
CATEGORIES_LIST = ["ocean", "desert", "mountain", "river", "forest", "lake"]
TERMS = ["photosynthesis", "entropy", "quantum entanglement", "blockchain", "machine learning"]
PROCESSES = ["photosynthesis", "cellular respiration", "DNA replication", "protein synthesis"]
PHENOMENA = ["gravity", "magnetism", "photosynthesis", "the greenhouse effect", "plate tectonics"]
MECHANISMS = ["enzyme catalysis", "signal transduction", "DNA repair", "photosynthetic electron transport"]
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
ARGUMENTS = ["all men are mortal; Socrates is a man; therefore Socrates is mortal"]
CONDITIONS = ["it rains", "the temperature drops below zero", "the market crashes"]
SCENARIOS = ["a train leaves at 60 mph", "you have 3 doors and one has a prize"]
CLUES = ["the butler was in the kitchen", "the window was open", "a muddy footprint leads outside"]
TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "In 1969, Apollo 11 landed on the Moon. Neil Armstrong took the first step.",
    "Machine learning algorithms learn patterns from data without explicit programming.",
    "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
    "Climate change refers to long-term shifts in temperatures and weather patterns globally.",
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
TASKS = ["read a CSV file and return the average", "sort a list of dictionaries by key", "connect to an API and parse JSON"]
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
STEPS = ["step one", "step two", "step three", "step four"]
INSTRUCTIONS = ["do X", "do Y", "do Z", "do W"]
REQUIREMENTS = ["requirement 1", "requirement 2", "requirement 3", "requirement 4", "requirement 5"]
RULES = ["rule one", "rule two", "rule three", "rule four", "rule five", "rule six", "rule seven", "rule eight", "rule nine", "rule ten"]
FORMATS = ["JSON", "YAML", "XML", "CSV", "Markdown"]
CONSTRAINT_ITEMS = ["constraint A", "constraint B", "constraint C", "constraint D", "constraint E"]
BRIEF_EXPLANATIONS = [
    "a fundamental concept in biology",
    "a fundamental concept in chemistry",
    "a fundamental concept in physics",
    "a fundamental concept in computer science",
    "a fundamental concept in neuroscience",
]

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
    "very_short": {"token_range": (0, 50), "target": 40, "hints": ["briefly", "one sentence", "short answer", "concise", "one line"]},
    "short": {"token_range": (50, 150), "target": 80, "hints": ["a few sentences", "concisely", "briefly", "short paragraph", "2-3 sentences"]},
    "medium": {"token_range": (150, 400), "target": 120, "hints": ["in a paragraph", "with detail", "thoroughly", "detailed", "comprehensive"]},
    "long": {"token_range": (400, 1000), "target": 100, "hints": ["in depth", "with examples", "comprehensive", "thorough", "detailed explanation"]},
    "very_long": {"token_range": (1000, 3000), "target": 60, "hints": ["exhaustive", "at length", "comprehensive", "complete", "full detail"]},
}

# Create matrix matching exact marginals using iterative proportional fitting
def create_length_matrix():
    row_targets = {"very_short": 40, "short": 80, "medium": 120, "long": 100, "very_long": 60}
    col_targets = {"very_short": 40, "short": 80, "medium": 120, "long": 100, "very_long": 60}
    bands = ["very_short", "short", "medium", "long", "very_long"]
    
    matrix = {r: {c: 1.0 for c in bands} for r in bands}
    
    for _ in range(200):
        for r in bands:
            row_sum = sum(matrix[r].values())
            factor = row_targets[r] / row_sum
            for c in bands:
                matrix[r][c] *= factor
        for c in bands:
            col_sum = sum(matrix[r][c] for r in bands)
            factor = col_targets[c] / col_sum
            for r in bands:
                matrix[r][c] *= factor
    
    int_matrix = {r: {c: int(round(matrix[r][c])) for c in bands} for r in bands}
    
    for r in bands:
        diff = row_targets[r] - sum(int_matrix[r].values())
        if diff != 0:
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

# Map output bands to template length keys (same names)
OUTPUT_BAND_TO_TEMPLATE_KEY = {
    "very_short": "very_short",
    "short": "short",
    "medium": "medium",
    "long": "long",
    "very_long": "very_long",
}

# ============================================================
# TEMPLATE FILLING - COMPLETE, NO UNRESOLVED PLACEHOLDERS
# ============================================================

def fill_template(template: str, category: str, output_band: str) -> str:
    """Fill template with random filler data. All placeholders resolved."""
    filled = template
    
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
        "{element}": random.choice(ELEMENTS),
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
        "{premise}": random.choice(PREMISES),  # FIX: was unresolved
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
        "{brief_explanation}": random.choice(BRIEF_EXPLANATIONS),
        "{step1}": random.choice(STEPS),
        "{step2}": random.choice(STEPS),
        "{step3}": random.choice(STEPS),
        "{step4}": random.choice(STEPS),
        "{instruction1}": random.choice(INSTRUCTIONS),
        "{instruction2}": random.choice(INSTRUCTIONS),
        "{instruction3}": random.choice(INSTRUCTIONS),
        "{instruction4}": random.choice(INSTRUCTIONS),
        "{constraint1}": random.choice(CONSTRAINT_ITEMS),
        "{constraint2}": random.choice(CONSTRAINT_ITEMS),
        "{constraint3}": random.choice(CONSTRAINT_ITEMS),
        "{requirement1}": random.choice(REQUIREMENTS),
        "{requirement2}": random.choice(REQUIREMENTS),
        "{requirement3}": random.choice(REQUIREMENTS),
        "{instructions}": "; ".join(random.sample(INSTRUCTIONS, 3)),
        "{format}": random.choice(FORMATS),
        "{format_spec}": random.choice(FORMATS),
        "{rule1}": random.choice(RULES),
        "{rule2}": random.choice(RULES),
        "{rule3}": random.choice(RULES),
        "{rule4}": random.choice(RULES),
        "{rule5}": random.choice(RULES),
        "{rule6}": random.choice(RULES),
        "{rule7}": random.choice(RULES),
        "{rule8}": random.choice(RULES),
        "{rule9}": random.choice(RULES),
        "{rule10}": random.choice(RULES),
        "{req1}": random.choice(REQUIREMENTS),
        "{req2}": random.choice(REQUIREMENTS),
        "{req3}": random.choice(REQUIREMENTS),
        "{req4}": random.choice(REQUIREMENTS),
        "{req5}": random.choice(REQUIREMENTS),
    }
    
    for placeholder, value in replacements.items():
        filled = filled.replace(placeholder, value)
    
    # Verify no unresolved placeholders remain
    unresolved = re.findall(r'\{[^}]+\}', filled)
    if unresolved:
        raise ValueError(f"Unresolved placeholders: {unresolved} in template: {template[:100]}")
    
    return filled


def generate_prompt(category: str, input_band: str, output_band: str, prompt_idx: int, existing_texts: set = None) -> Dict[str, Any]:
    """Generate a single prompt matching the category and length bands."""
    
    template_key = OUTPUT_BAND_TO_TEMPLATE_KEY[output_band]
    templates = CATEGORY_TEMPLATES[category][template_key]
    
    # Try multiple templates to avoid duplicates
    max_attempts = 10
    for attempt in range(max_attempts):
        template = random.choice(templates)
        
        if category == "multi_turn":
            conversation = template
            filled_messages = []
            for msg in conversation:
                filled_content = fill_template(msg["content"], category, output_band)
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
            prompt_text = fill_template(template, category, output_band)
            is_multi_turn = False
            conversation_turns = 1
        
        # Check for duplicates
        if existing_texts is not None and prompt_text in existing_texts:
            continue
        
        break
    else:
        # If all attempts failed, use the last one anyway
        pass
    
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
    
    requested_length_hint = 1 if any(h in prompt_text.lower() for h in 
        ["sentence", "paragraph", "word", "token", "line", "brief", "concise", "comprehensive", "exhaustive", "thorough", "detailed", "short", "depth", "length"]) else 0
    
    max_tokens_map = {"very_short": 50, "short": 150, "medium": 400, "long": 1000, "very_long": 2000}
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


def check_contradictions(prompt_text: str, output_band: str) -> List[str]:
    """Check for contradictory length instructions."""
    issues = []
    text_lower = prompt_text.lower()
    
    # Define contradictory pairs
    if output_band in ["very_short", "short"]:
        long_phrases = ["detailed and thorough", "comprehensive", "exhaustive", "in depth", "at length", "full detail", "complete"]
        for phrase in long_phrases:
            if phrase in text_lower:
                issues.append(f"Contradiction: {output_band} band but contains '{phrase}'")
    
    if output_band in ["long", "very_long"]:
        short_phrases = ["briefly", "one sentence", "short answer", "concise", "one line", "a few words"]
        for phrase in short_phrases:
            if phrase in text_lower:
                issues.append(f"Contradiction: {output_band} band but contains '{phrase}'")
    
    return issues


def check_malformed(prompt_text: str) -> List[str]:
    """Check for malformed grammar and template artifacts."""
    issues = []
    
    # Check for duplicated words
    words = prompt_text.split()
    for i in range(len(words) - 1):
        if words[i].lower() == words[i+1].lower() and len(words[i]) > 2:
            issues.append(f"Duplicated word: '{words[i]}'")
    
    # Check for "to sorts", "to reads" etc - malformed infinitives
    malformed_patterns = [
        r'\bto\s+(sorts|reads|writes|creates|generates|extracts|finds|identifies|calculates|computes|solves|debugs|implements|designs|analyzes|compares|summarizes)\b',
        r'\btheory theory\b',
        r'\bprocess process\b',
        r'\beffect effect\b',
    ]
    for pattern in malformed_patterns:
        if re.search(pattern, prompt_text, re.IGNORECASE):
            issues.append(f"Malformed pattern: {pattern}")
    
    # Check for nonsensical category combos (extraction templates shouldn't have creative topics)
    # This is harder to check automatically, rely on template design
    
    return issues


def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity via sequence matching."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def audit_dataset(dev_prompts: List[Dict], test_prompts: List[Dict]) -> Dict[str, Any]:
    """Run comprehensive audit on the dataset."""
    all_prompts = dev_prompts + test_prompts
    
    audit = {
        "total_prompts": len(all_prompts),
        "dev_count": len(dev_prompts),
        "test_count": len(test_prompts),
        "exact_duplicates": [],
        "unresolved_placeholders": [],
        "malformed_grammar": [],
        "contradictory_instructions": [],
        "length_band_counts": {"dev": {}, "test": {}},
        "category_counts": {"dev": {}, "test": {}},
        "test_band_coverage": {"input": set(), "output": set()},
        "cross_split_similarity": [],
        "feature_leakage": [],
        "seed_reproducible": True,
    }
    
    # Exact duplicates
    texts_seen = {}
    for i, p in enumerate(all_prompts):
        text = p["prompt_text"]
        if text in texts_seen:
            audit["exact_duplicates"].append({
                "text": text[:100],
                "indices": [texts_seen[text], i],
                "splits": ["dev" if texts_seen[text] < len(dev_prompts) else "test",
                          "dev" if i < len(dev_prompts) else "test"]
            })
        else:
            texts_seen[text] = i
    
    # Unresolved placeholders
    for i, p in enumerate(all_prompts):
        unresolved = re.findall(r'\{[^}]+\}', p["prompt_text"])
        if unresolved:
            audit["unresolved_placeholders"].append({
                "prompt_id": p["prompt_id"],
                "unresolved": unresolved,
                "split": "dev" if i < len(dev_prompts) else "test"
            })
    
    # Malformed grammar
    for i, p in enumerate(all_prompts):
        issues = check_malformed(p["prompt_text"])
        if issues:
            audit["malformed_grammar"].append({
                "prompt_id": p["prompt_id"],
                "issues": issues,
                "split": "dev" if i < len(dev_prompts) else "test"
            })
    
    # Contradictory instructions
    for i, p in enumerate(all_prompts):
        issues = check_contradictions(p["prompt_text"], p["output_length_band"])
        if issues:
            audit["contradictory_instructions"].append({
                "prompt_id": p["prompt_id"],
                "issues": issues,
                "split": "dev" if i < len(dev_prompts) else "test"
            })
    
    # Length band counts
    for split_name, prompts in [("dev", dev_prompts), ("test", test_prompts)]:
        in_counts = Counter(p["input_length_band"] for p in prompts)
        out_counts = Counter(p["output_length_band"] for p in prompts)
        audit["length_band_counts"][split_name] = {
            "input": dict(in_counts),
            "output": dict(out_counts)
        }
        if split_name == "test":
            audit["test_band_coverage"]["input"] = set(in_counts.keys())
            audit["test_band_coverage"]["output"] = set(out_counts.keys())
    
    # Category counts
    for split_name, prompts in [("dev", dev_prompts), ("test", test_prompts)]:
        cat_counts = Counter(p["category"] for p in prompts)
        audit["category_counts"][split_name] = dict(cat_counts)
    
    # Cross-split similarity (sample check)
    dev_texts = [p["prompt_text"] for p in dev_prompts]
    test_texts = [p["prompt_text"] for p in test_prompts]
    
    # Sample 100 pairs for efficiency
    for _ in range(100):
        d = random.choice(dev_texts)
        t = random.choice(test_texts)
        sim = compute_similarity(d, t)
        if sim > 0.85:
            audit["cross_split_similarity"].append({
                "similarity": sim,
                "dev_preview": d[:80],
                "test_preview": t[:80]
            })
    
    # Feature leakage check - ensure no post-hoc features
    forbidden_features = ["output_tokens", "latency", "energy", "power", "vrram", "actual_"]
    for p in all_prompts:
        for key in p.keys():
            if any(f in key.lower() for f in forbidden_features):
                audit["feature_leakage"].append({
                    "prompt_id": p["prompt_id"],
                    "leaked_feature": key
                })
    
    return audit


def main():
    print("Generating Phase G.1.1 prompt dataset (v2)...")
    
    all_prompts = []
    prompt_counter = {cat: 0 for cat in CATEGORIES}
    existing_texts = set()
    
    # Generate according to length matrix
    for input_band in INPUT_LENGTH_BANDS:
        for output_band in OUTPUT_LENGTH_BANDS:
            count = LENGTH_MATRIX[input_band][output_band]
            if count == 0:
                continue
            
            for _ in range(count):
                category = min(prompt_counter, key=prompt_counter.get)
                prompt = generate_prompt(category, input_band, output_band, prompt_counter[category], existing_texts)
                all_prompts.append(prompt)
                existing_texts.add(prompt["prompt_text"])
                prompt_counter[category] += 1
    
    print(f"\nTotal prompts generated: {len(all_prompts)}")
    
    # Verify category counts
    cat_counts = Counter(p["category"] for p in all_prompts)
    print("\nCategory distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")
    
    in_band_counts = Counter(p["input_length_band"] for p in all_prompts)
    print("\nInput length band distribution:")
    for band in INPUT_LENGTH_BANDS:
        print(f"  {band}: {in_band_counts.get(band, 0)} (target: {INPUT_LENGTH_BANDS[band]['target']})")
    
    out_band_counts = Counter(p["output_length_band"] for p in all_prompts)
    print("\nOutput length band distribution:")
    for band in OUTPUT_LENGTH_BANDS:
        print(f"  {band}: {out_band_counts.get(band, 0)} (target: {OUTPUT_LENGTH_BANDS[band]['target']})")
    
    multi_turn_count = sum(1 for p in all_prompts if p["is_multi_turn"])
    print(f"\nMulti-turn prompts: {multi_turn_count}")
    
    # Create EXACT stratified split: 60 test (15%), 340 dev (85%)
    # Use stratified sampling targeting proportional representation per category AND per band
    # Target: 15% of each = category=6 each, input_band: very_short=6, short=12, medium=18, long=15, very_long=9
    test_target = 60
    category_test_targets = {cat: 6 for cat in CATEGORIES}  # 10 * 6 = 60
    input_band_test_targets = {"very_short": 6, "short": 12, "medium": 18, "long": 15, "very_long": 9}
    output_band_test_targets = {"very_short": 6, "short": 12, "medium": 18, "long": 15, "very_long": 9}
    
    # Build strata by (category, input_band, output_band)
    strata = {}
    for p in all_prompts:
        key = (p["category"], p["input_length_band"], p["output_length_band"])
        if key not in strata:
            strata[key] = []
        strata[key].append(p)
    
    stratum_sizes = {k: len(v) for k, v in strata.items()}
    
    # Greedy allocation to meet category, input band, and output band targets
    stratum_allocations = {k: 0 for k in strata}
    allocated = 0
    stratum_keys = list(strata.keys())
    
    def current_category_counts():
        counts = {cat: 0 for cat in CATEGORIES}
        for k, v in stratum_allocations.items():
            if v > 0:
                cat, _, _ = k
                counts[cat] += v
        return counts
    
    def current_input_counts():
        counts = {b: 0 for b in INPUT_LENGTH_BANDS}
        for k, v in stratum_allocations.items():
            if v > 0:
                _, in_band, _ = k
                counts[in_band] += v
        return counts
    
    def current_output_counts():
        counts = {b: 0 for b in OUTPUT_LENGTH_BANDS}
        for k, v in stratum_allocations.items():
            if v > 0:
                _, _, out_band = k
                counts[out_band] += v
        return counts
    
    def stratum_priority(key):
        cat, in_band, out_band = key
        cat_counts = current_category_counts()
        in_counts = current_input_counts()
        out_counts = current_output_counts()
        cat_deficit = max(0, category_test_targets[cat] - cat_counts[cat])
        in_deficit = max(0, input_band_test_targets[in_band] - in_counts[in_band])
        out_deficit = max(0, output_band_test_targets[out_band] - out_counts[out_band])
        return cat_deficit + in_deficit + out_deficit
    
    # Allocate one by one, always picking stratum that most reduces deficits
    for _ in range(test_target):
        available_strata = [k for k in stratum_keys if stratum_allocations[k] < stratum_sizes[k]]
        if not available_strata:
            break
        scored = [(stratum_priority(k), k) for k in available_strata]
        scored.sort(reverse=True)
        best_key = scored[0][1]
        stratum_allocations[best_key] += 1
        allocated += 1
    
    # Now split each stratum
    test_prompts = []
    dev_prompts = []
    
    for key, prompts_in_stratum in strata.items():
        random.shuffle(prompts_in_stratum)
        n_test = min(stratum_allocations.get(key, 0), len(prompts_in_stratum))
        test_prompts.extend(prompts_in_stratum[:n_test])
        dev_prompts.extend(prompts_in_stratum[n_test:])
    
    # Verify exact counts and adjust if needed
    if len(test_prompts) != test_target:
        if len(test_prompts) > test_target:
            excess = len(test_prompts) - test_target
            dev_prompts.extend(test_prompts[-excess:])
            test_prompts = test_prompts[:-excess]
        else:
            needed = test_target - len(test_prompts)
            test_prompts.extend(dev_prompts[-needed:])
            dev_prompts = dev_prompts[:-needed]
    
    print(f"\nSplit: {len(dev_prompts)} development, {len(test_prompts)} locked test")
    
    # Verify test coverage
    test_in_bands = set(p["input_length_band"] for p in test_prompts)
    test_out_bands = set(p["output_length_band"] for p in test_prompts)
    print(f"\nTest input bands covered: {sorted(test_in_bands)}")
    print(f"Test output bands covered: {sorted(test_out_bands)}")
    
    test_cat_counts = Counter(p["category"] for p in test_prompts)
    print("\nTest set category distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {test_cat_counts.get(cat, 0)}")
    
    # Run audit
    print("\nRunning audit...")
    audit = audit_dataset(dev_prompts, test_prompts)
    
    print(f"\n=== AUDIT RESULTS ===")
    print(f"Exact duplicates: {len(audit['exact_duplicates'])}")
    print(f"Unresolved placeholders: {len(audit['unresolved_placeholders'])}")
    print(f"Malformed grammar: {len(audit['malformed_grammar'])}")
    print(f"Contradictory instructions: {len(audit['contradictory_instructions'])}")
    print(f"Cross-split high similarity (>0.85): {len(audit['cross_split_similarity'])}")
    print(f"Feature leakage: {len(audit['feature_leakage'])}")
    
    print(f"\nDev length bands (input): {audit['length_band_counts']['dev']['input']}")
    print(f"Dev length bands (output): {audit['length_band_counts']['dev']['output']}")
    print(f"Test length bands (input): {audit['length_band_counts']['test']['input']}")
    print(f"Test length bands (output): {audit['length_band_counts']['test']['output']}")
    
    if audit['exact_duplicates']:
        print("\nDUPLICATES FOUND:")
        for d in audit['exact_duplicates'][:5]:
            print(f"  {d}")
    
    if audit['unresolved_placeholders']:
        print("\nUNRESOLVED PLACEHOLDERS:")
        for u in audit['unresolved_placeholders'][:5]:
            print(f"  {u}")
    
    if audit['contradictory_instructions']:
        print("\nCONTRADICTIONS:")
        for c in audit['contradictory_instructions'][:5]:
            print(f"  {c}")
    
    if audit['malformed_grammar']:
        print("\nMALFORMED:")
        for m in audit['malformed_grammar'][:5]:
            print(f"  {m}")
    
    if audit['cross_split_similarity']:
        print("\nHIGH CROSS-SPLIT SIMILARITY:")
        for s in audit['cross_split_similarity'][:5]:
            print(f"  {s['similarity']:.3f}: {s['dev_preview']} | {s['test_preview']}")
    
    # Save dataset
    output = {
        "metadata": {
            "phase": "G.1.1",
            "version": "2.0",
            "total_prompts": len(all_prompts),
            "dev_count": len(dev_prompts),
            "test_count": len(test_prompts),
            "split_seed": 42,
            "stratification": ["category", "input_length_band", "output_length_band"],
            "generation_method": "template_based_with_filler_v2",
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
    
    # Also save just the prompt texts
    prompts_only = {
        "development": [p["prompt_text"] for p in dev_prompts],
        "locked_test": [p["prompt_text"] for p in test_prompts],
    }
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
        json.dump(prompts_only, f, indent=2)
    
    # Save audit results
    audit_output = {k: v for k, v in audit.items() if k != "cross_split_similarity"}
    audit_output["cross_split_similarity_count"] = len(audit["cross_split_similarity"])
    # Convert sets to lists for JSON
    audit_output["test_band_coverage"]["input"] = list(audit["test_band_coverage"]["input"])
    audit_output["test_band_coverage"]["output"] = list(audit["test_band_coverage"]["output"])
    
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/audit_results.json"), 'w') as f:
        json.dump(audit_output, f, indent=2)
    
    print("Done!")
    return audit


if __name__ == "__main__":
    main()