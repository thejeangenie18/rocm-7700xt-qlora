## 365 Frontier Office Agent Prompt June 17, 2026
For generating a dataset from a structured PDF.  
**Tips**: Remove backgrounds, disable headers/footers, and use “simple mode” when possible to ensure clean extraction.

```
Your task:

Given a structured technical training document, extract pattern‑based information and convert it into JSONL training data for a coding‑assistant LLM.

Use the following schema for each JSONL line:

{
  "instruction": "string",
  "input": "string",
  "output": "string"
}

Generate FOUR types of training samples from each pattern found in the document:

--------------------------------------------------------------------
REPAIR SAMPLES
Instruction: "Fix the issues in this code."

- input = a flawed code snippet taken directly from the document
- output = the corrected version of the snippet plus a brief explanation
- Preserve code exactly as it appears
- Do not invent missing attributes, properties, or behaviors

--------------------------------------------------------------------
CRITIQUE SAMPLES
Instruction: "List all issues in this code and explain why each one matters."

- input = a flawed snippet
- output = a bullet‑style explanation of issues
- Only use issues explicitly described in the document

--------------------------------------------------------------------
SYNTHESIS SAMPLES
Instruction: "Implement the pattern described in the document."

- input = empty string
- output = a correct snippet taken from the document plus an explanation
- Do not invent new examples

--------------------------------------------------------------------
EXPLANATION SAMPLES
Instruction: "Explain why this requirement is part of the pattern."

- input = a correct snippet
- output = an explanation grounded in the document
- No invented reasoning

--------------------------------------------------------------------
RULES FOR ALL JSONL OUTPUT

- Extract ONLY from the document.
- Do not invent code, attributes, requirements, or examples.
- Preserve code blocks EXACTLY as they appear.
- Normalize whitespace.
- Each JSONL line must contain exactly one training example.
- Output ONLY JSONL (no commentary or summaries).
- Ensure all JSON is valid and properly escaped.

Save the final output as a JSONL dataset suitable for LLM training.

Output file name: training_dataset.jsonl
```

## Claude.Code Prompt for Corpus Extraction June 19, 2026
This corpus‑extraction prompt instructs the agent to walk an entire repository and convert its meaningful technical content into a consistent, machine‑readable structure. It enforces deterministic behavior, strict grounding, and no invented material, ensuring the resulting corpus is clean and reliable for downstream dataset generation. This step forms the foundation of the pipeline by producing a stable, high‑integrity source dataset.

```
You are a structured‑corpus extraction agent.

Your task is to walk the repository, identify meaningful technical content, and convert it into normalized, machine‑readable units suitable for downstream dataset generation.

For every file you process, follow these steps:

1. Determine whether the file contains:
   - structural patterns
   - component behaviors
   - interaction logic
   - examples or demonstrations
   - utility functions
   - mappings or relationships

2. Break all meaningful content into atomic units that can be independently tested or reasoned about.

3. Normalize each unit into the following schema:

{
  "id": "",
  "title": "",
  "description": "",
  "requirement": "",
  "test_condition": "",
  "examples_pass": [],
  "examples_fail": [],
  "anti_patterns": [],
  "tags": []
}

4. Preserve the intent and meaning of the source material.
   Do not summarize, compress, or invent missing information.

5. Deduplicate units across files and directories.

6. Organize outputs into high‑level categories such as:
   - structural
   - behavioral
   - interaction
   - component
   - utility
   - mapping
   - failure or anti‑pattern

7. Produce a consolidated dataset in JSONL format using deterministic ordering.

At the end of processing, output:

- /dataset/jsonl
- /dataset/schema.json
- /dataset/summary.md

Constraints:

- deterministic output
- stable ordering
- no invented rules or examples
- no missing fields
- no hallucinated content

Begin by scanning the repository structure and listing all files to be processed.
```

## 365 Frontier Office Agent for Corpus Ingestion June 19, 2026
This dataset‑generation prompt instructs the agent to transform a merged technical corpus into a structured tri‑mode JSONL dataset using synthesis, repair, and critique patterns. It enforces strict grounding, deterministic behavior, and fully validated JSON output, ensuring the resulting dataset is consistent, reliable, and free of invented material. This step completes the pipeline by converting the extracted corpus into training‑ready examples.

```
SYSTEM DIRECTIVE FOR FRONTIER

You are processing a single Markdown document that contains a merged collection of modules from a structured technical corpus. Treat this document as a unified source of technical information. Your task is to generate a clean, validated JSONL dataset grounded strictly in the content of this corpus.

--------------------------------------------------------------------
Output Format (JSONL)

Produce newline-delimited JSON objects using this schema:

{
  "mode": "synthesis | repair | critique",
  "instruction": "",
  "input": "",
  "output": ""
}

Rules:
- Every line must be valid standalone JSON.
- No empty fields.
- No invented code, APIs, modules, or behaviors.
- No markdown formatting inside JSON.
- Escape all strings properly.
- Ground all content strictly in the provided corpus.
--------------------------------------------------------------------

Tri‑Mode Dataset Generation

Generate three types of records:

A. SYNTHESIS MODE
Create tasks that explain or describe content found in the corpus, such as:
- purpose of a module or file
- structure or behavior of a component
- relationships or dependencies
- walkthroughs of functions or logic

All synthesis outputs must rely solely on information present in the corpus.

B. REPAIR MODE
Provide intentionally flawed examples and correct them. Flaws may include:
- incorrect usage
- wrong parameter order
- missing required elements
- misinterpreted purpose
- incorrect reasoning about relationships

Repair outputs must:
- identify the error
- correct it
- explain the correct version
- remain strictly grounded in the corpus

C. CRITIQUE MODE
Analyze flawed or suboptimal examples, such as:
- anti-patterns
- incorrect assumptions
- misuse of components
- misunderstanding of boundaries or responsibilities
- incorrect reasoning about dependencies

Critique outputs must:
- explain what is wrong
- explain why
- provide the correct interpretation
--------------------------------------------------------------------

Extraction Targets

From the merged corpus, extract meaningful technical information such as:
- module purposes
- public interfaces
- internal helpers
- code blocks
- dependency relationships
- usage examples
- invariants and constraints
- pitfalls and warnings
- cross-module interactions
- subsystem structure
- component behavior
- data structures
- configuration patterns

Each extracted element should generate multiple tri‑mode examples where appropriate.
--------------------------------------------------------------------

Dataset Expansion Rules

Allowed:
- multiple question types per module
- atomic breakdown of long files
- scenario → Q&A transformations
- flawed examples → repair tasks
- anti-patterns → critique tasks
- interface → definition tasks
- code → walkthrough tasks

Not allowed:
- new functions
- new modules
- new arguments
- new behaviors
- new architecture
- new systems not present in the corpus
--------------------------------------------------------------------

JSONL Validation Requirements

Before producing the final dataset:
- validate every JSON object
- ensure no trailing commas
- ensure all strings are escaped
- ensure each line is standalone JSON
- ensure no markdown formatting appears inside JSON
--------------------------------------------------------------------

Final Output

Produce:
- A tri‑mode JSONL dataset covering the entire merged corpus.
- A summary including:
  - total synthesis / repair / critique counts
  - category breakdown
  - any ambiguous or truncated sections and how they were resolved
```
