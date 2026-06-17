Your task:

Given this 52‑page WhatSock training PDF, extract accessibility patterns and convert them into JSONL training data for a coding‑assistant LLM.

Follow this exact schema for each JSONL line:

{ "instruction": "string", "input": "string", "output": "string" }

Generate FOUR types of training samples from each pattern found in the PDF:

REPAIR SAMPLES Instruction: "Fix the accessibility issues in this code."

    input = a bad code snippet extracted from the PDF
    output = the corrected version of the snippet + a short explanation
    Preserve code exactly as it appears
    No invented ARIA attributes

CRITIQUE SAMPLES Instruction: "List all accessibility issues in this code and explain why each one matters."

    input = a bad snippet
    output = bullet‑style explanation of issues
    Only use issues explicitly described in the PDF

SYNTHESIS SAMPLES Instruction: "Implement an accessible [pattern title] according to the WhatSock training guide."

    input = empty string
    output = a good snippet from the PDF + explanation
    No invented examples

EXPLANATION SAMPLES Instruction: "Explain why [specific requirement] is required for this pattern."

    input = a good snippet
    output = explanation text from the PDF
    No hallucinated reasoning

RULES FOR ALL JSONL OUTPUT:

    Extract ONLY from the PDF. No invented code, ARIA attributes, or requirements.
    Preserve code blocks EXACTLY as they appear in the PDF.
    Normalize whitespace.
    Each JSONL line must be valid JSON and contain exactly one training example.
    Output ONLY JSONL. No commentary, no summaries.
    Save the final output as a JSONL dataset suitable for LLM training.

Output file name: whatsock_training.jsonl
