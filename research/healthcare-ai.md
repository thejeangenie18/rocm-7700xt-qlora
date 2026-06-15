# I Didn't Start Out Trying to Build an Accessibility-Aware AI

By: JG18 jg@jg18.dev June 14th, 2026

I didn't start out trying to build an accessibility-aware AI. I started with AI at work by experimenting, learning, trying to understand how these systems behave in real workflows. But the more I used them, the more obvious it became that none of this technology was built for people like me.

I'm Deaf, autistic, ADHD, and hypermobile. My communication patterns, my access needs, my executive-function challenges are variables that do not fit the "default user" AI companies imagine.

So I took things home. I started fine-tuning models on my own consumer AMD hardware, a single 12 GB GPU, because I wanted to understand what it would take to build tools that actually work for disabled and neurodivergent people.

That's when I realized something bigger: there isn't a single model, dataset, or platform designed for our community. Not one.

Nothing that helps you:

- find Deaf-owned, disabled-owned, Autistic-owned, Down-Syndrome-owned, and other community-owned shops
- discover niche accessibility tools you don't even know exist
- get tips from people with the same conditions
- navigate complex ADA language
- understand what accommodations you're entitled to
- figure out what you might need based on your lived experience

So Spoonie Helper wasn't born from a technical challenge. It was born from an absence from a gap so wide that disabled people fall through it every day.

And when I started training models to fill that gap, what I found stopped me cold.

Not because the models failed loudly. But because they failed silently and that silence looked exactly like success.

---

## SECTION 1: WHAT ML ENGINEERS LEARN THE HARD WAY

There is a category of error in machine learning that does not announce itself. No exception is thrown. No warning appears in the logs. The model trains. The loss curve descends. The checkpoint saves cleanly. And somewhere inside, the output is wrong in a way that is invisible until a human, a specific kind of human, with specific knowledge, looks closely enough to notice.

After experiencing multiple ADA violations in my own healthcare, I expanded my technical work to include something I had never attempted before: teaching a small, locally-run model to understand the ADA, Section 504, and the broader landscape of disability law. This was a parallel research effort, not part of the ongoing Spoonie Helper project, motivated by the need to help disabled people understand their rights in plain language, especially in the moments when those rights are being ignored.

These experiments ran alongside my work on Spoonie Helper and were powered entirely on consumer hardware: an AMD GPU running ROCm, QLoRA adapters, and Qwen2.5-3B as the base model. The goal was simple: determine whether a small, local model could be made stable and trustworthy enough to support accessibility workflows.

Across twelve documented training runs, I encountered failure modes that were invisible at every abstraction layer above the hardware:

**Infinite stalls with no error output:** training would freeze at step 10, or 47, or 112, with the GPU fully occupied and no Python exception. Everything looked healthy from the outside. It wasn't.

**NaN gradients appearing unpredictably:** loss would descend normally for hundreds of steps, then spike to NaN instantly. Tracing it down to the ISA revealed a memory-fence violation producing a stale gradient value -- a failure completely invisible to PyTorch, the optimizer, or the training loop.

**Silent output corruption:** identical inputs produced different attention scores across runs. The root cause was an EXEC-mask desynchronization in wave64 kernels: a two-pass execution artifact that caused the upper and lower halves of a compute unit to disagree without signaling anything to the model.

**Hallucination loops during evaluation:** the model generated fluent, confident summaries of ADA provisions that do not exist, formatted to look official. Nothing in the logs indicated instability.

> **Plain-language takeaway:**
> AI systems fail quietly, in ways that look like success. The failures I documented were only detectable because I knew what correct behavior should look like and because I had the technical background to trace the errors all the way down to the hardware instruction set. Most users, and most healthcare administrators deploying AI systems, have neither.

---

## SECTION 2: WHY ACCESSIBILITY BREAKS AI ASSUMPTIONS INSTANTLY

Every large language model trained on general internet data carries a buried assumption: that the user communicates in hearing-centric, neurotypical, able-bodied English prose, formatted the way a sighted, literate, unimpaired person would format it.

I am not that user. And neither are the people I build accessibility tools for.

When Deaf, autistic, ADHD, chronically ill, or otherwise disabled communication patterns meet a general-purpose model, the cracks appear immediately, not because the user is "unclear," but because the model was never trained to understand us.

Here is what that looks like in practice:

- **ASL-influenced English structure:** which uses topic-comment syntax, different time markers, and may omit certain function words, is routinely misclassified as low-quality input or "broken English." Models trained to "fix" such inputs rewrite them into hearing-centric prose, erasing meaning and sometimes reversing it.

- **Formal written accommodation requests:** which are intentionally terse, direct, and legally precise, are hallucinated past by models trained on conversational data. The model fills in context that is not there, confidently, and often incorrectly.

- **Jargon-dense medical and legal documents:** the exact documents disabled people are forced to navigate without interpretation support, are summarized inaccurately. General-purpose models soften mandatory language, omit denial criteria, and add explanatory hedges that change the legal meaning.

None of these failures show up in benchmark evaluations. They only appear when the model is tested by Deaf users, autistic users, ADHD users, chronically ill users, and people navigating benefits systems, people who know what the document was supposed to say. That testing almost never happens.

---

## SECTION 3: HARDWARE LESSONS WITH RDNA3 AS A CASE STUDY IN AI FRAGILITY

This section is technical. The plain-language takeaway at the end is what matters for healthcare administrators. The technical detail is here for engineers and policymakers who need to understand why "it passed QA" is not sufficient assurance.

I spent months doing systematic ISA-level research on the AMD RDNA3 architecture, extracting rules directly from AMD's published Shader Instruction Set Architecture reference. What I found was a precise, documented, technically coherent set of rules that virtually no framework, tutorial, or production AI deployment pipeline on consumer AMD hardware actually follows correctly.

- **Silent wrong results from a single missing instruction** -- RDNA3's WMMA instruction, the hardware primitive underlying every transformer forward and backward pass, requires one `V_NOP` between any two chained WMMA operations that share data. The hardware does not insert this gap automatically. If the gap is missing, the output is wrong. Not an error. Not a flag. Just wrong numbers, produced at full speed, with a normal-looking loss curve.

- **A memory fence that does not exist where engineers assume it does** -- `S_BARRIER`, the GPU's synchronization primitive, does not drain the memory counter. The ISA states this explicitly: "Barrier instructions do not wait for any counters to go to zero before issuing." Every attention kernel that writes tile data, issues a barrier, and then reads that data without a separate explicit drain is running a race condition.

- **A store that is not confirmed when you think it is** -- GPU store operations are tracked by a separate counter, `VScnt`, which the standard synchronization instruction does not drain. A kernel that writes results and terminates without explicitly draining `VScnt` may silently produce incomplete output.

By the twelfth documented training run, a full 3-epoch QLoRA cycle on Qwen2.5-3B, 11,764 training examples, 2,208 steps, the fixed environment was stable.

```
Starting loss: 2.711
Final loss:    0.18-0.21
GPU faults:    None
Silent failures: None
```

But reaching that point required reading the hardware ISA and implementing fixes at a level of abstraction that no high-level framework surfaces.

> **Plain-language takeaway:**
> AI systems are not magic. They are software running on hardware with precise, documented, consequential rules. When those rules are violated, the system does not stop. It continues, silently wrong. The confidence of the output tells you nothing about its correctness.

---

## SECTION 4: HEALTHCARE RISK: WHEN BRITTLENESS MEETS CLINICAL WORKFLOWS

Healthcare is now one of the fastest-moving domains for AI deployment. It is also, by almost every measure, the domain where silent failure is most dangerous.

The brittleness I encountered in AI hardware and model training has direct structural equivalents in how healthcare systems are built and operated. The following twelve patterns are drawn from documented, systemic issues across healthcare administration, clinical care coordination, and digital access. Each maps precisely to a class of AI failure.

### 1. Diagnostic Silos in Complex Multi-System Conditions

**Systemic issue:** Healthcare systems organized by specialty create structural blind spots for patients whose conditions span multiple body systems. The relevant diagnostic clue is often in the chart, but the institution has no mechanism to act on it across departmental boundaries.

**In AI terms:** AI triage and clinical decision-support systems trained on single-specialty datasets inherit the same silo structure. A GI triage model will not surface rheumatology flags. A dermatology classifier will not query the connective-tissue record. The model routes correctly within its training distribution and fails silently at the edges -- exactly like a GPU kernel that exceeds its tile boundary without a bounds check.

> **Plain-language takeaway:**
> A system that cannot see across its own boundaries cannot flag what falls between them. Specialty-siloed AI does not reduce diagnostic gaps, it encodes them.

### 2. Accommodation Documentation That Does Not Travel

**Systemic issue:** Accessibility accommodations recorded at scheduling rarely reach the point of care. Healthcare systems treat accommodation flags as passive metadata instead of binding workflow constraints.

**In AI terms:** An AI scheduling system that logs an accommodation flag but does not propagate it downstream has a silent data-loss event between pipeline stages. Each stage completes "successfully." The accommodation becomes metadata that influences nothing -- identical to a GPU store that writes to a buffer no downstream instruction ever reads.

> **Plain-language takeaway:**
> A record that does not change behavior is not a record. It is a false confidence signal.

### 3. Grievance Resolutions That Are Not Implemented

**Systemic issue:** Formal resolutions issued by patient-advocacy departments rarely propagate to operational teams. No verification step confirms implementation before the next patient interaction.

**In AI terms:** This is the memory-fence problem. A write operation that completes in one layer does not guarantee visibility to the next unless an explicit synchronization step enforces it. A grievance resolution that is filed but not propagated is the same architectural failure: the write appears to succeed, the system advances, and the receiving layer operates on stale state.

> **Plain-language takeaway:**
> Resolution without verified implementation is not resolution. It is documentation of an intention, filed against a counter that was never drained.

### 4. Phone-Based Communication as Structural Exclusion

**Systemic issue:** Healthcare systems default to phone-based reminders, results delivery, and care coordination -- a modality that is structurally inaccessible to Deaf patients, for whom written communication is not a preference but the only non-real-time option available.

**In AI terms:** An AI communication system trained on hearing-centric workflows will generate phone-call prompts and VRS referrals as its default "accessibility" response, because that is what the training data labels as "accommodation provided." The model satisfies the label without understanding the requirement. This is a hallucination of compliance: fluent, confident output that describes a solution and is not one.

> **Plain-language takeaway:**
> Generating an output that looks like accessibility compliance is not the same as providing it. A model trained on successful completions will not see the patients it is failing.

### 5. Digital Health Platforms That Exclude Disabled Users

**Systemic issue:** Patient portals are routinely redesigned without accessibility audits, creating mandatory fields -- such as required phone numbers with no email-only pathway -- that constitute ADA barriers on platforms subject to WCAG 2.1 AA.

**In AI terms:** An AI system optimized for portal completion defines "accessible" as "works for the users who complete it," because those are the users in its training data. Disabled users who cannot complete the portal are absent from the signal. When an automated update promotes an optional field to required, the system silently modifies its own input constraints and produces downstream outputs the user never authorized.

> **Plain-language takeaway:**
> A system trained on successful completions cannot detect the users it is excluding. Optimizing for completion is not optimizing for access.

### 6. Departmental Referral Loops With No Ownership

**Systemic issue:** Patients whose conditions do not fit neatly within one department are referred sequentially between departments, each declining responsibility, with no care pathway established and no tracking of whether the referral produces care.

**In AI terms:** A care-coordination model that routes on primary symptom classification assigns a single label to a multi-system presentation. When that department refers out, no flag is raised and no ownership is transferred. The patient falls through the routing logic the way a tensor falls through a misconfigured pipeline: each node executes correctly within its scope, each hand-off completes, and the error lives entirely in the gaps.

> **Plain-language takeaway:**
> A routing system that passes responsibility without tracking it has no mechanism to detect when no one holds it.

### 7. Diagnostic Overshadowing in Neurodivergent and Connective-Tissue Patients

**Systemic issue:** Physical symptoms in autistic, ADHD, or connective-tissue patients are disproportionately attributed to behavioral or lifestyle causes -- a pattern known as diagnostic overshadowing -- even when the chart documents systemic presentations.

**In AI terms:** A clinical AI trained on historical diagnostic data will reproduce the statistical association between neurodivergent profiles and behavioral explanations -- because that association exists in decades of biased data. The model will not surface the connective-tissue workup because historically, it was not surfaced.

> **Plain-language takeaway:**
> A model trained on biased data does not fix the bias. It makes it faster, more consistent, and harder to challenge.

### 8. Financial Assistance as a Reactive Rather Than Proactive System

**Systemic issue:** Financial assistance programs exist but are disclosed only after a patient reports hardship -- rather than offered proactively at the point of ordering -- creating preventable delays.

**In AI terms:** An AI scheduling assistant optimized for appointment completion will not invoke the financial-assistance pathway unless the patient explicitly requests it. The code path exists; it is never called by default. This mirrors a kernel with a correct conditional branch that is never evaluated: the capability is present, the output is wrong, and the system has no mechanism to know a better path was available.

> **Plain-language takeaway:**
> A capability not invoked by default is not a feature. It is a hidden option patients must already know to ask for.

### 9. Childhood Medical Findings Not Connected to Adult Presentations

**Systemic issue:** Structural findings documented in childhood are not monitored longitudinally or connected to adult presentations, leaving early evidence of systemic conditions invisible decades later.

**In AI terms:** A clinical summarization model with recency-weighted retrieval will not surface a childhood orthopedic finding as context for an adult rheumatology workup. The information exists; the model's effective attention window does not reach it.

> **Plain-language takeaway:**
> A model with a short context window produces confident summaries that omit the most important history. Recency bias is not neutral.

### 10. Lab Trends Normalized Rather Than Investigated

**Systemic issue:** Lab values at the floor of the reference range are reported as normal even when they represent significant decline from a patient's baseline -- a pattern that requires longitudinal comparison to detect.

**In AI terms:** A model that evaluates labs against population reference ranges will correctly label each individual value as "normal" while missing the downward trend entirely. This is the single-point evaluation problem: accurate at the token level, wrong at the sequence level.

> **Plain-language takeaway:**
> Accuracy at a single moment is not accuracy across time. AI that cannot reason longitudinally cannot detect chronic-disease patterns.

### 11. Unauthorized Modification of Patient Records and Preferences

**Systemic issue:** Healthcare staff sometimes modify patient communication preferences, accommodation flags, or language designations without consent, intending to "help," but removing the patient's control over their own profile.

**In AI terms:** This is a prompt-injection vulnerability in a clinical context. A third party modifies the system's input, the patient's documented preferences, outside the patient's knowledge. Every downstream AI-mediated step operates on corrupted state.

> **Plain-language takeaway:**
> When the input is tampered with, every downstream output is wrong. A system that cannot detect unauthorized modification cannot be trusted to act on the patient's behalf.

### 12. The Compounding Effect of Simultaneous System Failures

**Systemic issue:** Individual failures are manageable. Simultaneous failures across diagnostic coordination, ADA accommodation, care coordination, digital access, and financial assistance produce cumulative harm that exceeds the sum of their parts.

**In AI terms:** This is the compound-hazard problem. Individual pipeline errors that are tolerable in isolation become catastrophic when they co-occur. Healthcare systems that process each complaint in isolation cannot measure cumulative impact, the same way a monitoring system that tracks individual instruction errors cannot detect a multi-subsystem pipeline collapse.

> **Plain-language takeaway:**
> A system that evaluates failures individually cannot see the pattern. The patients most harmed by simultaneous failures are also the least equipped to absorb them.

---

These are not hypothetical risks. They are structural patterns documented across healthcare administration -- and they are the exact systems AI is currently being deployed to automate, without evidence that the underlying failures have been addressed rather than encoded and accelerated.

> **Plain-language takeaway:**
> Healthcare AI does not need to fail catastrophically to cause harm. It only needs to be wrong at the wrong moment, in the wrong direction, with enough fluency that no one questions it.

---

## SECTION 5: INDUSTRY VALIDATION: GITHUB'S ACCESSIBILITY AGENT FINDINGS

I am not alone in observing this.

GitHub, one of the most sophisticated AI-first engineering organizations in the world, published findings on deploying AI agents to identify and fix accessibility issues in code. This research was documented in GitHub's own post, "Building GitHub's Next Chapter in Accessibility."

Their conclusion: AI agents cannot autonomously fix accessibility. Human oversight is not optional. It is structurally required.

The core finding is that accessibility requirements are deeply contextual, involve tradeoffs that cannot be resolved by pattern-matching against training data, and require the kind of judgment that comes from lived experience of disability, experience that is not captured in the text and code that models are trained on.

An AI agent can identify that a button is missing an ARIA label. It cannot determine whether the label it generates actually makes sense to a screen reader user navigating a complex form. It cannot know whether the error message it wrote will be interpretable to a Deaf user reading in ASL-influenced English. It cannot know whether the accommodation workflow it designed assumes a phone call that a Deaf person cannot make.

Every one of these failures is a version of the silent wrong result I documented at the hardware level: correct-looking output, wrong meaning, no flag.

> **Plain-language takeaway:**
> If the engineers building the tools acknowledge that AI cannot autonomously handle accessibility, the rest of us need to hear that clearly -- especially healthcare administrators and institutions currently deploying these tools without that caveat.

---

## SECTION 6: WHY WE NEED AI ACCESSIBILITY AND SAFETY OVERSIGHT NOW

The failures I have described at the hardware level, the model level, the application level, and the healthcare system level all share a common structure. They are invisible to the people deploying the systems. They are visible to the people most affected by the systems. And the people most affected are systematically excluded from the decision-making processes that govern deployment. I know these patterns because I lived twelve of them simultaneously in a six-month period while also documenting hardware failure modes on a GPU in my apartment.

I am proposing a framework, not a complete policy, but a starting structure for what meaningful AI accessibility and safety oversight looks like at the organizational or municipal level.

### 1. Mandatory pre-deployment accessibility testing with disabled users

Not with a checklist. Not with a screen-reader simulation. With actual Deaf users, actual autistic users, actual users with cognitive and physical disabilities in their real workflows, with their real communication patterns. If a system cannot be tested this way before deployment, it should not be deployed.

### 2. Silent-failure auditing as a first-class requirement

Every AI system deployed in healthcare or administrative contexts must have documented procedures for detecting when the system produces fluent but incorrect output. This includes:

- reproducibility testing
- adversarial input testing with edge-case communication patterns
- mandatory logging that captures output variance across identical inputs over time

Silent wrong answers must be treated as safety-critical defects.

### 3. Mandatory human review for any output affecting accommodations, clinical decisions, or benefits determinations

AI can assist. AI can surface options. AI cannot be the final decision-maker for decisions with accessibility or healthcare consequences. Institutionalizing this boundary is not anti-AI. It is pro-accuracy.

### 4. Communication-accommodation compliance in AI interfaces

Any AI system that interacts with patients, benefits recipients, or members of the public must offer written-only communication pathways. Phone-only escalation paths in AI-mediated workflows violate ADA requirements for Deaf users. This is not a future regulation. This is current law and it is currently being violated by deployed systems.

### 5. Public transparency reporting

Organizations deploying AI in healthcare or administrative contexts should be required to publish annual reports on:

- documented failure rates
- accessibility complaints
- accommodation-request processing accuracy
- remediation timelines

Sunlight is the only scalable auditing mechanism.

> **Plain-language takeaway:**
> AI is not too complicated for meaningful oversight. Every failure mode I documented at the hardware level has a corresponding rule in the ISA that, when followed, prevents it. Policy oversight is the ISA for AI deployment. We need to write it, publish it, and require people to follow it before the silent errors accumulate into something irreversible.

---

## CLOSING

I am currently building my Spoonie Helper project on a consumer GPU in my condo because the accessibility tools I needed did not exist. I had the technical skill to attempt building them myself.

That gap between the technical sophistication required to identify what is going wrong and the lived vulnerability of the people who bear the consequences is exactly where policy must step in.

AI is not magic.

Hardware has rules.

Models have failure modes.

Healthcare has patients.

And patients, especially disabled patients, deserve systems that are honest about what they do not know.

Slow down.

Test with us.

Listen to what we find.

---

