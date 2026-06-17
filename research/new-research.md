# BUILT FOR PEOPLE, BETTER FOR MACHINES

## Why Accessibility Improves AI Behavior

With Empirical Evidence from RDNA3 ISA Research · ML Training Logs · Healthcare Systems Analysis

jg@jg18.dev · jg18.dev · Somewhere in a pipeline

June 2026

RDNA3 ISA Research Repository: [github.com/thejeangenie18/rocm-7700xt-qlora](https://github.com/thejeangenie18/rocm-7700xt-qlora)

---

## Abstract

Accessibility built in from the start is a quality multiplier, not a compliance checkbox. It reduces rework costs by up to 100x, improves AI model extraction accuracy by up to 75%, expands audience reach to over 1.3 billion people with disabilities worldwide, and produces documentation that outperforms inaccessible alternatives on every measurable axis. These are not separate benefits. They are the same benefit, expressed at different layers of the same system.

This thesis advances a unified argument: accessibility reduces ambiguity, and ambiguity reduction is the mechanism by which both human readers and machine systems achieve accurate, reliable output. The evidence for this claim is drawn from six domains - healthcare, software engineering, UX design, AI/ML, documentation pipelines, and government services - and is grounded in empirical data from twelve documented ML training runs on consumer AMD RDNA3 hardware. Those training runs revealed, at the instruction-set level, the same principle that governs accessible content design: when structure is absent, systems do not fail loudly. They continue, silently wrong.

The cost structure of deferred accessibility is well-documented. The market opportunity is quantified. The performance gains for both human users and machine systems are reproducible. Accessibility is not a feature. It is the foundation on which everything else either holds or collapses.

> **KEY FINDING:**
> Accessibility built in from the start costs less, reaches more people, performs better with AI tools, produces reproducible and auditable artifacts, and eliminates entire categories of rework. There is no budget, timeline, or team size at which skipping it makes financial sense.

**KEYWORDS**

Accessibility-first design · WCAG compliance · ambiguity reduction · AI/ML structural requirements · RDNA3 ISA · QLoRA training · healthcare AI · retrofit cost · 1-10-100 rule · semantic markup · silent failure modes · disabled-led design

---
<nav aria-label="Table of contents">

## Table of Contents

- **[§1 Introduction – The Argument in Full](ca://s?q=Go_to_section_1_Introduction)**
- **[§2 The Retrofit Tax – Why Deferred Accessibility Always Costs More](ca://s?q=Go_to_section_2_Retrofit_Tax)**
- **[§3 Accessibility as Structure – Semantic Markup as Shared Infrastructure](ca://s?q=Go_to_section_3_Accessibility_as_Structure)**
- **[§4 Structure Is Signal – How Accessibility Improves AI Behavior](ca://s?q=Go_to_section_4_Structure_Is_Signal)**
- **[§5 The Economic Case – Quantified Returns Across Six Domains](ca://s?q=Go_to_section_5_Economic_Case)**
- **[§6 Real‑World Evidence – Healthcare, Engineering, and Government](ca://s?q=Go_to_section_6_Real_World_Evidence)**
- **[§7 Hardware as Proof – RDNA3 ISA and the Ambiguity Penalty](ca://s?q=Go_to_section_7_Hardware_as_Proof)**
- **[§8 AI Brittleness in Clinical Contexts – When Silent Failures Meet Healthcare](ca://s?q=Go_to_section_8_AI_Brittleness)**
- **[§9 A Framework for Oversight – Policy as Instruction Set](ca://s?q=Go_to_section_9_Framework_for_Oversight)**

### Section 10 (Expanded)

- **[§10 Conclusion – The Constraint That Makes Everything Else Work](ca://s?q=Go_to_section_10_Conclusion)**
  - **[§10.1 Original Prompt (Reproducibility Reference)](ca://s?q=Go_to_section_10_1_Original_Prompt)**
  - **[§10.2 Evidence Screenshots](ca://s?q=Go_to_section_10_2_Evidence_Screenshots)**
    - **[Extraction Task Definition — Strict, Deterministic, No‑Hallucination Constraints](ca://s?q=Go_to_screenshot_1)**
    - **[Automated PDF Parsing and JSONL Conversion Sequence](ca://s?q=Go_to_screenshot_2)**
    - **[Automated JSONL Generation and Validation Sequence](ca://s?q=Go_to_screenshot_3)**
    - **[Extracted Pattern Summary and Sample‑Type Breakdown](ca://s?q=Go_to_screenshot_4)**
    - **[Advanced Accessibility Pattern Extraction (Items 15–30)](ca://s?q=Go_to_screenshot_5)**
    - **[JSONL Dataset View – Verbatim Extraction and Balanced Sample Types](ca://s?q=Go_to_screenshot_6)**

- **[References](ca://s?q=Go_to_References)**
- **[Appendix A – Empirical Evidence: RDNA3 Training Logs](ca://s?q=Go_to_Appendix_A)**
</nav>
---

## §1 Introduction: The Argument in Full

Accessibility is not charity. It is the single highest-ROI design decision available at the point of project inception, and the only one that simultaneously reduces legal exposure, expands the addressable market, improves AI and machine-parsing performance, and cuts downstream maintenance costs. Every other architectural decision affects a subset of those outcomes. Accessibility affects all of them at once.

The central argument of this thesis is precise: accessibility reduces ambiguity, and ambiguity reduction is the mechanism by which both human readers and machine systems achieve accurate, reliable output. This is not a metaphor. It is a structural claim, and it holds at every layer of the stack - from the semantic markup of a web page, to the instruction-stream typing of a GPU compute unit, to the memory-fence sequencing of a machine-learning kernel. In each case, the system that receives well-typed, unambiguous input produces correct output. The system that receives ambiguous input continues operating and produces output that looks correct but is not.

This thesis makes two arguments in parallel: a financial one and a technical one. The financial argument is grounded in six domains of evidence - healthcare, software engineering, UX design, AI/ML, documentation pipelines, and government services. The technical argument is grounded in empirical data from twelve documented ML training runs on consumer AMD RDNA3 hardware. Those runs revealed, at the instruction-set level, the same failure mode that inaccessible content produces at the application level: silent wrong results, invisible to every abstraction layer above the point of origin.

The scale of the market begins the financial case. Over 1.3 billion people worldwide live with a disability, representing trillions in collective spending power. Forrester Research documented what organizations actually experienced after implementing accessible technologies: nearly 80% reported improved customer experience, and 67% cited risk mitigation as a primary driver of their accessibility investment. These are not edge-case outcomes. They are the documented returns on a decision that most organizations still treat as optional.

The author of this thesis is Deaf, autistic, ADHD, and hypermobile. The communication patterns, access needs, and executive-function challenges that characterize this profile are variables that do not fit the 'default user' most AI systems are designed for. That gap - between the technical sophistication required to identify what is going wrong and the lived vulnerability of the people who bear the consequences - is precisely where both engineering and policy must intervene.

The RDNA3 ISA research documented here emerged from a direct attempt to build tools that work for disabled and neurodivergent people on consumer hardware. What began as a question about why FlashAttention was hanging became a systematic extraction of ISA-level rules that virtually no framework, tutorial, or production AI deployment pipeline on consumer AMD hardware actually follows correctly. The parallel to accessibility is not incidental. It is structural. Both domains share the same failure mode: ambiguity inserted at the authoring stage propagates downstream and compounds. Structure is not cosmetic. It is the signal the system runs on.

[Forrester Research, 2016; nascio.org, 2024; accessibility.works, 2025; AMD RDNA3 ISA Reference, 2023]

---

## §2 The Retrofit Tax: Why Deferred Accessibility Always Costs More

Most organizations treat accessibility as a post-launch audit item, something to address once the product is live and complaints have started arriving. That sequencing decision is not a minor process inefficiency. It is the root cause of every cost explosion, every lawsuit, and every six-month remediation sprint that consumes the roadmap for the following year.

The problem is not that organizations fail to care about accessibility. The problem is that they defer it to a phase where fixing it is structurally expensive. Understanding why requires examining the documented cost structure of deferred work - and the legal exposure that accelerates it.

### 2.1 The 1-10-100 Rule

WebAIM's analysis of post-launch remediation is direct: fixing accessibility issues after launch costs approximately 10x more than addressing them during initial development. The further downstream a defect travels, the more systems it has touched, the more documentation it has contaminated, and the more expensive it becomes to unwind. This is not a phenomenon unique to accessibility - it is the documented behavior of any architectural defect that propagates through a system before being caught.

The structural cost model is known as the 1-10-100 rule: a defect costs 1x to fix at the design phase, approximately 10x to fix during implementation or testing, and approximately 100x to fix under legal or post-release pressure. Accessibility defects follow this rule with particular severity because they tend to be architectural, embedded in templates, component libraries, and data schemas rather than isolated to a single feature.

A missing ARIA role in a component library is not one defect. It is every instance of that component, across every page, across every user session, compounding silently until someone files a complaint or a lawsuit.

> **HARD CONSTRAINT:**
> Every sprint that ships inaccessible code is a future sprint of debt. The 1-10-100 rule is not a theory - it is the documented cost structure of deferred accessibility work across industries. Accessibility is not a feature toggle. It is a load-bearing structural property. Defer it and the debt compounds; it does not wait.

### 2.2 Legal Exposure Is Accelerating

Digital accessibility lawsuits in the U.S. surged more than 300% from 2018 to 2023. A single web accessibility lawsuit can cost up to $350,000 in legal fees alone, before remediation costs, before reputational damage, and before the forced architectural overhaul that typically follows a settlement. This is not a niche compliance risk manageable by a legal checkbox. It is the mainstream market reality, and it is accelerating as AI search tools increasingly penalize unstructured content at the discovery layer.

The solution is not more auditing after the fact. It is structural: accessibility must be embedded in the information architecture from the first keystroke. The same principle holds at the hardware level. RDNA3's ISA rules are not suggestions appended to a working system. They are the conditions under which the system produces correct output at all. Violate them and the system does not stop. It continues, silently wrong. The retrofit tax applies equally to GPU kernels and to web applications: the cost of fixing a structural violation after the system is in production is always a multiple of the cost of building it correctly from the start.

The following table illustrates the cost differential across five common accessibility features. The multiplier column reflects documented remediation cost ratios from industry research.

| Feature | Build-In Cost | Retrofit Cost | Multiplier |
|---|---|---|---|
| Color contrast | Design token choice (~0 dev time) | Component library audit + full redesign pass | 10-30x |
| ARIA labels | Added at build (~minutes per component) | Post-hoc instrumentation of entire UI surface | 15-25x |
| Alt text on images | Written at upload (~seconds per image) | Audit + retroactive writing for all archived images | 20x |
| Accessible PDF template | Template configured once; all exports inherit | Document-by-document recreation and tagging | 5-15x |
| Form error handling | Built into initial UX spec | UX research + redesign + full retest cycle | 8-20x |

**Table 1.** Build-in vs. retrofit cost comparison by accessibility feature. Multiplier reflects documented remediation cost ratio.

[WebAIM via eajournals.org; inspekter.com, 2025]

---

## §3 Accessibility as Structure: Semantic Markup as Shared Infrastructure

Accessible content is structurally sound content. WCAG guidelines - proper HTML semantics, heading hierarchy, alt text, ARIA roles, descriptive link text - are not bolt-on features added after the real architecture is complete. They define the information architecture that both humans and machines depend on. When accessibility is omitted, it is not only users with disabilities who suffer the gap. It is every downstream system that attempts to parse, index, summarize, or route the content.

This is the core insight that connects accessibility to AI behavior: the structural properties that make content accessible to a screen reader user are the same structural properties that make content parseable by a search crawler, a large language model, and a voice interface. These are not separate optimizations. They are the same optimization, applied at the same layer of the stack, producing benefits that propagate upward through every system that consumes the content. Section 4 quantifies those benefits in measurable AI performance terms.

### 3.1 Semantic Structure as Shared Infrastructure

A 2025 SEMrush analysis of 10,000 websites quantified the traffic consequence of this structure gap directly: accessible, WCAG-compliant sites gained 23% more organic search traffic and ranked for 27% more keywords than non-compliant peers. Non-accessible sites lost up to 20-30% of their traffic to emerging AI search tools that favor well-structured content, not because the content was worse, but because the structure made it unreadable to the systems that now mediate discovery. The content existed. The signal did not.

Accessible design also functions as a data quality strategy. Semantic markup and standardized labels reduce ambiguity, enabling both humans and machines to parse content accurately and avoid interpretation errors. When every heading is a heading, every table cell has a defined scope, and every link communicates its destination, the error surface for both human misreading and machine misclassification drops to near zero.

Ambiguity is not a neutral property of content. It is a cost that every downstream consumer pays, repeatedly, at every point of contact.

[accessibility.works, 2025; 365 Researcher ROI Digest, 2025]

### 3.2 The Hardware Parallel: RDNA3 Typed Instruction Streams

The AMD RDNA3 Instruction Set Architecture provides a precise hardware-level illustration of this principle. RDNA3's compute units are optimized for typed, structured instruction streams - dual-issue scalar and vector pipelines that perform most efficiently when instruction types are clearly differentiated and predictable. Ambiguous or mixed instruction sequences degrade throughput, not by producing errors, but by forcing the hardware to resolve ambiguity at runtime. This is the same throughput penalty that AI systems pay when parsing unstructured content.

The most instructive specific case is VOPD (dual-issue vector operation). VOPD is a wave32-only instruction that doubles throughput for specific VALU pairings. In wave64 mode, VOPD instructions are silently skipped - no error, no diagnostic, no half-speed warning. The instruction simply does not execute. The kernel continues. The output looks normal. The throughput loss is invisible until someone with the right knowledge examines the profiler output.

This is the hardware equivalent of unstructured content reaching an AI parser: the system continues, produces output, and the degradation is invisible until someone with the right knowledge looks closely enough to notice. In both cases, the system was given input that did not match the format it was optimized for. In both cases, the system did not stop. In both cases, the failure was silent. And in both cases, the fix is the same: provide typed, structured input that matches the format the system is designed to consume.

[AMD RDNA3 ISA Reference, 2023; RDNA3 ISA Research Compilation, June 2026]

> **Engineering Note:**
> When building an AT product dataset for local inference, structuring the JSON with consistent field names, descriptive labels, and no ambiguous abbreviations directly improved TinyLlama's recommendation accuracy on a Pi 3B+ inference endpoint. Structure is not overhead - it is the signal the model runs on. The same principle applies at every scale, from embedded inference hardware to enterprise LLM pipelines.

---

## §4 Structure Is Signal: How Accessibility Improves AI Behavior

The claim that accessibility improves AI behavior is not a design aspiration. It is a measurable, reproducible, quantified finding. AI models extract meaning from structure. Accessible documents are semantically structured documents. That structure is machine-readable signal, and the performance difference between structured and unstructured input is not marginal. It is the difference between a model that reasons correctly and one that hallucinates, misattributes, or silently drops data.

The evidence below draws on two independent lines of research: structured-data retrieval studies that measure AI accuracy directly, and the RDNA3 ISA research that demonstrates the same failure mode at the hardware level. Together, they establish that ambiguity is not a content problem. It is an architectural one.

### 4.1 Quantified Performance Gains from Structured Input

A 2025 XBRL study on large language model retrieval from corporate financial reports measured this directly. When financial data was in structured, accessible XBRL format versus unstructured text or HTML, the error rate for key metrics dropped from 15-29% (plain text/HTML) to approximately 7-9% (XBRL), a 40-75% reduction in extraction errors. The most dramatic result came from scaling errors - misreading units such as 'millions' vs. 'billions': these fell from approximately 8% in unstructured text to just 0.11% with XBRL's context tags. The only variable was structure. The content was identical. The signal was not.

A 2025 systematic review of knowledge graph-augmented LLMs confirmed the same directional finding at a broader level: integrating structured knowledge graphs significantly improves LLM accuracy and reduces hallucinations, while boosting reasoning capability and explainability. The mechanism is identical to what the XBRL study found - structure reduces the search space the model must navigate, increasing confidence and decreasing error. Ambiguity is not a neutral property of input. It is a tax on every inference operation the model performs.

Search engines and AI models parse content through mechanisms directly analogous to screen readers - both require the same structural signals to route, weight, and interpret content accurately. Building for a screen reader user and building for an AI agent are not different tasks. They are the same task, applied to the same structural layer, producing benefits that propagate to every consumer of the content.

[xbrl.us, 2025; aclanthology.org, 2025; accessibility.works, 2025]

| Accessibility Feature | AI Benefit When Present | Degradation When Absent |
|---|---|---|
| Semantic heading hierarchy (H1 to H3) | Correct document outline extraction; accurate section weighting | Flat text blob; no structure signal; misranked content |
| Table with scoped headers | Accurate row/column parsing and data attribution | Cell values misattributed; row/column confusion |
| Alt text on charts and images | Chart data available as parseable text | Image content entirely opaque to the model |
| Descriptive link text | Link intent and destination parseable | 'click here' carries zero semantic value |
| Logical reading order | Content parsed in author-intended sequence | Layout order vs. reading order mismatch; mangled output |
| ARIA landmark roles (nav, main, aside) | Document regions identifiable for differential weighting | Undifferentiated content; navigation chrome treated as body text |
| XBRL/structured data format [xbrl.us, 2025] | 40-75% fewer extraction errors | Scaling errors, unit misreads, silent data loss |

**Table 2.** Accessibility features and their direct AI performance impact - with and without structural compliance.

> **KEY FINDING:**
> Accessible design is increasingly synonymous with AI/SEO optimization. Both search engines and AI agents parse content much like screen readers. Building for accessibility is building for the machines that are increasingly the first readers of your content, before any human ever reaches it. Accessible structure is not a human accommodation. It is the input format the entire modern information stack is optimized for.

### 4.2 The RDNA3 Memory Model: Ambiguity as Architectural Failure

The RDNA3 ISA provides a technically precise model for understanding what happens when ambiguity is present at the structural level. RDNA3 tracks outstanding memory operations through four separate counters: VMcnt (vector memory loads), VScnt (vector memory stores), LGKMcnt (LDS, scalar memory, and flat operations), and EXPcnt (exports). These counters are entirely separate - S_WAITCNT does not drain VScnt. A kernel that writes results and terminates without explicitly draining VScnt may produce incomplete output silently.

The write appears to succeed. The system advances. The receiving layer operates on stale state. This is the hardware equivalent of an accessibility gap in a documentation pipeline. In both cases, the failure is invisible at every abstraction layer above the point of origin. It only becomes visible when someone with the right knowledge - a hardware engineer reading ISA output, or a Deaf user reading a document that was supposed to accommodate them - looks closely enough to notice.

The ISA is unambiguous on S_BARRIER: 'Barrier instructions do not wait for any counters to go to zero before issuing.' A barrier after LDS stores without a preceding lgkmcnt(0) is a data race, not a synchronized operation. The same mismatch occurs in accessibility: a compliance checkbox that does not change behavior is not compliance. It is a false confidence signal.

[AMD RDNA3 ISA Reference, 2023; RDNA3 ISA Research Compilation, June 2026]

---

## §5 The Economic Case: Quantified Returns Across Six Domains

Early accessibility implementation is not a cost. It is cost avoidance at scale. Every dollar invested at the design phase eliminates $10-$100 in downstream remediation, litigation, extended QA cycles, and support volume. The math does not change with team size, budget bracket, or industry. The only variable is when you pay, and the later you pay, the more you pay.

The following subsections present quantified returns across three dimensions: development efficiency and code quality, user experience and revenue, and market scale and legal risk. Each dimension draws on independent research, and the findings are consistent across all of them.

### 5.1 Development Efficiency and Code Quality

Teams that embedded accessibility into development cycles from the beginning saw a ~27% improvement in development cycle efficiency and a ~35% reduction in long-term maintenance costs relative to teams that retrofitted accessibility after initial release. These gains are a direct consequence of standardized, semantic code - the same discipline that produces accessible output also produces cleaner architecture with less accumulated debt.

Accessibility is not an additional constraint on top of good engineering. It is the constraint that enforces good engineering. Quality metrics track in the same direction: organizations practicing inclusive engineering reported ~15% fewer JavaScript errors, ~22% fewer runtime exceptions, and a ~25% reduction in page load times. Accessible code is cleaner code. The constraints of accessibility - clear roles, predictable structure, deterministic state - impose the same discipline that produces robust systems generally.

[eajournals.org, 2025]

### 5.2 User Experience and Revenue

User experience outcomes are equally concrete. Companies implementing comprehensive accessibility guidelines saw a 63% increase in overall user satisfaction and 37% higher task completion rates across digital platforms. Task completion rate is not a soft metric. It is a direct revenue driver - every user who cannot complete a task is a transaction that did not happen.

The revenue case is documented at individual company level. One financial services firm attributed an estimated £13 million in added annual revenue to conversion improvements following WCAG implementation across their platform. In e-commerce, accessible interfaces reduced customer abandonment rates by ~13% and increased successful transactions by ~17%. Accenture's 'Getting to Equal' disability inclusion study measured the enterprise-level performance differential: companies leading in accessible and inclusive practices achieve 28% higher revenue and 30% better profit margins than their peers.

[eajournals.org, 2025; Accenture, 2018]

### 5.3 Market Scale and Legal Risk

Over 1.3 billion people worldwide live with a disability, collectively controlling trillions in spending power. Organizations that build accessibly from the start gain access to this customer base without additional spend. Organizations that do not build accessibly exclude it entirely, and often do not discover that exclusion until it appears as a lawsuit, a churn spike, or a support volume anomaly that cannot be explained by any other variable.

The legal risk is not static. Digital accessibility lawsuits in the U.S. surged more than 300% from 2018 to 2023. A single web accessibility lawsuit can cost up to $350,000 in legal fees alone, before remediation costs, before reputational damage, and before the forced architectural overhaul that typically follows a settlement. This is the 1-10-100 rule expressed in litigation: the cost of fixing a structural violation after it has been litigated is always a multiple of the cost of building it correctly from the start.

[nascio.org, 2024; eajournals.org, 2025]

---

## §6 Real-World Evidence: Healthcare, Engineering, and Government

The economic case for accessibility-first design is not theoretical. It is documented in operational budget lines, patient outcomes, defect rates, and revenue figures across multiple sectors. Two sectors - healthcare and software engineering - illustrate the compounding return on early accessibility investment with particular clarity, because in both cases the cost of inaccessibility is measured in concrete, auditable outcomes rather than estimates.

### 6.1 Healthcare: Accessibility as Operational Infrastructure

The U.S. Social Security Administration's shift to accessible online services produced a direct and measurable operational result: savings of millions in operational costs as in-person visits and call center volume dropped, particularly for seniors and users with limited digital proficiency. Accessibility did not add cost to service delivery. It reduced it, by enabling users to complete tasks independently that previously required staff assistance.

Patient portal accessibility correlates with concrete clinical and economic outcomes. Portals that are navigable and accessible correlate with fewer office visits, better medication adherence, and higher patient satisfaction, reducing overall care costs across the care cycle. The mechanism is direct: if a patient can access their records, manage their prescriptions, and communicate with their care team without requiring staff assistance, the staff time and facility resources associated with those interactions are recaptured.

Accessibility is not an accommodation layered on top of the system. It is the property that makes the system function as intended.

The Minnesota Department of Health and Human Services provides a case study in proactive accessibility embedding. By incorporating accessibility requirements into an online licensing system from the project's outset, not as a post-launch audit, the project avoided expensive retrofits and delivered improved experience for all users. Project leaders reported clearer documentation, better vendor performance, and smoother overall project delivery as direct outcomes of the proactive approach. The accessibility requirement clarified expectations, reduced ambiguity, and produced better procurement outcomes across the board.

In digital banking, one accessible self-service implementation reduced call center workload by up to 40% and lowered operational costs accordingly. The reduction was not attributable to fewer customers - it was attributable to customers being able to complete tasks independently. Accessibility, in this context, is not a cost center. It is a capacity multiplier.

[nascio.org, 2024; adoc-studio.app, 2025]

> **Field Note:**
> At Denver's Department of Transportation and Infrastructure (DOTI), pushing citywide interpreter access documentation and mandatory training was not an abstract policy exercise - it had direct operational outcomes. Accessible documentation reduced clarification requests, improved compliance, and created a paper trail that held up under audit. Accessible documentation is not overhead. It is the artifact that eliminates the follow-up call, the clarification email, and the audit finding.

### 6.2 Software Engineering: Accessibility as Code Quality

Teams practicing proactive accessibility saw a ~27% improvement in development cycle efficiency and ~35% reduction in long-term maintenance costs. These gains are not attributable to accessibility work specifically - they are attributable to the discipline that accessible development requires. Semantic, well-structured code is inherently more maintainable, more testable, and more resistant to regression than code written without structural constraints. Accessibility is the constraint that enforces the discipline. Remove the constraint and the discipline tends to erode.

Inclusive engineering correlates with fewer overall defects across the entire codebase: ~15% fewer JavaScript errors, ~22% fewer runtime exceptions, and a ~25% reduction in page load times. The discipline of writing accessible code - clear roles, predictable structure, deterministic state - inherently produces cleaner, more robust architectures. These are not separate wins from the accessibility work. They are the same win, because the structural properties that make code accessible are the same structural properties that make code maintainable.

The innovation spillover from accessibility work has historically produced the most widely adopted features in consumer technology. Voice interfaces, closed captions, responsive design, predictive text, and autocomplete all originated as accessibility solutions before becoming standard features used by everyone. The pattern is consistent: solving for the constrained use case produces solutions that generalize. The constraint is not a limitation. It is the design pressure that produces better outcomes for all users.

[eajournals.org, 2025]

---

## §7 Hardware as Proof: RDNA3 ISA and the Ambiguity Penalty

The AMD RDNA3 Instruction Set Architecture provides the most technically precise evidence available for the central claim of this thesis: that ambiguity, inserted at the structural level, propagates downstream and compounds. The ISA documents a set of rules that, when followed, produce correct output at rated throughput. When violated, the system does not stop. It continues, silently wrong. The confidence of the output tells you nothing about its correctness.

This is not a metaphor for accessibility failures. It is the same failure mode, expressed at the hardware level. The research documented in this section emerged from a direct attempt to run QLoRA training on an RDNA3 GPU (Radeon RX 7700 XT) using ROCm 7.2.1. What started as a question about why FlashAttention was hanging became a systematic extraction of ISA-level rules that virtually no framework, tutorial, or production AI deployment pipeline on consumer AMD hardware actually follows correctly. The findings are documented across twelve training runs spanning May-June 2026, and they are reproducible.

### 7.1 Silent Failure as Architectural Property

There is a category of error in machine learning that does not announce itself. No exception is thrown. No warning appears in the logs. The model trains. The loss curve descends. The checkpoint saves cleanly. And somewhere inside, the output is wrong in a way that is invisible until a human - a specific kind of human, with specific knowledge - looks closely enough to notice.

Across twelve documented training runs, the following failure modes were encountered, all invisible at every abstraction layer above the hardware:

- **Infinite stalls with no error output:** training would freeze at step 10, or 47, or 112, with the GPU fully occupied and no Python exception. Everything looked healthy from the outside. It was not.
- **NaN gradients appearing unpredictably:** loss would descend normally for hundreds of steps, then spike to NaN instantly. Tracing it down to the ISA revealed a memory-fence violation producing a stale gradient value - a failure completely invisible to PyTorch, the optimizer, or the training loop.
- **Silent output corruption:** identical inputs produced different attention scores across runs. The root cause was an EXEC-mask desynchronization in wave64 kernels - a two-pass execution artifact that caused the upper and lower halves of a compute unit to disagree without signaling anything to the model.
- **Hallucination loops during evaluation:** the model generated fluent, confident summaries of ADA provisions that do not exist, formatted to look official. Nothing in the logs indicated instability.

> **PLAIN-LANGUAGE TAKEAWAY:**
> AI systems fail quietly, in ways that look like success. The failures documented here were only detectable because the researcher knew what correct behavior should look like and had the technical background to trace the errors all the way down to the hardware instruction set. Most users - and most healthcare administrators deploying AI systems - have neither.

### 7.2 ISA Hazards and Their Accessibility Parallels

The following ISA-level hazards were documented through systematic extraction of AMD's published RDNA3 Shader Instruction Set Architecture reference. Each has a direct structural parallel in accessibility design, not as analogy, but as the same failure mode expressed at a different layer of the stack.

#### WMMA Dependency Hazard (ISA §5.4)

RDNA3 requires a bubble between dependent WMMA instructions: 'Dependent WMMA instructions must be separated by at least one VALU or V_NOP.' Triton does not insert this bubble, causing silent corruption in matmul and attention kernels.

The accessibility parallel: a document pipeline that does not enforce heading hierarchy between sections produces output that looks structured but is not - downstream parsers, whether human or machine, receive corrupted signal with no indication that anything is wrong.

#### WMMA Ignores EXEC Masking (ISA §5.4.3)

RDNA3 forces EXEC = all-ones for WMMA: 'WMMA instructions execute with EXEC forced to all active lanes.' Triton uses predication for partial tiles, producing incorrect results.

The accessibility parallel: an ARIA role that is present in the markup but not propagated to the accessibility tree is the same architectural failure - the label exists, the system advances, and the receiving layer operates on stale state.

#### FLAT Requires Full s_waitcnt(0) (ISA §8.2)

FLAT instructions increment both VMcnt and LGKMcnt simultaneously. The ISA states: 'The only sensible S_WAITCNT value to use after Flat instructions is zero.' Triton uses partial waits, causing race conditions and stalls.

The accessibility parallel: a compliance audit that checks individual components without verifying end-to-end propagation is a partial wait - it does not drain the counter, and the system advances on the assumption that the operation completed when it did not.

#### S_BARRIER Is Not a Memory Fence (ISA §8.3)

The ISA is unambiguous: 'Barrier instructions do not wait for any counters to go to zero before issuing.' Triton treats S_BARRIER like CUDA's `__syncthreads()`, causing data races.

The accessibility parallel: a grievance resolution that is filed but not propagated to operational teams is the same architectural failure - the write appears to succeed, the system advances, and the receiving layer operates on stale state.

#### Store Visibility Requires VScnt Drain (ISA §8.2)

S_WAITCNT does not drain VScnt. Triton does not emit `S_WAITCNT_VSCNT null, 0`, meaning stores are not visible across kernels.

The accessibility parallel: an accommodation flag that is logged but does not propagate downstream has a silent data-loss event between pipeline stages. Each stage completes 'successfully.' The accommodation becomes metadata that influences nothing - a write that was never read.

[AMD RDNA3 ISA Reference, 2023; RDNA3 ISA Research Compilation, June 2026; README.md, June 2026]

| ISA Hazard | Hardware Effect | Accessibility Parallel |
|---|---|---|
| Missing WMMA V_NOP bubble | Silent wrong matmul results | Missing heading hierarchy: corrupted document outline for all downstream parsers |
| EXEC mask not restored before WMMA | Incorrect attention scores | ARIA role present but not propagated: false compliance signal |
| FLAT partial wait (not waitcnt 0) | Race conditions, stale reads | Partial accessibility audit: undetected downstream failures |
| S_BARRIER without lgkmcnt drain | Data race in tile kernels | Grievance filed but not implemented: stale operational state |
| VScnt not drained before kernel end | Stores not visible cross-kernel | Accommodation flag logged but not propagated: silent data loss |
| VOPD silently skipped in wave64 | Invisible throughput loss | Inaccessible content silently excluded from AI discovery |

**Table 3.** RDNA3 ISA hazards and their structural parallels in accessibility design.

### 7.3 The Known-Working Path: Structure as the Fix

After twelve documented training runs, a fully RDNA3-safe QLoRA pipeline was established. The configuration that produces stable, correct results requires: BF16 compute with F32 accumulators; no Triton kernels; no FlashAttention; ROCm-native matmuls only (rocBLAS/hipBLAS); standard PEFT QLoRA adapters with no fused kernels; and Transformers + Accelerate only.

This configuration works not because it is clever, but because it respects the ISA's rules for WMMA/MFMA, EXEC, FLAT/SMEM, and LDS. It provides typed, structured input to a system optimized for typed, structured input.

The final training run - a full 3-epoch QLoRA cycle on Qwen2.5-3B, 11,764 training examples, 2,208 steps - achieved a starting loss of 2.711 and a final loss of 0.18-0.21 with no GPU faults and no silent failures. Reaching that point required reading the hardware ISA and implementing fixes at a level of abstraction that no high-level framework surfaces. The accessibility parallel is exact: reaching a fully accessible system requires reading the structural requirements and implementing them at the level of the information architecture, not the surface layer. The fix is always structural. The surface layer is always downstream of the problem.

ROCm 7.2.4 subsequently introduced kernel-correctness updates that partially address several of the documented RDNA3 ISA-level hazards, including improved WMMA/MFMA dependency handling, corrected EXEC mask restoration, more complete waitcnt insertion for FLAT/SMEM, and improved LDS access patterns. These changes confirm that the hazards described were architectural rather than user-error or environment-related - exactly as the ISA analysis predicted. When the structural rules are followed, the system produces correct output. When they are not, it does not.

[RDNA3 ISA Research Compilation, June 2026; README.md, June 2026; Appendix A]

---

## §8 AI Brittleness in Clinical Contexts: When Silent Failures Meet Healthcare

Healthcare is now one of the fastest-moving domains for AI deployment. It is also, by almost every measure, the domain where silent failure is most dangerous. The brittleness documented at the hardware level - systems that continue operating and produce output that looks correct but is not - has direct structural equivalents in how healthcare AI systems are built and operated. The failure modes are not hypothetical. They are documented patterns, and they are the exact patterns that AI is currently being deployed to automate.

Every large language model trained on general internet data carries a buried assumption: that the user communicates in hearing-centric, neurotypical, able-bodied English prose, formatted the way a sighted, literate, unimpaired person would format it. When Deaf, autistic, ADHD, chronically ill, or otherwise disabled communication patterns meet a general-purpose model, the cracks appear immediately, not because the user is 'unclear,' but because the model was never trained to understand them. The model produces fluent, confident output. The output is wrong. Nothing in the logs indicates a problem.

### 8.1 Twelve Structural Failure Patterns

The following twelve patterns are drawn from documented, systemic issues across healthcare administration, clinical care coordination, and digital access. Each maps precisely to a class of AI failure documented at the hardware level. The structural logic is identical in every case: a system that was not designed to handle the full range of inputs it receives will fail silently at the edges of its training distribution.

#### 1. Diagnostic Silos in Complex Multi-System Conditions

Systemic issue: Healthcare systems organized by specialty create structural blind spots for patients whose conditions span multiple body systems. In AI terms: AI triage and clinical decision-support systems trained on single-specialty datasets inherit the same silo structure. A GI triage model will not surface rheumatology flags. The model routes correctly within its training distribution and fails silently at the edges - exactly like a GPU kernel that exceeds its tile boundary without a bounds check.

#### 2. Accommodation Documentation That Does Not Travel

Systemic issue: Accessibility accommodations recorded at scheduling rarely reach the point of care. In AI terms: an AI scheduling system that logs an accommodation flag but does not propagate it downstream has a silent data-loss event between pipeline stages. Each stage completes 'successfully.' The accommodation becomes metadata that influences nothing - identical to a GPU store that writes to a buffer no downstream instruction ever reads.

#### 3. Grievance Resolutions That Are Not Implemented

Systemic issue: Formal resolutions issued by patient-advocacy departments rarely propagate to operational teams. In AI terms: this is the memory-fence problem. A write operation that completes in one layer does not guarantee visibility to the next unless an explicit synchronization step enforces it. A grievance resolution that is filed but not propagated is the same architectural failure: the write appears to succeed, the system advances, and the receiving layer operates on stale state.

#### 4. Phone-Based Communication as Structural Exclusion

Systemic issue: Healthcare systems default to phone-based reminders and care coordination - a modality that is structurally inaccessible to Deaf patients, for whom written communication is not a preference but the only non-real-time option available. In AI terms: an AI communication system trained on hearing-centric workflows will generate phone-call prompts as its default 'accessibility' response, because that is what the training data labels as 'accommodation provided.' This is a hallucination of compliance: fluent, confident output that describes a solution and is not one.

#### 5. Digital Health Platforms That Exclude Disabled Users

Systemic issue: Patient portals are routinely redesigned without accessibility audits, creating mandatory fields - such as required phone numbers with no email-only pathway - that constitute ADA barriers on platforms subject to WCAG 2.1 AA. In AI terms: an AI system optimized for portal completion defines 'accessible' as 'works for the users who complete it,' because those are the users in its training data. Disabled users who cannot complete the portal are absent from the signal. The model cannot detect the users it is excluding.

#### 6-12. Additional Structural Failure Patterns

The remaining seven patterns follow the same structural logic. Departmental referral loops with no ownership mirror routing systems that pass responsibility without tracking it - no mechanism exists to detect when no one holds it. Diagnostic overshadowing in neurodivergent and connective-tissue patients illustrates how a model trained on biased data does not fix the bias; it makes it faster, more consistent, and harder to challenge.

Financial assistance as a reactive rather than proactive system demonstrates that a capability not invoked by default is not a feature - it is a hidden option patients must already know to ask for. Childhood medical findings not connected to adult presentations show that a model with a short context window produces confident summaries that omit the most important history. Lab trends normalized rather than investigated reveal that accuracy at a single moment is not accuracy across time.

Unauthorized modification of patient records is a prompt-injection vulnerability in a clinical context: when the input is tampered with, every downstream output is wrong. And the compounding effect of simultaneous system failures demonstrates that a system that evaluates failures individually cannot see the pattern - the patients most harmed by simultaneous failures are also the least equipped to absorb them.

> **PLAIN-LANGUAGE TAKEAWAY:**
> Healthcare AI does not need to fail catastrophically to cause harm. It only needs to be wrong at the wrong moment, in the wrong direction, with enough fluency that no one questions it. These are not hypothetical risks. They are structural patterns documented across healthcare administration, and they are the exact systems AI is currently being deployed to automate, without evidence that the underlying failures have been addressed rather than encoded and accelerated.

### 8.2 Industry Validation: GitHub's Accessibility Agent Findings

GitHub, one of the most sophisticated AI-first engineering organizations in the world, published findings on deploying AI agents to identify and fix accessibility issues in code. Their conclusion: AI agents cannot autonomously fix accessibility. Human oversight is not optional. It is structurally required.

The core finding is that accessibility requirements are deeply contextual, involve tradeoffs that cannot be resolved by pattern-matching against training data, and require the kind of judgment that comes from lived experience of disability - experience that is not captured in the text and code that models are trained on. An AI agent can identify that a button is missing an ARIA label. It cannot determine whether the label it generates actually makes sense to a screen reader user navigating a complex form. It cannot know whether the error message it wrote will be interpretable to a Deaf user reading in ASL-influenced English. It cannot know whether the accommodation workflow it designed assumes a phone call that a Deaf person cannot make.

Every one of these failures is a version of the silent wrong result documented at the hardware level: correct-looking output, wrong meaning, no flag. If the engineers building the tools acknowledge that AI cannot autonomously handle accessibility, the rest of us need to hear that clearly - especially healthcare administrators and institutions currently deploying these tools without that caveat.

[GitHub, 'Building GitHub's Next Chapter in Accessibility'; LinkedIn Article, June 2026]

---

## §9 A Framework for Oversight: Policy as Instruction Set

The failures documented at the hardware level, the model level, the application level, and the healthcare system level all share a common structure. They are invisible to the people deploying the systems. They are visible to the people most affected by the systems. And the people most affected are systematically excluded from the decision-making processes that govern deployment.

This is not a coincidence. It is the predictable outcome of designing systems without the input of the people who will reveal their failure modes. The framework proposed here is organized around the same principle that makes the RDNA3 ISA work: explicit rules, enforced at the right level of abstraction, that prevent silent failures from propagating downstream. Policy oversight is the ISA for AI deployment. The rules must be written, published, and required before the silent errors accumulate into something irreversible.

### 9.1 Five Structural Requirements

#### 1. Mandatory Pre-Deployment Accessibility Testing with Disabled Users

Not with a checklist. Not with a screen-reader simulation. With actual Deaf users, actual autistic users, actual users with cognitive and physical disabilities in their real workflows, with their real communication patterns. If a system cannot be tested this way before deployment, it should not be deployed.

This is the equivalent of the RDNA3-safe path: the configuration that produces stable, correct results requires testing against the actual hardware, not a simulation of it. Simulations do not surface the failure modes that matter.

#### 2. Silent-Failure Auditing as a First-Class Requirement

Every AI system deployed in healthcare or administrative contexts must have documented procedures for detecting when the system produces fluent but incorrect output. This includes: reproducibility testing; adversarial input testing with edge-case communication patterns; and mandatory logging that captures output variance across identical inputs over time.

Silent wrong answers must be treated as safety-critical defects - the same way a missing V_NOP between dependent WMMA instructions is treated as a correctness failure, not a performance suggestion.

#### 3. Mandatory Human Review for Accommodation, Clinical, and Benefits Decisions

AI can assist. AI can surface options. AI cannot be the final decision-maker for decisions with accessibility or healthcare consequences. Institutionalizing this boundary is not anti-AI. It is pro-accuracy.

The RDNA3 parallel: rocBLAS and hipBLAS are the known-working path precisely because they implement the necessary ordering, waitcnt, and LDS rules internally. Human review is the equivalent of using the library that already encodes the rules correctly, rather than writing custom kernels that may violate them silently.

#### 4. Communication-Accommodation Compliance in AI Interfaces

Any AI system that interacts with patients, benefits recipients, or members of the public must offer written-only communication pathways. Phone-only escalation paths in AI-mediated workflows violate ADA requirements for Deaf users. This is not a future regulation. This is current law, and it is currently being violated by deployed systems. The fix is structural: build the written pathway in from the start, or pay the retrofit multiplier when the violation is litigated.

#### 5. Public Transparency Reporting

Organizations deploying AI in healthcare or administrative contexts should be required to publish annual reports on: documented failure rates; accessibility complaints; accommodation-request processing accuracy; and remediation timelines. Sunlight is the only scalable auditing mechanism.

The RDNA3 parallel: the hazards documented in this research were only discoverable because AMD published the ISA. Transparency is the prerequisite for correctness. A system whose failure modes are not documented cannot be fixed.

> **PLAIN-LANGUAGE TAKEAWAY:**
> AI is not too complicated for meaningful oversight. Every failure mode documented at the hardware level has a corresponding rule in the ISA that, when followed, prevents it. Policy oversight is the ISA for AI deployment. We need to write it, publish it, and require people to follow it before the silent errors accumulate into something irreversible.

---

## §10 Accessibility‑Structured Documents Produce Exceptionally Clean Training Data (WhatSock Case Study)  

During the development of a personal engineering assistant model, I generated a 78‑record JSONL dataset from a 52‑page WhatSock accessibility training PDF. The document was printed to PDF with headers/backgrounds removed and processed using Microsoft's 365 Office Agent Frontier’s dataset‑extraction workflow.

**Key Result:**  
Frontier produced a fully valid dataset with 0 JSON errors, no hallucinated ARIA attributes, and no invented patterns.  
All extracted code snippets were verbatim from the source material.  

This is a strong empirical demonstration of a broader principle:  
`Accessibility patterns are inherently machine‑readable because they are deterministic, hierarchical, and semantically explicit.` 

**Dataset Summary**  
- Total records: 78
- REPAIR: 15
- CRITIQUE: 15
- SYNTHESIS: 18
- EXPLANATION: 30
- Validation errors: 0

**Patterns Successfully Extracted**

Frontier extracted 30+ WhatSock patterns, including:  
- ARIA Checkbox (simulated control)
- Native checkbox accessibility tree mapping
- Invalid ARIA Menu structures
- Misuse of `role="textbox"`
- Toggle button state patterns (`aria-pressed`)
- Drag‑and‑drop semantics (`aria-grabbed`, `aria-describedby`)
- `role="application"` misuse
- Region labeling (`aria-label`, `aria-labelledby`)
- Scrollable region rules
- Live region variants (`aria-live`, `aria-atomic`, `aria-relevant`)
- Offscreen text patterns
- Simulated button keyboard interaction rules
- `aria-activedescendant` for Listbox/Menu
- Compound widget constraints
- AT‑specific behavior (NVDA, JAWS, VoiceOver, ZoomText)
- Accessibility Tree vs DOM precedence
- Drag‑and‑drop offscreen button technique

**Why This Matters**  
This dataset demonstrates that accessibility‑aligned engineering content produces exceptionally clean machine‑learning training material.  
The structure enforced by WhatSock patterns:  
- reduces ambiguity
- eliminates hallucination risk
- improves extraction fidelity
- increases reasoning quality
- produces deterministic training examples

This supports the broader thesis of this research:
`**Accessibility is not only beneficial for end‑users — it also improves AI behavior because accessibility is structure.**`

**Implications for Model Training**

This dataset is now part of the training corpus for the personal engineering assistant model.
Expected benefits include:  
- improved ARIA reasoning
- correct pattern synthesis
- accurate markup repair
- reduced hallucination of attributes/roles
- better cross‑AT behavior modeling
- stronger structural reasoning overall

This is the first empirical confirmation that accessibility‑structured documents can serve as **high‑quality, low‑noise training material** for small, local models.

### 10.1 Original Prompt (Reproducibility Reference)

To maintain transparency and allow independent verification of the extraction workflow, the full prompt used to generate the dataset is included here:  
[WhatSock Frontier Extraction Prompt](whatsock.prompt.md)

While the source PDF cannot be shared due to licensing and distribution restrictions, the prompt itself is sufficient for readers to understand the extraction method, the expected behavior, and the structure of the resulting dataset.

### 10.2 Evidence Screenshots

The following screenshots document the extraction fidelity and dataset quality produced by Frontier.
They serve as empirical support for the claims made in this section and illustrate how accessibility‑structured documents behave under automated dataset generation.

---

1. Extraction Task Definition — Strict, Deterministic, No‑Hallucination Constraints

![A dark‑themed terminal window displays instructions for converting a 52‑page WhatSock accessibility training PDF into JSONL training data. The text emphasizes strict rules about not adding GPU‑specific code, not inventing examples, and extracting only what appears in the PDF. It outlines how to generate repair and critique samples using a fixed instruction‑input‑output schema.](../evidence/2026-06-17-prompt.png)  
**Alt Text:**  
A dark‑themed terminal window displays instructions for converting a 52‑page WhatSock accessibility training PDF into JSONL training data. The text emphasizes strict rules about not adding GPU‑specific code, not inventing examples, and extracting only what appears in the PDF. It outlines how to generate repair and critique samples using a fixed instruction‑input‑output schema.  
**Image Description:**  
The image shows a dark‑themed computer screen containing a block of instructional text for an accessibility‑focused data‑extraction task. The text explains that the user must convert a 52‑page WhatSock training PDF into JSONL training data for a coding‑assistant model, following a strict schema with “instruction,” “input,” and “output” fields. The instructions stress that no CUDA, Triton, FlashAttention, GPU logic, device maps, training code, or optimization code may be introduced, and that no examples, ARIA attributes, or code may be invented. The text states that all output must be deterministic and must come directly from the PDF. It then describes the first two categories of training samples to generate: repair samples, which require fixing accessibility issues in code snippets taken from the PDF and returning corrected code with a short explanation, and critique samples, which require listing accessibility issues in a snippet and explaining why each issue matters. The visible portion ends mid‑sentence as it continues describing the critique sample format.

2. Automated PDF Parsing and JSONL Conversion Sequence

![A dark‑themed interface labeled "Office Agent" shows an automated workflow that parses a WhatSock accessibility PDF, extracts patterns, converts them into JSONL training data, and executes several Python commands. The interface displays step‑by‑step logs, file reads, and a final message indicating the task is complete.](../evidence/2026-06-17-prompt2.png)  
**Alt Text:**  
A dark‑themed interface labeled “Office Agent” shows an automated workflow that parses a WhatSock accessibility PDF, extracts patterns, converts them into JSONL training data, and executes several Python commands. The interface displays step‑by‑step logs, file reads, and a final message indicating the task is complete.  
**Image Description:**  
The image shows a dark‑themed screen titled “Office Agent,” presenting a sequence of automated steps for processing accessibility training material. The interface displays a series of status messages describing how the system parses a WhatSock accessibility PDF, extracts patterns, and converts them into JSONL training data. Several lines show Python commands being executed, including reading intermediate files such as `pdf_content.txt` and running small inline scripts to process extracted text. The interface shows each stage completing successfully as the agent moves through parsing, chunking, extraction, and conversion. At the bottom of the screen, three rectangular buttons appear: one labeled “Convert into workflow,” another labeled “Expand coverage to more ARIA patterns,” and a third labeled “Analyze training data quality and balance.” The sequence ends with a confirmation that the task has completed.

3. Automated JSONL Generation and Validation Sequence

![A dark‑themed interface shows an automated workflow generating and validating a JSONL dataset from extracted WhatSock accessibility patterns. The screen displays file creation steps, Python command execution, validation messages confirming all records are correct, and a final summary of dataset counts.](../evidence/2026-06-17-prompt3.png)  
**Alt Text:**  
A dark‑themed interface shows an automated workflow generating and validating a JSONL dataset from extracted WhatSock accessibility patterns. The screen displays file creation steps, Python command execution, validation messages confirming all records are correct, and a final summary of dataset counts.  
**Image Description:**  
The image shows a dark‑themed screen displaying an automated sequence of steps for generating and validating JSONL training data. The interface reports that it is creating two files named `todo.md` and `gen_jsonl.py`, then running a Python script to produce the dataset. After generation, the system executes another Python command to verify that all JSONL records are valid, followed by a spot‑check of sample entries from each training‑sample type. The log confirms that all seventy‑eight records pass validation. The interface then rewrites the todo.md file and states that the dataset `whatsock_training.jsonl` is ready. At the bottom, a small summary table appears, showing the total number of records and the counts for repair, critique, synthesis, and explanation samples. The screen ends with a clear indication that the dataset creation process has completed successfully.

4. Extracted Pattern Summary and Sample-Type Breakdown

![A dark‑themed interface displays a table summarizing dataset counts for four accessibility‑training sample types, followed by a numbered list of ARIA and accessibility patterns extracted from a PDF. Buttons at the bottom offer options to convert the process into a workflow, expand coverage, or analyze data quality.](../evidence/2026-06-17-prompt4.png)  
**Alt Text:**  
A dark‑themed interface displays a table summarizing dataset counts for four accessibility‑training sample types, followed by a numbered list of ARIA and accessibility patterns extracted from a PDF. Buttons at the bottom offer options to convert the process into a workflow, expand coverage, or analyze data quality.  
**Image Description:**  
The image shows a dark‑themed screen presenting a small table that lists the number of training samples generated for each category: repair samples, critique samples, synthesis samples, explanation samples, and a row confirming zero JSON validation errors. Beneath the table is a heading indicating that the patterns were extracted exclusively from the PDF, followed by a long numbered list of accessibility and ARIA patterns. The list includes issues such as simulated checkboxes missing required roles, incorrect ARIA menu markup, misuse of roles like textbox and application, labeling requirements for regions, live region behaviors, offscreen text techniques, simulated button interactions, and ARIA menu behaviors involving aria-haspopup and aria-activedescendant. At the bottom of the interface, three rectangular buttons appear, labeled “Convert into workflow,” “Expand coverage to more ARIA patterns,” and “Analyze training data quality and balance.” The overall layout shows a completed extraction process with a clear summary of the dataset and the specific accessibility patterns it covers.

5. Advanced Accessibility Pattern Extraction (Items 15–30)

![A dark‑themed interface displays items 15 through 30 of an accessibility and ARIA pattern list, covering topics such as simulated buttons, ARIA menus, live region behaviors, assistive‑technology differences, and drag‑and‑drop techniques. A note at the bottom states that all code snippets come directly from the PDF with nothing invented.](../evidence/2026-06-17-prompt5.png)  
**Alt Text:**  
A dark‑themed interface displays items 15 through 30 of an accessibility and ARIA pattern list, covering topics such as simulated buttons, ARIA menus, live region behaviors, assistive‑technology differences, and drag‑and‑drop techniques. A note at the bottom states that all code snippets come directly from the PDF with nothing invented.  
**Image Description:**  
The image shows a dark‑themed screen containing the continuation of a numbered list of accessibility and ARIA patterns, beginning at item fifteen and ending at item thirty. The visible entries describe a range of advanced interaction and assistive‑technology behaviors, including simulated button activation using onClick and onKeyDown, ARIA menu patterns involving aria-haspopup and aria-activedescendant, live region configurations such as aria-relevant set to “removals,” and differences in how NVDA handles focus and click events for ARIA radio buttons. Additional items reference aria-activedescendant usage in listboxes, aria-pressed states on toggle buttons, compound component rules, offscreen text techniques for pseudo roles inside gridcells, live region caveats for auto‑rotating carousels, and one‑tab‑stop design patterns for complex widgets. The list also includes platform‑specific behaviors such as VoiceOver on iOS triggering focus and blur events through touch, JAWS differences between Virtual Cursor and Applications or Forms Mode, and ZoomText limitations with aria-label and aria-labelledby. The final entries mention the precedence of the Accessibility Tree compared to the DOM, the use of role="document" inside role="application," and an associated drag‑and‑drop technique that uses an offscreen button. At the bottom, a short note states that all code snippets are taken directly from the PDF and that no ARIA attributes, examples, or requirements were invented.

6. JSONL Dataset View - Verbatim Extraction and Balanced Sample Types

![A dark‑themed code editor displays a JSONL training file with columns for instruction, input, and output. The visible entries contain accessibility and ARIA tasks extracted from a WhatSock PDF, including repair, critique, implementation, and explanation prompts. Dataset statistics appear at the top, showing counts and percentages for each instruction type.](../evidence/2026-17-dataset.png)  
**Alt Text:**  
A dark‑themed code editor displays a JSONL training file with columns for instruction, input, and output. The visible entries contain accessibility and ARIA tasks extracted from a WhatSock PDF, including repair, critique, implementation, and explanation prompts. Dataset statistics appear at the top, showing counts and percentages for each instruction type.  
**Image Description:**  
The image shows a dark‑themed code editor open to a file named “whatsock_training.jsonl.” The editor displays the file in a tabular view with columns labeled instruction, input, and output. Each row represents a JSONL training record derived from the WhatSock accessibility training PDF. The visible entries include tasks such as fixing accessibility issues in code, listing issues and explaining their impact, implementing ARIA checkboxes, menus, labeled inputs, drag‑and‑drop components, scrollable regions, and live regions, as well as explaining why specific ARIA roles, states, and structural rules are required. The top of the interface shows dataset statistics, including percentages for each instruction type and a total count of seventy‑eight records. The screen conveys a structured, validated dataset intended for training an accessibility‑aware coding assistant, with all examples sourced directly from the PDF.

---

## §11 Conclusion: The Constraint That Makes Everything Else Work

Accessibility is the constraint that eliminates the largest categories of downstream cost. It is not a burden on the budget, it is the discipline that protects it. And across every domain examined in this research, the finding is consistent: early accessibility investment produces compounding returns; deferred accessibility produces compounding debt. The only difference between the two outcomes is when the decision is made.

Healthcare portals that embedded accessibility from the design phase saw reduced call volume, better patient adherence, and cleaner audit trails. Enterprise software teams that built accessibly from the start saw fewer defects, faster cycles, and lower maintenance costs. UX‑focused organizations saw higher task completion, higher satisfaction, and measurable revenue gains. Government services that planned for accessibility upfront delivered better vendor outcomes and avoided retrofits that would have consumed the entire project margin.

And now, with the WhatSock → Frontier → JSONL case study, the same pattern is visible in AI systems. A 52‑page accessibility‑structured document produced a perfectly clean dataset: 78 records, zero validation errors, no hallucinated attributes, and no invented patterns. The reason is the same mechanism observed everywhere else: accessibility enforces structure, and structure is what makes machine reasoning reliable. When content is deterministic, hierarchical, and semantically explicit, AI systems extract it cleanly, interpret it correctly, and behave predictably.

The pattern does not change across sectors. The mechanism is always the same: **structure reduces error, and accessibility enforces structure.**

### 10.1 The RDNA3 Closing Frame

The RDNA3 parallel holds here as a closing frame, not as metaphor, but as evidence. Just as RDNA3's compute units require structured, typed instruction streams to perform at rated throughput - dual-issue pipelines that degrade when instruction types are ambiguous - AI models and human readers alike require structured, accessible content to parse accurately and efficiently. The efficiency floor is set at the design phase. Ambiguity inserted at the authoring stage does not get resolved later; it propagates downstream and compounds.

The twelve training runs documented in this research demonstrate this principle at the hardware level with empirical precision. The final stable configuration works not because it is clever, but because it is correct - because it provides typed, structured input to a system optimized for typed, structured input. The same is true of accessible design: it does not work because it is generous. It works because it is structurally sound. The constraint is not a limitation. It is the design pressure that produces better outcomes for every system that consumes the output.

### 10.2 The Market Math and the Moral Arithmetic

The market math closes the financial case. 1.3 billion people with disabilities, trillions in collective spending power, a 300%+ surge in accessibility lawsuits from 2018 to 2023, and a $350,000 average lawsuit cost before remediation - this is not a niche compliance risk manageable by a legal checkbox. It is the mainstream market reality, and it is accelerating as AI search tools increasingly penalize unstructured content at the discovery layer.

But the financial case, compelling as it is, is downstream of a simpler claim: systems built for the full range of human variation are better systems. They are more robust, more maintainable, more accurate, and more honest about what they do not know. The disabled-led framing of this thesis is not incidental to its technical argument. It is the source of it.

The failure modes documented here - at the hardware level, at the model level, at the healthcare system level - were visible because someone who lives at the intersection of these systems was paying attention. That attention is not a special skill. It is the predictable outcome of including the people most affected by a system in the process of designing it.

Accessibility built in from the start costs less, reaches more people, performs better with AI tools, produces reproducible and auditable artifacts, and eliminates entire categories of rework. There is no budget, timeline, or team size at which skipping it makes financial sense. The constraint is not a burden - it is the discipline that makes everything else work. Build it in from the first keystroke, or pay the 1-10-100 multiplier later. The choice is always made at the design phase, even when it feels like it is not being made at all.

> **KEY FINDING:**
> Accessibility built in from the start is not a cost. It is the only architectural decision that simultaneously reduces legal exposure, expands the addressable market, improves AI performance, and cuts downstream maintenance costs. The evidence across six domains is not ambiguous. The numbers are not close. Build it in from the first keystroke.

---

## References

1. 365 Researcher. "Accessibility from the Start: A Research-Backed ROI Digest." Internal Research Digest, 2025.
2. WebAIM / eajournals.org. "Cost of Post-Launch Accessibility Remediation." Referenced in 365 ROI Digest, 2025.
3. inspekter.com. "The 1-10-100 Rule Applied to Accessibility." 2025.
4. Forrester Research. "The Business Impact of Accessible Technologies." 2016.
5. nascio.org. "Digital Accessibility and Government Service Delivery: Cost-Benefit Evidence." 2024.
6. xbrl.us. "LLM Accuracy on Structured vs. Unstructured Financial Data." 2025.
7. aclanthology.org. "Knowledge Graph-Augmented LLMs: Accuracy and Hallucination Reduction." Systematic Review, 2025.
8. accessibility.works. "SEMrush Analysis: WCAG Compliance and Organic Search Traffic." 2025.
9. eajournals.org. "Accessibility ROI: Development Efficiency, UX Metrics, and Revenue Impact." Multi-industry study, 2025.
10. adoc-studio.app. "Accessible Documentation and Support Cost Reduction." 2025.
11. AMD. "RDNA3 Shader Instruction Set Architecture Reference Guide." AMD Developer Documentation, August 2023. https://gpuopen.com/rdna3-isa/
12. Accenture. "Getting to Equal: The Disability Inclusion Advantage." 2018.
13. Cerebral Palsy Foundation. "Accessible Innovations in Technology." 2025.
14. Jillian. "RDNA3 ISA Research Compilation - ML Training on RX 7700 XT." Technical Reference Compilation, June 2026. https://github.com/thejeangenie18/rocm-7700xt-qlora
15. Jillian. "Triton-Generated Kernels Produce Incorrect Results or Stalls on RDNA3 Due to ISA-Level Hazards." README.md, June 2026. https://github.com/thejeangenie18/rocm-7700xt-qlora
16. Jillian. "AI Brittleness, Accessibility, and Why Healthcare Must Slow Down." LinkedIn Article, June 14, 2026.
17. GitHub. "Building GitHub's Next Chapter in Accessibility." GitHub Engineering Blog, 2025.
18. Appendix A - Evidence Screenshots & Training Logs. DeafBlind-Accessible, Section-Aware Version. https://github.com/thejeangenie18/rocm-7700xt-qlora/appendix-a.md

---

## Appendix A - Empirical Evidence: RDNA3 Training Logs

This appendix summarizes the empirical evidence supporting the RDNA3 ISA-level hazards and undocumented behaviors described in §7. All evidence is drawn from real training and inference logs collected between May 29 and June 13, 2026, on an RDNA3 GPU (Radeon RX 7700 XT) using ROCm 7.2.1.

Full screenshots and logs are available at: [github.com/thejeangenie18/rocm-7700xt-qlora/appendix-a.md](https://github.com/thejeangenie18/rocm-7700xt-qlora/appendix-a.md)

All screenshots in the repository include descriptive alt-text for screen-reader accessibility, following the same structural principles documented in this thesis.

| Evidence ID | Date | Model / Run | Key Finding Supported |
|---|---|---|---|
| A1 | May 28, 2026 | GPT-Neo-125M QLoRA Demo | Stable BF16 training; hipBLASLt override warning; no Triton kernels; correct gradient behavior |
| A2 | May 29, 2026 | Qwen2.5-3B QLoRA Training | SDMA interaction (§6.5); hipBLASLt fallback; stable BF16 compute; no Triton in successful runs |
| A3 | May 30, 2026 | TinyLlama Training Output | Long-running stability with BF16; no FLAT_SCRATCH faults; no WMMA to FLAT deadlocks |
| A4 | June 6, 2026 | Qwen2.5-3B Inference Stability | EXEC masking correctness; no partial-tile corruption; stable inference after LoRA merge |
| A5 | June 7, 2026 | Post-RDNA3-Fix Qwen2.5-3B | SDMA required for stable training; no hangs; correct WMMA ordering; RDNA3-safe path validated |
| A7 | June 10, 2026 | QLoRA Training (10k Dataset) | Stable QLoRA on RDNA3; no MFMA hazards; no EXEC mask desync; smooth LR decay |
| A8 | June 10, 2026 | Qwen2.5-3B Merge | Successful LoRA-into-base merge; no RDNA3 faults; clean model artifact write-out |
| A9 | June 10, 2026 | TinyLlama Training Run | Stable gradient checkpointing; smooth loss descent 3.6 to 1.4; no FLAT_SCRATCH faults |
| A10 | June 10, 2026 | TinyLlama LoRA Merge | Correct base model loading; stable merge-and-unload; clean artifact generation |
| A11 | June 10, 2026 | First Self-Instruct Run | GPU-accelerated model load with no RDNA3 faults; stable environment for self-instruct |
| A11.1 | June 10, 2026 | I/O Consistency Review | ADA dataset integrity verified; accessibility-focused plain-language generation confirmed |
| A12 | June 12, 2026 | First Full 3-Epoch QLoRA Cycle | Starting loss 2.711 to final loss 0.18-0.21; 11,764 examples; 2,208 steps; no faults |
| A12.1 | June 12, 2026 | TinyLlama 3-Epoch Run | Stable 3-epoch cycle; loss 2.0 to 0.3; clean LoRA adapter export |

**Table A1.** Summary of empirical evidence from RDNA3 training and inference logs, May-June 2026.
