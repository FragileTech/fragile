## Proof Sketcher Pipeline

This package automates the full life cycle for Fragile proof sketches. It drafts a structured sketch from theorem metadata, orchestrates a dual-review validation, and returns an action plan operators can trust.

### What the Pipeline Does
- `sketch_pipeline.ProofSketcherAgent` is the top-level orchestrator used by the CLI. It receives theorem context (label, statement, origin doc, status) and coordinates drafting plus validation with a single call.
- Drafting happens inside `sketcher.ProofSketchAgent`, which assembles a `ProofSketch` object. It leans on `SketchStrategist` helpers to retrieve label data, map dependencies, produce alternative strategies, and expand the roadmap/checklist sections required by `agent_schemas/sketch.json`.
- Validation is handled by `sketch_validator.SketchValidator`. It spins up two independent `SketchRefereeAgent` reviewers (Gemini + Codex personas from `sketch_referee_analysis.py`) that audit completeness, logical flow, dependency hygiene, and technical deep dives.
- The validator aggregates reviewer JSON into a `SketchValidationReport` plus numeric `Scores`. The report contains metadata, embedded reviews, a consensus analysis, and a prioritized action plan so operators know whether to promote the sketch to expansion.
- CLI entry points such as `run_sketch_agent.py`, `run_sketch_validator.py`, and `run_sketch_agent --pipeline` invoke these modules, while dashboards (for example `proof_pipeline_dashboard.py`) read the resulting report artifacts.

### Module Interaction Diagram

```mermaid
graph TD
    subgraph Drafting Stage
        A["run_sketch_agent.py / ProofSketcherAgent"] --> B["ProofSketchAgent<br/>sketcher.py"]
        B --> C[ProofSketch<br/>JSON]
        B --> D[SketchStrategist & helpers<br/>sketch_strategist.py]
    end

    subgraph Validation Stage
        C --> E[SketchValidator<br/>sketch_validator.py]
        E --> F1["SketchRefereeAgent (Gemini persona)<br/>sketch_referee_analysis.py"]
        E --> F2["SketchRefereeAgent (Codex persona)<br/>sketch_referee_analysis.py"]
        F1 --> G[SketchValidationReview #1]
        F2 --> H[SketchValidationReview #2]
        G --> I[Synthesis & Action Plan]
        H --> I
        I --> J[SketchValidationReport<br/>+ Scores]
    end

    J --> K[Dashboards & Operators<br/>proof_pipeline_dashboard.py]
```

Use this diagram as a map when wiring new tools into the pipeline: add drafting logic by extending `ProofSketchAgent`, plug additional referee agents into `SketchValidator`, or surface new metrics by enriching the `Scores` model that the dashboards consume.

### DSPy Module Workflow Diagram

The data pipeline below decomposes every `dspy.Module` participating in the workflow. Read it left-to-right to see how orchestrators call sub-agents, how each component produces structured artifacts, and how the validator fuses dual referee reviews into a publishable report with quantitative scores.

```mermaid
flowchart LR
    CLI["CLI Entrypoints\n(run_sketch_agent.py,\nrun_sketch_validator.py,\nmathster CLI)"] --> PSW["ProofSketcherAgent\n(dspy.Module)"]

    subgraph Drafting["Drafting Stage — ProofSketchAgent (dspy.Module)"]
        direction TB
        PSW --> PSA["ProofSketchAgent"]
        PSA --> Statement[ProofStatementAgent]
        PSA --> Strat1["SketchStrategist\n(primary LM)"]
        PSA --> Strat2["SketchStrategist\n(secondary LM)"]
        Strat1 --> Synth[StrategySynthesizer]
        Strat2 --> Synth
        PSA --> Deps[DependencyLedgerAgent]
        PSA --> Proof[DetailedProofAgent]
        PSA --> DeepDive[TechnicalDeepDiveAgent]
        PSA --> Checklist[ValidationChecklistAgent]
        PSA --> Alt[AlternativeApproachesAgent]
        PSA --> Future[FutureWorkAgent]
        PSA --> Roadmap[ExpansionRoadmapAgent]
        PSA --> Cross[CrossReferencesAgent]
        Statement --> Assembler[ProofSketch Assembly]
        Synth --> Assembler
        Deps --> Assembler
        Proof --> Assembler
        DeepDive --> Assembler
        Checklist --> Assembler
        Alt --> Assembler
        Future --> Assembler
        Roadmap --> Assembler
        Cross --> Assembler
        Assembler --> Sketch["ProofSketch (dict)"]
    end

    Sketch --> Validator["SketchValidator\n(dspy.Module)"]

    subgraph Validation["Validation Stage — SketchValidator (dspy.Module)"]
        direction TB
        Validator --> Metadata["dspy.Predict\nReportMetadataSignature"]
        Validator --> GeminiWrapper["SketchRefereeAgent\n(Gemini persona)"]
        Validator --> CodexWrapper["SketchRefereeAgent\n(Codex persona)"]

        subgraph Gemini["Gemini Review — SketchRefereeAgent components"]
            direction TB
            GeminiWrapper --> CompG[CompletenessCorrectnessAgent]
            CompG --> FlowG[AgentLogicalFlowValidation]
            FlowG --> DepG[AgentDependencyValidation]
            DepG --> TechG[AgentTechnicalDeepDiveValidation]
            TechG --> OverallG[AgentOverallAssessment]
            OverallG --> ReviewG[SketchValidationReview #1]
        end

        subgraph Codex["Codex Review — SketchRefereeAgent components"]
            direction TB
            CodexWrapper --> CompC[CompletenessCorrectnessAgent]
            CompC --> FlowC[AgentLogicalFlowValidation]
            FlowC --> DepC[AgentDependencyValidation]
            DepC --> TechC[AgentTechnicalDeepDiveValidation]
            TechC --> OverallC[AgentOverallAssessment]
            OverallC --> ReviewC[SketchValidationReview #2]
        end

        ReviewG --> Consensus["dspy.ChainOfThought\nConsensusAnalysisSignature"]
        ReviewC --> Consensus
        Consensus --> Actions["dspy.ChainOfThought\nActionableItemsSignature"]
        Actions --> Synthesis["dspy.ChainOfThought\nSynthesisAndActionPlanSignature"]
        Metadata --> ReportBuilder["dspy.Predict\nSketchValidationReportSignature"]
        Synthesis --> ReportBuilder
        ReportBuilder --> Report[SketchValidationReport]
        ReviewG --> Scores[Scores extraction]
        ReviewC --> Scores
        Report --> Scores
    end

    Report --> Dashboard["Dashboards & Operators\n(proof_pipeline_dashboard.py,\nreport viewers)"]
    Scores --> Dashboard
```

This workflow diagram doubles as a dependency checklist: when instrumenting new DSPy agents, add them near the relevant cluster (drafting, referee analysis, consensus, or synthesis) so downstream dashboards automatically benefit from their outputs.
