# PURE: Posterior-Preserving Uncertainty-Aware Routing for Multimodal RAG

**PURE** is a posterior-preserving framework for uncertainty-aware routing in multimodal Retrieval-Augmented Generation (RAG). It addresses the pre-retrieval evidence-allocation problem: deciding which evidence actions to execute before observing the actual retrieved content.

![PURE Framework](assets/Figure_01.png)

## Overview

In multimodal RAG, a query may be answered without retrieval, or supported by paragraph-level text, document-level context, or image-based evidence. Existing routers typically commit to a single path, discarding distributional information over alternative evidence sources. PURE instead preserves routing uncertainty as an operational signal across the entire pipeline, integrating evidential routing, budget-aware branch selection, and posterior-coupled answer verification.

## Architecture

![Pipeline](assets/Figure_02.png)

PURE consists of three stages:

1. **Evidential Router** — Estimates both action preference and evidence strength using Dirichlet distributions, producing a continuous posterior surrogate rather than a collapsed classification decision.

2. **Branch-Budget-Aware Selector** — Converts evidential signals into an executable branch-ranking policy under a budget encoding, selectively preserving uncertain but potentially useful retrieval paths.

3. **Posterior-Coupled Verifier** — Resolves disagreement among candidate answers by combining branch-level posterior weights with answer consistency and evidence availability signals.

![Router](assets/Figure_03.png)
![Verifier](assets/Figure_05.png)

## Results

PURE achieves consistent improvements over strong baselines under a shared retriever set, generator, and evaluation protocol:

| Benchmark | Best Baseline | PURE |
|-----------|--------------|------|
| Natural Questions (F1) | 49.46 | **57.64** |
| HotpotQA (F1) | 46.03 | **46.12** |
| WebQA (ROUGE-L) | 41.45 | **43.42** |
| TriviaQA OOD (F1) | 76.25 | **83.23** |

Gains are most pronounced on retrieval-sensitive benchmarks where branch choice materially changes the available evidence. Cross-generator evaluation with GPT-4o and GLM-4V yields consistent aggregate trends.

## Installation

```bash
conda create -n pure python=3.12 -y
conda activate pure
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
bash script/0_dataset.sh
bash script/1_preprocess.sh
```

### Router Training

Train the evidential router with DistilBERT or T5-large backbone:

```bash
bash script/2_train.sh {distilbert|t5-large}
```

### Evaluation

```bash
bash script/4_eval.sh \
    --model_path {generator-path} \
    --router_model {router-model} \
    --target {dataset}
```

Full pipeline scripts for all baselines and PURE variants are provided under `script/`.

## Project Structure

```
├── route/                    # Router implementations and training
│   ├── core/                 # PURE evidential routers
│   ├── train/                # Training scripts
│   ├── baselines/            # Baseline adapters (Adaptive-RAG, Self-RAG, CRAG)
│   ├── gpt/                  # GPT-based routing
│   └── qwen/                 # Qwen-based routing
├── eval/                     # Evaluation framework
│   ├── retrieve/             # Retrieval backends
│   └── utils/                # Generator wrappers
├── analysis/                 # Diagnostics and analysis
├── preprocess/               # Data preprocessing
├── tools/                    # Utility tools
├── script/                   # Shell pipeline scripts
├── paper/                    # LaTeX source
├── assets/                   # Figures
└── requirements.txt
```

## Benchmarks

In-domain evaluation covers MMLU, SQuAD, Natural Questions, HotpotQA, and WebQA. Zero-shot out-of-domain transfer is evaluated on TriviaQA, TruthfulQA, LaRA, and Visual-RAG.

## Baselines

PURE is compared against fixed retrieval policies, UniversalRAG-style routing, Adaptive-RAG, Self-RAG, CRAG-style evidence gating, and hard top-k verifier baselines under the same executable-action interface and evaluation protocol.

## Citation

```bibtex
@article{PURE,
  title={PURE: Posterior-Preserving Uncertainty-Aware Routing with Evidential 
         Selection and Verification for Multimodal Retrieval-Augmented Generation},
  author={},
  journal={},
  year={}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
