# d3-multi

**A fourth question type for decision models. `multi` answers "which of these options apply?" with every
option that applies, from zero to many, in one forward pass.**

Decision models such as [d3-mini](https://huggingface.co/vllm-sr/d3-mini) answer three question types:

| Type | Answers | Returns |
|---|---|---|
| `choice` | Which single option fits best? | One option |
| `score` | Where does this sit on an ordered scale? | One level |
| `noul` | Is this statement true? | One probability |
| **`multi`** | **Which options apply?** | **Every option that applies, a calibrated score for each, and the cutoff** |

Without `multi`, a multi-label task is asked as one yes/no question per label. Ten labels cost ten forward
passes, each label is judged without seeing the others, and the yes-threshold must be tuned for each task.

`multi` adds one option to the question, the **cutoff**. The model scores the cutoff in the same pass and the
same softmax as the labels, and keeps every label that outscores it. Because the cutoff is scored from the
input, the model sets a new threshold for every input. The default cutoff works without tuning. An offset is there when a task needs it.

The method and its evidence are in the companion paper, **[The Cutoff Is an Option](https://akta.pro/blog/the-cutoff-is-an-option-single-pass-multi-label-classification-with-adaptive-thresholding)**.

## Results

We fine-tuned d3-mini for `multi` on public multi-label datasets and compared it with the base model. Every
accuracy number below is micro-F1 on held-out test data. "1 pass" means one forward pass per input.

### Out of the box: no threshold tuning on either side

| Test set | Labels | Base: one `noul` per label (p > 0.5) | Base: `choice` top-1 | **d3-mini-multi (1 pass)** |
|---|---|---|---|---|
| arXiv categories | 155 | 0.29 | 0.65 | **0.67** |
| Civil Comments (toxicity attributes) | 7 | 0.59 | 0.41 | **0.78** |
| DBpedia-14 | 14 | 0.90 | 0.97 | **0.98** |
| ECtHR-A (articles violated) | 10 | 0.66 | 0.69 | **0.75** |
| ECtHR-B (articles alleged) | 10 | 0.69 | 0.72 | **0.80** |
| GoEmotions | 28 | 0.40 | **0.56** | **0.56** |
| **UNFAIR-ToS (never seen in training)** | 8 | 0.60 | 0.19 | **0.66** |
| **TweetTopic (never seen in training)** | 19 | 0.63 | **0.68** | **0.68** |
| **Hate-speech targets (never seen in training)** | 8 | 0.69 | **0.76** | 0.72 |

`multi` beats per-label `noul` on all nine test sets, including the three domains that are not in the training
data. Against `choice` top-1, it wins on six, ties on two, and trails on hate-speech targets, where most comments
target a single group.

### Against a tuned baseline

A strong baseline tunes its threshold for each test set. We picked the base model's best threshold by two-fold
cross-validation on the test set itself, which a real user can do only with labelled data. `multi` stays
untuned.

| Test set | Best tuned base method | Tuned base F1 | **d3-mini-multi, untuned** |
|---|---|---|---|
| arXiv categories | `choice` + threshold | 0.67 | 0.67 |
| Civil Comments | `noul` per label | 0.66 | **0.78** |
| DBpedia-14 | `multi` on base | 0.97 | **0.98** |
| ECtHR-A | `choice` + threshold | 0.72 | **0.75** |
| ECtHR-B | `choice` + threshold | 0.77 | **0.80** |
| GoEmotions | `choice` + threshold | **0.57** | 0.56 |
| UNFAIR-ToS (unseen) | `multi` on base | **0.77** | 0.66 (0.76 with offset +2.4) |
| TweetTopic (unseen) | `choice` + threshold | **0.72** | 0.68 (0.70 with offset −0.75) |
| Hate-speech targets (unseen) | `choice` top-1 | **0.76** | 0.72 (0.75 with offset −0.5) |

Untuned `multi` beats the tuned base on four of the six training domains, and is within 0.01 on the other two.
On the three unseen domains the tuned base is ahead. With a tuned offset, `multi` comes within 0.02 of it.

### Cost

One `multi` question replaces one `noul` question per label. Prompt tokens per input:

| Test set | Labels | `multi` | One `noul` per label | Ratio |
|---|---|---|---|---|
| arXiv categories | 155 | 1,920 | 41,533 | 21.6x |
| GoEmotions | 28 | 260 | 3,248 | 12.5x |
| DBpedia-14 | 14 | 276 | 2,547 | 9.2x |
| ECtHR-A/B | 10 | 1,341 | 11,974 | 8.9x |
| Civil Comments | 7 | 273 | 1,200 | 4.4x |
| UNFAIR-ToS | 8 | 293 | 1,219 | 4.2x |

vLLM serves at most 64 questions per request for d3-mini, so a per-label approach over more than 64 labels
needs several requests. One `multi` question takes up to 254 options.

### Calibration

A `multi` score is σ(zᵢ − z_REF), the probability that the option applies. Expected calibration error, on
options scored above 0.05:

| Test set | Base `noul` per label | **d3-mini-multi** |
|---|---|---|
| arXiv categories | 0.243 | **0.090** |
| Civil Comments | **0.048** | 0.109 |
| DBpedia-14 | 0.141 | **0.024** |
| ECtHR-A | 0.183 | **0.065** |
| ECtHR-B | 0.184 | **0.088** |
| GoEmotions | 0.253 | **0.044** |
| UNFAIR-ToS (unseen) | 0.261 | **0.240** |

### Effect on the other question types

Fine-tuning for `multi` must not break `choice`, `score` and `noul`. We checked this on public benchmarks:

- **Decision Index public benchmarks.** We ran the base and fine-tuned models on the same 10,763 requests
  (29,053 questions) from 42 public benchmarks. The public index is 55.10 for d3-mini-multi and 55.05 for the
  base model: a change of +0.01 points, with a standard error of 0.22.

## Quickstart

Serve the model with vLLM, then put the `multi` proxy in front of it:

```bash
pip install -r requirements.txt
vllm serve akta-pro/d3-mini-multi --served-model-name d3-mini-multi --port 8100 --no-enable-prefix-caching
python -m d3multi.serve --upstream http://localhost:8100 --port 8200
```

Ask a `multi` question:

```bash
curl -s localhost:8200/v1/systemone -H 'content-type: application/json' -d '{
  "model": "d3-mini-multi",
  "state": "The order arrived damaged yesterday. The customer was charged twice and wants a replacement today.",
  "questions": {
    "teams": {
      "type": "multi",
      "instructions": "Which teams need to act on this request?",
      "criteria": {
        "returns": "Refunds, replacements and damaged deliveries",
        "billing": "Payments, invoices and charges",
        "technical": "Product setup and faults",
        "shipping": "Delivery tracking and couriers"
      }
    }
  }
}'
```

The response (scores rounded to four decimals):

```json
{
  "id": "decision-b949693620a47b32",
  "object": "structured_decision",
  "created": 1792849929,
  "model": "d3-mini-multi",
  "answers": {
    "teams": {
      "type": "multi",
      "selected": ["returns", "billing"],
      "scores": {
        "returns": 0.9867,
        "billing": 0.9108,
        "shipping": 0.2438,
        "technical": 0.0494
      },
      "cutoff": 0.0116
    }
  },
  "usage": {
    "input_tokens": 189,
    "output_tokens": 0
  },
  "diagnostics": {
    "teams": {
      "label_mass": 1.0,
      "argmax_is_label": true
    }
  }
}
```

`choice`, `score` and `noul` questions pass through the proxy unchanged, and they can share a request with
`multi` questions. The full request and response format is in [docs/multi-spec.md](docs/multi-spec.md).

### Without the proxy

`multi` needs no server change. A client can send the `choice` question itself and decode the answer:

```python
from d3multi import multi

question = multi.to_choice({"instructions": "Which teams need to act?", "criteria": criteria})
# POST it to /v1/systemone as an ordinary choice question, then:
answer = multi.decode(response["answers"]["teams"]["probabilities"], offset=0.0)
```

### Choosing the offset

`offset` shifts the cutoff for all inputs: keep options whose margin over the cutoff is above it.

| `offset` | Effect |
|---|---|
| 0 (default) | What the model was trained for. Start here. |
| Negative, for example −1 | More options per input: higher recall, lower precision |
| Positive, for example +1 | Fewer options per input: higher precision, lower recall |

## How it works

The model reads the input, the options and the cutoff option `REF` in one pass, and gives each a raw score
zᵢ. The server returns the usual softmax. The ratio of two probabilities cancels the softmax's shared
denominator:

```text
log(p_i / p_REF) = z_i − z_REF
selected = { i : z_i − z_REF > offset }
```

So the cutoff is not diluted as options are added. If nothing beats the cutoff, the
answer is empty: "none of these" needs nothing extra.

## Limits

- **Unseen domains.** On UNFAIR-ToS, untuned `multi` reaches recall 0.94 but precision 0.51: it tags too many
  sentences on a set where 89% of inputs have no label. On hate-speech targets, where most comments target one
  group, `choice` top-1 is ahead (0.76 against 0.72). More training domains and more "none" rows are the next step.
- **254 options per question.** The readout scores 255 codes, and the cutoff takes one. A larger label set
  needs a retrieval step first, or several questions.
- **Near-zero margins.** Options within a hair of the cutoff can flip with batch shape or numeric precision.

<sub>All test sets are English text. We did not test other languages, or `multi` questions about images or video.</sub>

## Acknowledgements

`multi` is built on [d3-mini](https://huggingface.co/vllm-sr/d3-mini) by the vLLM Semantic Router Team. We thank the authors of the public datasets used for training and testing:

- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Google Research)
- [arXiv metadata](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) (arXiv)
- [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (Jigsaw and Civil Comments)
- [LexGLUE: ECtHR, SCOTUS and UNFAIR-ToS](https://huggingface.co/datasets/coastalcph/lex_glue) (CoAStaL, University of Copenhagen)
- [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) (DBpedia)
- [Medical Abstracts](https://huggingface.co/datasets/TimSchopf/medical_abstracts)
- [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) (Amazon)
- [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (UC Berkeley D-Lab)
- [TweetTopic](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi) (Cardiff NLP)
- [BoolQ](https://huggingface.co/datasets/google/boolq) (Google Research)
- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) (Tel Aviv University)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) and [QASC](https://huggingface.co/datasets/allenai/qasc) (Allen Institute for AI)
- [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) (Stanford NLP)
- The [Decision Index](https://github.com/apolinario/decision-index) public benchmark kit

## License

Apache-2.0. d3-mini is Apache-2.0 (Copyright 2026 vLLM Semantic Router Team), built on Qwen3.5-4B
(Apache-2.0).

## Citation

```bibtex
@misc{d3_multi_2026,
  title        = {The Cutoff Is an Option: Single-Pass Multi-Label Classification
                  with Adaptive Thresholding},
  author       = {Balot, Harsh and {akta.pro}},
  year         = {2026},
  howpublished = {\url{https://akta.pro/blog/the-cutoff-is-an-option-single-pass-multi-label-classification-with-adaptive-thresholding}},
  note         = {Code: \url{https://github.com/akta-pro/d3-multi}}
}
```
