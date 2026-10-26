# The `multi` question type

`multi` answers "which of these options apply?" with every option that applies, from zero to many, in
one forward pass. It sits beside the three System One types:

| Type | Answers | Returns |
|---|---|---|
| `choice` | Which single option fits best? | One option and a probability for each |
| `score` | Where does this sit on an ordered scale? | A probability-weighted level |
| `noul` | Is this statement true? | One probability |
| **`multi`** | **Which options apply?** | **The options that apply, a calibrated score for each, and the cutoff** |

## Request

```json
{
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
      },
      "offset": 0
    }
  }
}
```

| Field | Rule |
|---|---|
| `criteria` | An object of 1 to 254 options: key to description, or to `null` for none. |
| `instructions` | Optional. Default: "Which options apply?" |
| `offset` | Optional, default 0. A negative value selects more options and a positive value fewer. |
| Reserved | The key `REF` is the cutoff option. A request may not use it. |

## Response

```json
{
  "teams": {
    "type": "multi",
    "selected": ["returns", "billing"],
    "scores": {"returns": 0.99, "billing": 0.91, "shipping": 0.24, "technical": 0.05},
    "cutoff": 0.0116
  }
}
```

| Field | Meaning |
|---|---|
| `selected` | Options whose margin over the cutoff is greater than `offset`, highest first. Empty when nothing applies. |
| `scores` | For each option, σ(zᵢ − z_REF): the calibrated probability that it applies. |
| `cutoff` | The probability the model gave the cutoff option. A high value means the model saw little that applies. |

## How it runs

The server sends a `multi` question to the model as a `choice` question with one more option:

- **Key:** `REF`.
- **Text:** "Cutoff: no remaining option applies".
- **Instructions:** the request's instructions, plus one fixed paragraph that says every option that applies is
  correct, there can be several or none, and each applies option must score above `REF`.

The model scores `REF` in the same pass and the same softmax as the options. The response is decoded
from the returned probabilities:

```text
margin_i = log(p_i / p_REF) = z_i − z_REF
selected = { i : margin_i > offset }
score_i  = σ(margin_i)
```

The cutoff is part of the input, and the model sets its value again for every request. The exact strings
are in [`d3multi/multi.py`](../d3multi/multi.py). The model was trained on them, so a client must not
change them.

## Limits

- **254 options per question.** The readout scores 255 codes, and the cutoff takes one.
- **More options need more questions.** Splitting a label set across questions gives only an approximate
  answer, because each question's scores see only its own options.
- **Margins near zero can flip** with batch shape or precision. FP8 keeps answers. FP4 moves many of them.
