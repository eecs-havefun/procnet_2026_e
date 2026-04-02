# EPAL Event Inference in the Updated W2NER + ProcNet Framework

_Date: 2026-04-02_

## 1. Purpose of this note

This note consolidates the recent discussion into one report and adds a dialogue summary.

The focus is:

1. how the **event inference** method in the paper **Joint Learning Event-specific Probe and Argument Library with Differential Optimization for Document-Level Multi-Event Extraction** can be used in the **updated** repositories:
   - `eecs-havefun/W2NER_2026_e`
   - `eecs-havefun/procnet_2026_e`
2. how its mechanism can be applied **inside ProcNet**, after the W2NER → ProcNet coupling is already working
3. how it can help solve two problems at once:
   - **cross-event reuse of the same entity**
   - **multi-role use of the same entity inside one event**

---

## 2. Current framework snapshot

### 2.1 W2NER side

The updated `W2NER_2026_e` repository is no longer only exporting the original `entity` list.  
Its `predict()` path now also exports a ProcNet-friendly `procnet_entities` structure. The repo code and snippets show that the prediction record now includes fields such as:

- `token_indices`
- `b`, `e`
- `type_id`
- `type`
- `score`
- `head`
- `text`
- `key`
- `cluster_key`

This means the W2NER side has already moved from “plain NER output” toward “typed entity sidecar for downstream ProcNet consumption”.

### 2.2 ProcNet side

The updated `procnet_2026_e` repository already supports external typed entities in its running path.

From the current repo snapshot and code snippets:

- `run.py` includes switches such as:
  - `--use_procnet_pred_entities`
  - `--typed_entities_dir`
  - `--use_procnet_entity_nodes`
- `DocEEProcessor` already contains logic for:
  - reading sidecar entities
  - resolving `sent_id`, `b`, `e`
  - resolving type names
  - normalizing sidecar items into internal typed-entity objects
  - merging them into the document object

This means the **coupling problem is largely solved at the interface level**.  
The system is no longer blocked on “how to pass entities from W2NER to ProcNet”.

### 2.3 What remains unsolved after coupling

After the interface is opened, the main remaining problem is **event inference quality**, especially:

1. **cross-event confusion**  
   the same entity appears in multiple events and ProcNet may merge or confuse those events

2. **same-event multi-role conflict**  
   the same entity should be allowed to fill multiple roles in one event when the schema semantics require it, but the current ProcNet-style decoding is not naturally designed for that

So the bottleneck has shifted from **format coupling** to **event decoding structure**.

---

## 3. What EPAL contributes that matters here

The paper introduces three key ideas that matter for your current stage:

1. **event-specific probe**
2. **event-specific argument library**
3. **role-indexed slot filling in event inference**

The most relevant part for your question is the third one.

### 3.1 Event-specific probe

EPAL initializes a probe for each candidate entity, so each probe acts like a possible event instance detector.

The point is not “one event type = one probe”; instead, it is closer to:

- each entity gets a chance to act as an event anchor
- different probes can discover different event instances
- some probes will predict `None`

This is especially useful when one document contains multiple homogeneous events and shared arguments.

### 3.2 Event-specific argument library

For each probe, EPAL rebuilds the representation of candidate arguments under the current event view.

That means:

- the same entity does **not** keep one fixed global representation
- under probe A, entity X gets one event-conditioned representation
- under probe B, the same entity X gets another representation

This is the core reason EPAL is strong on **cross-event reuse**.

### 3.3 Role-indexed slot filling

This is the most important mechanism for your ProcNet question.

In EPAL, after obtaining the event-specific argument library, the model does **not** mainly ask:

> for each entity, which single role does it belong to?

Instead, it asks:

> for each role, which candidate argument should fill this slot?

Operationally, EPAL obtains a matrix whose:

- rows correspond to roles
- columns correspond to candidate arguments (plus a virtual CLS candidate for missing roles)

Then it applies **row-wise softmax**.

So the decoding logic becomes:

- `role_1 -> pick one argument`
- `role_2 -> pick one argument`
- `role_3 -> pick one argument`
- ...

This is a **role-indexed filling strategy**.

---

## 4. Why this matters more than the current ProcNet decoding pattern

Your current ProcNet pipeline is still fundamentally centered around:

- **proxy nodes**
- global event-set learning
- entity-conditioned event decoding

That is good for global optimization, but it also tends to behave like:

> for one proxy-event, classify each entity into one role / null

This style has a structural limitation.

### 4.1 Limitation for same-event multi-role

If decoding is essentially entity-indexed, a single entity is usually pushed toward **one dominant role label** within one event.

That is fine when the schema assumes one entity can fill only one role.  
But if your actual coupled framework must handle cases like:

- one date expression serving as two roles
- one place expression serving as two slots
- one normalized span legitimately reused inside one event

then entity-indexed role classification becomes restrictive.

By contrast, EPAL's role-indexed filling naturally allows this:

- role A can select entity X
- role B can also independently select entity X

because the softmax is performed **per role row**, not across roles for one entity.

So for **same-event multi-role**, EPAL's slot filling is structurally better matched to the problem.

### 4.2 Strength for cross-event reuse

For multiple events, the harder part is not only “can one entity be reused?”, but:

> can the model reuse the same entity **without collapsing multiple events into one**?

EPAL handles this by combining:

- event-specific probes
- event-specific argument libraries
- probe-label alignment
- role contrastive learning

So the same entity can appear:

- in event A under probe A
- in event B under probe B

and each usage is represented differently in the event-conditioned space.

That is exactly the kind of mechanism you need after W2NER and ProcNet are already coupled.

---

## 5. The key judgment for your case

## My judgment in one sentence

**The most suitable EPAL idea to import into your current ProcNet is not the whole model first, but the "role-indexed slot filling over an event-specific argument library".**

In other words:

- **do not replace W2NER**
- **do not immediately replace ProcNet's whole proxy-node framework**
- **do replace or augment ProcNet's event-argument decoding stage**

This is the highest-value integration point.

---

## 6. Recommended integration path inside ProcNet

## 6.1 Keep W2NER unchanged

Your W2NER side already produces `procnet_entities`.  
That part is now useful and stable enough.

So the EPAL idea should be imported **after** typed entities have already entered ProcNet.

This keeps the pipeline clear:

- W2NER = mention/type candidate provider
- ProcNet = event reasoning and slot filling

## 6.2 Keep proxy nodes in the first upgraded version

A full switch from ProcNet proxy nodes to EPAL entity-initialized probes would be a major rewrite.

You do not need to do that first.

A much more practical route is:

- keep ProcNet's proxy nodes as event hypotheses
- for each proxy node, build an **event-specific argument library**
- decode arguments using **role-indexed filling** rather than only entity-indexed classification

This gives you an **EPAL-style decoder on top of ProcNet's existing event representation**.

That is the most reasonable middle path.

## 6.3 Build a proxy-conditioned argument library

For each proxy node `z_i`, construct a library:

- candidate typed entities from sidecar / gold / merged entity nodes
- one virtual `CLS` argument for missing roles

Then, instead of using only one score per `(proxy, entity)` role label in an entity-centric way, compute a matrix:

- row = role
- column = candidate argument

and do per-row softmax.

This turns the ProcNet event decoder into:

- event type prediction
- role-by-role argument selection

That is exactly the part of EPAL that addresses your problem most directly.

## 6.4 Add explicit support for repeated entity use inside one event

This is critical.

When you switch to role-indexed filling, **do not add a hard constraint that one entity may only be used once inside the same event**.

Instead, make the default behavior:

- each role independently chooses one candidate or CLS
- the same candidate may be selected by multiple roles

Then add **optional consistency rules** afterward only if the schema requires them.

This is how the decoder becomes able to solve the **single-entity multi-role** problem.

## 6.5 Keep event-set optimization first, add EPAL losses later

ProcNet's current strong point is global event-set optimization through the proxy-node framework.

So in the first upgrade stage, keep:

- the proxy-node event hypothesis mechanism
- the global event-set training logic as much as possible

Then add EPAL-inspired auxiliary objectives gradually:

1. first add **role-indexed slot filling**
2. then add **role contrastive loss**
3. then consider **probe/proxy alignment stabilization**

This staged path is safer than trying to import all EPAL machinery at once.

---

## 7. What exactly this solves

## 7.1 Same entity used in multiple events

EPAL-style event-conditioned argument libraries help because the same entity is re-encoded under different event hypotheses.

So the system can represent:

- entity X as part of event A
- entity X as part of event B

without forcing both usages into one shared static argument embedding.

This directly attacks **cross-event confusion**.

## 7.2 Same entity used in multiple roles in one event

Role-indexed slot filling helps because the unit of prediction changes from:

- entity -> one role

to:

- role -> one argument

Under this decoding view, one entity can legitimately occupy:

- `role_1`
- `role_2`

inside the same event.

This directly attacks **same-event multi-role conflict**.

---

## 8. What should not be done first

There are three things I would **not** recommend as the first move.

### 8.1 Do not replace the whole ProcNet backbone immediately

A full rewrite from proxy nodes to entity-initialized probes plus new alignment is too large for the first iteration.

### 8.2 Do not put EPAL logic back into W2NER

The problem now is not mention extraction anymore.  
It is event reasoning.

So the right insertion point is still ProcNet.

### 8.3 Do not start with full alignment loss first

Probe-label alignment is powerful, but it is also one of the most invasive parts of EPAL.

You should first verify that **role-indexed filling itself already improves**:

- event F1
- per-role F1
- multi-event recall
- multi-role cases

before moving to alignment-heavy redesign.

---

## 9. A practical 3-stage roadmap

## Stage 1: EPAL-lite on ProcNet

Goal: solve the decoding bottleneck with minimal rewrite.

Do:

- keep current proxy nodes
- keep current typed entity input path
- replace/augment event decoding with role-indexed slot filling
- add one CLS candidate per event for missing roles

Expected benefit:

- immediate support for same-event multi-role reuse
- cleaner slot-based event tables

## Stage 2: Add event-conditioned argument contrast

Goal: improve cross-event discrimination.

Do:

- keep proxy nodes
- compute event-conditioned argument representations
- add role contrastive loss between same-type events

Expected benefit:

- reduce confusion among homogeneous events
- better multi-event recall

## Stage 3: Alignment-aware proxy refinement

Goal: get closer to full EPAL behavior.

Do:

- add a proxy-to-event alignment mechanism
- optionally initialize some proxy hypotheses from high-confidence entities
- stabilize proxy/event correspondence during training

Expected benefit:

- less event collapse
- more stable training on multi-event documents

---

## 10. Cost and risk assessment

### 10.1 Structural cost

Low to medium if you only replace the decoder.  
High if you also replace the training objective and proxy initialization strategy.

### 10.2 Engineering cost

The lowest-cost EPAL import is:

- new argument-library builder
- new role-indexed decoder head
- new loss for row-wise role filling

This is much cheaper than a full architecture rewrite.

### 10.3 Research risk

The biggest risk is not implementation failure, but **metric mismatch**:

- the model may get better at role-level correctness
- yet not improve final event-set F1 unless event matching remains stable

So every change must still be evaluated at:

- event F1
- per-role F1
- multi-event F1
- case-based analysis on same-entity multi-role examples

---

## 11. Final conclusion

### Core conclusion

For your updated repositories, the best way to use the EPAL paper is:

**import EPAL's event-specific argument library + role-indexed slot filling into the ProcNet decoding stage, while keeping W2NER unchanged and keeping ProcNet proxy nodes in the first enhanced version.**

### Why this is the best fit

Because your current framework status is already:

- W2NER can export ProcNet-friendly typed entities
- ProcNet can already ingest typed sidecar entities

So the remaining problem is no longer coupling.  
It is the **event inference structure**.

And among EPAL's ideas, the one that most directly targets your two remaining problems is:

- **role-indexed slot filling** for same-event multi-role
- **event-conditioned argument library** for cross-event entity reuse

### Strong recommendation

Do not try to port full EPAL at once.  
Use this priority order:

1. role-indexed slot filling
2. event-conditioned argument library
3. role contrastive loss
4. alignment stabilization
5. optional probe-like proxy redesign

---

# Dialogue Summary

## A. Early discussion: coupling status

We first reviewed your three internal notes and reached a common conclusion:

- W2NER sentence-level `procnet_entities` can already be converted into ProcNet doc-level sidecar JSONL
- type normalization, doc aggregation, sentence ordering, and composite keys are already working
- `DocEEProcessor` can load the sidecar successfully
- the pipeline is no longer blocked by span alignment or format mismatch

At that point, the main unresolved issue was identified as:

- semantic mismatch of `date/time`
- insufficient coverage of some key roles
- event reasoning after coupling

## B. Next-step planning

We then turned the strategy into concrete next steps.

The priority was set as:

1. stop spending effort on format-layer repairs
2. move to semantic closure and ablation
3. evaluate ProcNet under four conditions:
   - baseline
   - stable typed entities only
   - stable typed entities + `date/time` resolver
   - gold typed entities

The purpose of these ablations was to determine whether the real bottleneck lies in:

- sidecar usefulness
- time-role disambiguation
- W2NER recall
- or ProcNet itself

## C. Repository cleanup discussion

We also reviewed both repositories from a “clean repo” perspective.

The main conclusion was that several categories are unnecessary for a clean public repository:

- conversation summaries / history
- logs
- generated outputs
- full data directories
- full sidecar result directories
- personal tool config files

The recommended principle was:

- keep code
- keep scripts
- keep tiny examples
- move large derived data out of the repository
- generate reproducible artifacts through scripts instead of tracking them directly

## D. EPAL summary discussion

We then summarized the EPAL paper and clarified a key distinction:

- EPAL is strongest on **cross-event reuse of the same entity**
- EPAL is not primarily designed to solve **same-event multi-role sharing**
- however, its role-indexed slot filling **does structurally allow** the same entity to be selected by multiple roles in one event

This was an important correction in the discussion, because it showed that the paper is relevant to **both** problems in your framework, though more directly to the multi-event one.

## E. Current focused conclusion

After you clarified that both problems matter and the solution should be introduced **in ProcNet**, we refined the final conclusion to:

- keep W2NER as typed-entity producer
- solve the remaining issues in ProcNet
- import EPAL's **role-indexed slot filling** and **event-specific argument library**
- do this first as a decoder-level enhancement rather than a full model replacement

This is the current recommended research and engineering direction.

---

# Reference Notes

## Repositories checked

- `https://github.com/eecs-havefun/W2NER_2026_e`
- `https://github.com/eecs-havefun/procnet_2026_e`

## Key current observations from repo snapshot

- `W2NER_2026_e` currently exports `procnet_entities` in prediction output.
- `procnet_2026_e` currently supports `typed_entities_dir` and `use_procnet_pred_entities`.
- `DocEEProcessor` already contains sidecar normalization and merge logic.

## Main paper discussed

- Hu et al., 2025  
  **Joint Learning Event-specific Probe and Argument Library with Differential Optimization for Document-Level Multi-Event Extraction**

## Internal notes discussed earlier

- `2026040215summary.txt`
- `w2ner_procnet_script_and_summary.md`
- `w2ner_procnet_current_conclusion.md`
- `epal_event_inference_detailed_explanation.md`
