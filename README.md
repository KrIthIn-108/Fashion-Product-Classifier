# 👕 Fashion Product Classifier 

 **[Try the Live Demo ⏩ ](https://stylesight-vit-01.streamlit.app)** — Upload an image to classify garments in real-time.

> **A fashion classifier that started with a failed viva question, disappeared for almost two years, and came back as a CLIP + Vision Transformer pipeline.**

---

## 🧵 The Story Behind This Project

This project did not start as a brand-new idea.

It started with an old one.

Back in college, I built a **CNN-based fashion product classifier** as part of a college project. The idea was simple:

**Give the model a fashion image → classify it into useful fashion labels.**

The model looked good on the training dataset, and for a while, I thought the hard part was done.

Then came the viva.

An external asked me:

> **“What happens if I give your model an image of an apple or a pen?” 🍎🖊️**

That question exposed the biggest weakness in my implementation.

The CNN was happy to classify images from the fashion domain, but it had no real mechanism to understand:

> **“This isn't fashion at all.”**

It could also perform poorly on unfamiliar web images because the model had become too dependent on the training data.

I couldn't properly solve that problem back then.

But the question stayed with me.

Almost **two years later**, I decided to rebuild the project instead of simply accepting the old limitations.

This time, I replaced the old CNN-heavy approach with a more modern computer-vision pipeline using:

**CLIP → Guardrail → Vision Transformer → Fine-tuned Fashion Attributes → JSON**

And that became this project.

---

# 🎯 Mission

The main purpose of this project is to **streamline fashion product labeling for e-commerce and online stores**.

Imagine an online store receiving thousands of product images every day.

Someone still needs to attach useful metadata to those images:

- What kind of product is this?
- Which category does it belong to?
- What is the article type?
- What colour is it?
- What is its typical usage?

Doing this manually can become slow and expensive at scale.

The vision behind this project is therefore:

> **Give the system a fashion product image, let it understand what it is, extract useful product attributes automatically, and return structured metadata that can be consumed by an e-commerce workflow.**

The longer-term goal is not just to build a classifier, but to move toward an **AI-assisted fashion cataloging system**.

---

# 🧠 What Does the System Do?

The current pipeline follows this flow:

```text
                    Input Image
                         │
                         ▼
                ┌─────────────────┐
                │  CLIP Guardrail │
                │ Fashion or not? │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │                 │
             No / OOD           Clothing
                │                 │
                ▼                 ▼
           Reject image      Fine-tuned ViT
                                  │
                                  ▼
                         ┌───────────────────┐
                         │  Multi-task Heads │
                         └─────────┬─────────┘
                                   │
              ┌────────────┬───────┼───────────┬─────────┐
              ▼            ▼       ▼           ▼         ▼
          Category    SubCategory ArticleType Colour    Usage
              │            │       │           │         │
              └────────────┴───────┴───────────┴─────────┘
                                   │
                                   ▼
                              Top-K JSON
```

The model currently predicts five fashion attributes:

| Task | Classes |
|---|---:|
| Master Category | 4 |
| Sub Category | 26 |
| Article Type | 54 |
| Base Colour | 46 |
| Usage | 7 |

The final cleaned dataset contains **41,957 usable records**.

---

# 🛡️ Why CLIP?

One of the biggest lessons from the original project was that a classifier should not blindly classify everything it sees.

A traditional classifier trained only on fashion classes can still be forced to choose one of those classes when given an unrelated image.

That is exactly how the old **apple / pen** problem happened.

So CLIP is used as the first layer of the pipeline.

### CLIP = Contrastive Language–Image Pre-training

Instead of immediately asking:

> “Which fashion class is this?”

we first ask:

> **“Does this image look like clothing/fashion or something else?”**

The current guardrail uses the pretrained:

```text
openai/clip-vit-base-patch32
```

with candidate concepts representing:

```text
Clothing
Non-Clothing
```

A confidence threshold is used as a gate before the fashion classifier is allowed to run.

Conceptually:

```text
Apple 🍎
   ↓
CLIP
   ↓
Non-Clothing
   ↓
Reject

Shirt 👕
   ↓
CLIP
   ↓
Clothing
   ↓
Continue to ViT
```

This does **not** make CLIP a perfect out-of-distribution detector, but it provides an important first layer of protection against unrelated inputs.

---

# 👁️ Why Vision Transformer (ViT)?

The original project used multiple CNN models.

For the refurbished version, the project moved to a pretrained **Vision Transformer (ViT)**.

The model used is:

```text
google/vit-base-patch16-224
```

The idea behind transfer learning is simple:

Instead of teaching a vision model everything from zero, start with a model that has already learned useful visual representations from a large image corpus.

Then adapt it to the fashion domain.

The ViT-Base architecture used here contains:

- 12 Transformer layers
- 768-dimensional hidden representations
- 224 × 224 image inputs

Rather than creating five completely independent neural networks, the project uses one shared ViT backbone with five classification heads.

```text
                 Pretrained ViT
                      │
                Shared features
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   Category       SubCategory    ArticleType
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
     Colour          Usage       ...
```

This is a **multi-task learning** setup: one shared visual backbone produces features, while five classification heads specialise in the five fashion attributes.

---

# 🔧 How the ViT Was Fine-Tuned

The Hugging Face ViT starts with pretrained visual knowledge.

The project then adds custom classification heads for the fashion tasks:

```text
Master Category → 4 outputs
Sub Category    → 26 outputs
Article Type    → 54 outputs
Base Colour     → 46 outputs
Usage           → 7 outputs
```

During training:

1. The image is processed into ViT-compatible tensors.
2. The pretrained ViT extracts visual features.
3. Each classification head receives those shared features.
4. Each head produces logits for its own task.
5. Cross-entropy losses are calculated.
6. The losses are combined and used to update the model.
7. The pretrained backbone and task-specific heads are fine-tuned on the fashion dataset.

The final checkpoint therefore contains more than just the original Hugging Face weights. It contains the learned state of the **custom multi-task fashion model** after training.

---

# 🧪 Training Journey: Three Experiments

The training was carried out in a **Google Colab environment using a T4 GPU**.

Rather than assuming the first model was good enough, the project was treated as a sequence of controlled experiments.

## Experiment 1 — Baseline ViT

The first experiment used:

```text
Pretrained ViT
+
Original training data
+
Standard CrossEntropyLoss
```

Result:

**Test Average Macro F1 = 73.08%**

This became the baseline.

Task-level Macro F1:

| Task | Macro F1 |
|---|---:|
| Master Category | 99.81% |
| Sub Category | 89.13% |
| Article Type | 87.71% |
| Base Colour | 39.75% |
| Usage | 49.02% |

The model was already strong on broader categories, but the results exposed weaker performance in **Base Colour** and **Usage**.

---

## Experiment 2 — Data Augmentation

The next idea was to improve generalization by making the training images more varied.

Training-only transformations included:

- Random resize/crop
- Horizontal flip
- Small rotation
- Mild brightness/contrast changes

Aggressive colour changes were intentionally avoided because **colour itself is one of the prediction targets**.

Result:

**Validation Average Macro F1 = 71.58%**

This was worse than the baseline.

That was useful.

It showed that more image variation was not automatically solving the core problem.

The analysis suggested that **long-tail class imbalance and visually similar colour classes** were more important issues.

---

## Experiment 3 — Class-Weighted Loss

The third experiment kept the original images but changed what the model considered important during learning.

Instead of giving every class equal influence regardless of how frequently it appears, class weights were calculated from the training split.

Conceptually:

```text
Common class  → lower weight
Rare class    → higher weight
```

This tells the loss function:

> **“Don't let the large classes dominate learning. Mistakes on rare classes matter too.”**

Result:

**Test Average Macro F1 = 75.45%**

That is an improvement of:

**73.08% → 75.45%**

or

**+2.37 percentage points**

Task-level Macro F1:

| Task | Baseline | Class-Weighted |
|---|---:|---:|
| Master Category | 99.81% | 99.91% |
| Sub Category | 89.13% | 92.21% |
| Article Type | 87.71% | 88.37% |
| Base Colour | 39.75% | 37.50% |
| Usage | 49.02% | 59.28% |

The class-weighted model is therefore the **current best model**.

Interestingly, Base Colour remained the main weakness, showing that the problem is not only class imbalance. Several colour labels are also visually similar.

---

# 📊 Why Macro F1?

Accuracy alone can be misleading when a dataset is heavily imbalanced.

For example, in the Usage task:

```text
Casual       → 3291 examples
Smart Casual → 9 examples
Party        → 1 example
Travel       → 1 example
```

A model can get a high accuracy score by doing very well on the huge classes while performing terribly on tiny classes.

Macro F1 gives each class equal importance.

So the project uses:

- **Accuracy** → overall correctness
- **Precision** → diagnostic insight into false positives
- **Recall** → diagnostic insight into missed classes
- **Macro F1** → primary comparison metric across experiments

This made Macro F1 the most useful single metric for deciding which training strategy was actually better.

---

# 🗂️ Dataset

The project uses the **Fashion Product Images (Small)** dataset from Kaggle.

**Kaggle dataset:**  
`[ADD KAGGLE DATASET LINK HERE]`

After cleaning, the project retained:

```text
41,957 usable records
```

### Dataset Targets

| Attribute | Number of Classes |
|---|---:|
| Master Category | 4 |
| Sub Category | 26 |
| Article Type | 54 |
| Base Colour | 46 |
| Usage | 7 |

### Data Split

```text
Train      → 33,565
Validation → 4,196
Test       → 4,196
```

The split was fixed and deterministic for fair comparison between experiments.

The dataset pipeline also verified image availability, and the final set contained **no missing images for the selected records**.

---

# ⚙️ Training Setup

Training was performed in **Google Colab** with a T4 GPU.

Core training configuration:

```text
Model                  : google/vit-base-patch16-224
Framework              : PyTorch
Optimizer              : AdamW
ViT learning rate      : 2e-5
Classification heads LR: 1e-4
Weight decay            : 0.01
Batch size              : 32
Epochs                  : 4
AMP / Mixed Precision   : Enabled
DataLoader workers      : 2
Pinned memory           : Enabled
Scheduler               : CosineAnnealingLR
Early stopping          : Used
```

Four epochs were not chosen because “four is the magic number”.

Validation performance improved through the fourth epoch, while validation loss had already started to plateau. The experiment therefore stopped at that point rather than blindly spending more compute and risking overfitting.

---

# 🚀 Final Inference Pipeline

The final application combines the trained model with the CLIP guardrail.

```text
User uploads image
        ↓
     CLIP
        ↓
Fashion / Non-Fashion?
        │
   ┌────┴─────┐
   │          │
  No         Yes
   │          │
Reject      ViT
              ↓
       Five predictions
              ↓
       Top-K predictions
              ↓
        JSON response
```

The backend currently returns **Top-K = 4** predictions for each task.

Example:

```json
{
  "articleType": [
    {
      "label": "Tshirts",
      "confidence": 0.91
    },
    {
      "label": "Tops",
      "confidence": 0.05
    },
    {
      "label": "Tunics",
      "confidence": 0.02
    },
    {
      "label": "Shirts",
      "confidence": 0.01
    }
  ]
}
```

These confidence values are the model's softmax scores; they should not be interpreted as perfectly calibrated probabilities.

---

# 🖥️ Application

A Streamlit interface was built around the inference pipeline.

Current UI features include:

- Image upload
- CLIP clothing/non-clothing validation
- Fine-tuned ViT predictions
- Top-K prediction control
- Confidence display
- Confidence bars
- “Before Fine-Tuning” comparison using the original Google ViT ImageNet head
- Explanatory feature/tooltips
- Warning for multi-product/outfit images

### Important Limitation

The current model is designed around **one fashion product per image**.

For outfit photos or images containing multiple products, predictions can become mixed because the classifier produces a single set of attribute predictions for the image rather than detecting and classifying each object separately.

---

# 🤗 Hugging Face

The trained class-weighted model is hosted on Hugging Face.

### Model Repository

`KrIthIn-108/fashion-product-classifier-vit`

**Hugging Face:**  
[Model 🤖](https://huggingface.co/KrIthIn-108/fashion-product-classifier-vit)

The repository contains the trained model checkpoint used for inference.

The model checkpoint is based on the custom multi-task architecture built on:

```text
google/vit-base-patch16-224
```

The checkpoint represents the fine-tuned fashion model rather than the original pretrained ViT.

---
# 📈 Current Results

The current best model is the **Class-Weighted ViT**.

**Test Average Macro F1: 75.45%**

The model is particularly strong at:

- Master Category
- Sub Category
- Article Type

Usage also improved significantly after class weighting.

The biggest remaining weakness is:

**Base Colour**

This is partly caused by severe class imbalance and partly by visually similar colour categories such as different shades of blue, grey, brown, metallic tones, and related labels.

---

# 🧩 What This Project Taught Me

The interesting part of this project was not simply replacing CNN with ViT.

The bigger lesson was the workflow:

```text
Old model
   ↓
Real-world failure
   ↓
Identify the failure mode
   ↓
Clean the data
   ↓
Establish a baseline
   ↓
Run controlled experiments
   ↓
Analyze errors
   ↓
Change the training strategy
   ↓
Build an inference pipeline
   ↓
Add a guardrail
   ↓
Turn the model into an application
```

The apple and the pen that broke the original project ended up shaping the architecture of the new one.

---

# 🛠️ Tech Stack

```text
Python
PyTorch
Hugging Face Transformers
CLIP
Vision Transformer (ViT)
scikit-learn
Pillow
Streamlit
Hugging Face Hub
Google Colab
```

---

# 📁 Project Structure

```text
fashion-product-classifier/
│
├── app.py
├── predictor.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model/
    └── class_weighted_vit_model.pth   # local / ignored
```

The large model checkpoint is hosted separately on Hugging Face rather than being committed to the GitHub repository.

---

# ⚠️ Limitations

This project is intentionally presented as an evolving ML engineering system rather than a perfect production classifier.

Current limitations include:

- Base Colour remains the weakest task.
- Some classes have extremely low support.
- CLIP is a guardrail, not a perfect OOD detector.
- Confidence scores are not calibrated probabilities.
- One image is treated as one product.
- Outfit / multi-product images can produce mixed predictions.
- Very rare classes remain difficult to learn reliably.

These limitations are part of the reason the project is valuable: the goal was not to hide model failures, but to identify them and design the system around them.


## 🙌 Acknowledgements

- **Kaggle** — Fashion Product Images (Small) dataset
- **Hugging Face** — pretrained Transformer models and model hosting
- **Google ViT** — `google/vit-base-patch16-224`
- **OpenAI CLIP** — `openai/clip-vit-base-patch32`
- **Google Colab** — GPU-based experimentation and training

---

## 🔗 Links

| Resource | Link |
|---|---|
| Kaggle Dataset | [Dataset Link 📊](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) |
| Hugging Face Model | [Finetuned VIT 🤖](https://huggingface.co/KrIthIn-108/fashion-product-classifier-vit) |
| GitHub Repository | [GITHUB ⚙️](https://github.com/KrIthIn-108/Fashion-Product-Classifier) |
| Live Demo | [Demonstration ⏯️ ](https://stylesight-vit-01.streamlit.app/) |



---
## 🤖 A Note on the Build

And yes — this project was **vibe-coded with ChatGPT over a weekend**. 😄

But "vibe-coded" doesn't mean I just pressed copy-paste and hoped for the best.

My role was to **define the problem, decide the direction of the project, evaluate the outputs, question the results, make the technical decisions, and keep steering the implementation toward the system I actually wanted to build**.

ChatGPT helped me with the heavy lifting around coding, debugging, restructuring, explanations, and implementation ideas — but I was the one continuously providing the context, reviewing what was produced, identifying what didn't make sense, and deciding what stayed and what got changed.

So, essentially:

```text
Me
→ Problem + Direction + Decisions + Validation

ChatGPT
→ Coding + Debugging + Iteration + Implementation Assistance

```
---
And there was, of course, one tiny problem with this workflow...

I ran out of ChatGPT tokens. 💀

So while building this project, there was an unexpected deployment strategy:

```text
ChatGPT:
"You're out of tokens."

Me:
```
![Mr Bean Description](https://c.tenor.com/sECp6qPhI3UAAAAd/tenor.gif)
```text
Mr. Bean mode activated ☕🪑
        ↓
      WAIT
        ↓
"Alright, we're back."
        ↓
Continue coding
```
---

### The END as of now!!