# BLIP-2 — Bootstrapping Language–Image Pretraining with Frozen Image Encoders and LLMs

> DS 5690 — Gen AI Models in Theory & Practice (2025F)  
> Zhiting Zhou
> Li et al., 2023 — “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models” (arXiv:2301.12597)

---

## Question 1
How many different ways can we make a large language model convert an image into text?
<details>
  <summary>Answer</summary>
  
   Firstly, internet data often contains a large number of inaccurate or irrelevant image-text pairs, leading models to learn incorrect or imprecise information. Secondly, the vast scale and complexity of the data make extracting useful information costly, limiting the efficiency of model learning. 
   1. End-to-End Fine-Tuning: Train the entire vision–language model jointly — both the image encoder and the language model.
   2. Feature Injection: Keep the large language model (LLM) frozen and feed image features directly as soft prompts or through added cross-attention layers.
   3. Adapter / Bridge Module: Insert a small trainable transformer that translates visual embeddings into the LLM’s text space (this is the strategy used by BLIP-2).
   4. Visual Tokenization: Treat image patches as tokens and feed them directly to a multimodal LLM.

</details>

## Overview 

BLIP‑2 makes multimodal training much cheaper by keeping the image encoder and the LLM frozen, and learning only a lightweight Querying Transformer (Q‑Former) to bridge from images to language. It pretrains Q‑Former in two stages: (1) representation learning with an image encoder (ITC/ITM/ITG), and (2) generative learning by prompting a frozen LLM with projected visual queries. Despite far fewer trainable parameters, BLIP‑2 matches or beats larger end‑to‑end models on zero‑shot VQA, captioning, and retrieval.

- **Context.** Vision–language pretraining (VLP) models got huge and expensive; end‑to‑end training is prohibitive.  
- **Problem.** How to achieve strong multimodal performance without full end‑to‑end training?  
- **Approach.** Freeze a **strong image encoder** (e.g., CLIP ViT) and a **strong LLM** (e.g., Flan‑T5/OPT), and train a small **Q‑Former** that extracts a compact set of learned visual queries and prompts the LLM.  
- **Results.** State‑of‑the‑art zero‑shot VQA, strong captioning and retrieval—while using vastly fewer trainable params than e2e models.

---

## Question 2
Why do you think BLIP-2 keeps both the image encoder and the language model frozen instead of fine-tuning them end-to-end?
<details>
  <summary>Answer</summary>
- Efficiency, avoiding catastrophic forgetting, and reusing strong unimodal models.
- Bridging rather than retraining.
</details>

## Architecture (High‑Level)


![BLIP-2 Overview](./images/blip2.PNG)


## Two‑Stage Pretraining

### Stage 1 — *Vision–Language Representation Learning* (with frozen image encoder)
Jointly optimize three objectives to make queries language‑relevant:  
- **ITC** (Image–Text Contrastive): align global image/text embeddings.  
- **ITM** (Image–Text Matching): binary “match?” classification with hard negatives.  
- **ITG** (Image‑Grounded Generation): force queries to contain all info needed to generate text.

### Stage 2 — *Vision→Language Generative Learning* (with frozen LLM)
- Project query features to the LLM’s token dim and **prepend** them to the text tokens (soft prompts).  
- Train only **Q‑Former + projection**, keeping LLM **frozen**, to enable captioning/QA conditioned on visual prompts.

---

## Formal Pseudocode

```python
# Notation:
#   E_img: frozen image encoder (ViT)
#   Q: Q‑Former (trainable)
#   P: linear projection from Q‑Former output to LLM token dim (trainable)
#   LLM: frozen large language model (decoder or encoder‑decoder)
#   ITC, ITM, ITG: stage‑1 losses; LM_loss / PrefixLM_loss: stage‑2 losses

# ---------- Stage 1: Representation Learning ----------
for image, text in D_image_text:
    V = E_img(image)                  # frozen features
    Z = Q(image_feats=V, text=None)   # queries attend to V via cross‑attention
    loss = ITC(Z, text) + ITM(Z, text) + ITG(Z, text)
    update(Q)                         # only Q‑Former is updated

# ---------- Stage 2: Generative Learning ----------
for image, text in D_image_text:
    V = E_img(image)                  # frozen features
    Z = Q(image_feats=V, text=None)   # extract visual queries (language‑relevant)
    V_prompt = P(Z)                   # project to LLM embedding size
    loss = LM_loss(LLM(prompt=V_prompt, text=text))  # or PrefixLM for encoder‑decoder
    update(Q, P)                      # only Q‑Former + projection are updated
```            


### Method
- The BLIP-2 framework, short for Bootstrapping Language-Image Pre-training 2, introduces an efficient approach for aligning vision and language without the need for end-to-end training of massive multimodal models. Instead of jointly training an image encoder and a language model from scratch, BLIP-2 leverages two powerful pre-trained unimodal components: a frozen image encoder (such as CLIP ViT-L/14 or EVA-CLIP ViT-g/14) and a frozen large language model (LLM) such as OPT or FlanT5. Between these two frozen modules lies the only trainable component — the Querying Transformer (Q-Former) — which serves as a lightweight bridge that learns how to translate visual representations into language-understandable embeddings.

The training of BLIP-2 proceeds in two stages, each targeting a distinct aspect of cross-modal alignment.

Stage One: Vision–Language Representation Learning.
In this stage, the model aims to teach Q-Former how to extract the visual information that is most relevant to textual semantics. Given a pair of image and caption, Q-Former interacts with the frozen image encoder through cross-attention layers using a fixed set of learnable query tokens. It is optimized jointly by three complementary objectives:
(1) Image-Text Contrastive Learning (ITC) to align image and text embeddings in a shared latent space;
(2) Image-Grounded Text Generation (ITG) to generate captions conditioned on visual features; and
(3) Image-Text Matching (ITM) to discriminate whether an image–text pair is correctly matched.
By combining these objectives, Q-Former gradually learns to focus on semantically meaningful regions of the image while filtering out irrelevant visual details.

Stage Two: Vision-to-Language Generative Learning.
After Q-Former has learned to represent images in a language-related manner, the second stage connects it to a frozen LLM to endow the whole system with natural-language generation ability. The key idea is to project the output of Q-Former — a set of 32 visual query embeddings — into the same dimensional space as the LLM’s word embeddings, and then prepend these visual embeddings to the text input sequence. They act as soft visual prompts that condition the LLM on the image content.

To illustrate, consider an input image showing a cat wearing sunglasses.

The frozen image encoder first extracts dense visual features.

Q-Former compresses these features into 32 informative queries that summarize “what the image is about.”

These queries are linearly mapped to the token space and inserted before the text prompt fed into the LLM, for example:
“visualprompts A cat wearing sunglasses”.

The LLM (e.g., OPT or FlanT5) then generates the natural language output such as “A cat wearing sunglasses sitting on a beach.”
During training, the system minimizes the standard language-modeling loss so that the generated text aligns with ground-truth captions or answers. In essence, the second stage teaches Q-Former to speak the LLM’s language — to express visual information in a form that the LLM can interpret.

This two-stage strategy offers several advantages. Because both the image encoder and the LLM remain frozen, the pre-training cost is drastically reduced. Moreover, it avoids catastrophic forgetting in the LLM while still achieving strong vision–language alignment. Through this process, BLIP-2 can perform a wide range of zero-shot multimodal tasks — from image captioning and visual question answering to instruction-based image-to-text generation — demonstrating that the lightweight Q-Former is sufficient to bridge the gap between powerful unimodal models.
- **Cross‑attends** to frozen ViT features to pull out language‑relevant information.  
- Feeds projected queries as **soft visual prompts** to the frozen LLM.


---

## Critical Analysis (What’s strong / What’s missing)
**Strengths**
- Large *frozen* components preserve pretrained knowledge; few trainable params → compute‑efficient.
- Two‑stage scheme reduces catastrophic forgetting and improves zero‑shot performance.
- Modular: can “harvest” better ViTs/LLMs over time.

**Limitations / Open Questions**
- Single‑pair pretraining lacks multi‑image interleaving → weak in‑context multimodal examples; limited few‑shot gains.  
- Quality still bounded by the LLM’s knowledge (bias, hallucination).  
- Visual reasoning can fail on novel or complex scenes; struggles with very long visual contexts.

---

## Impact
- Helped establish the **“frozen LLM + visual adapter”** recipe used by LLaVA, InstructBLIP, MiniGPT‑4, etc.  
- Lowered the barrier to building visual assistants on modest compute while staying competitive with very large e2e models.

---

## Demo (Captioning + VQA)
Use the notebook [`demo.ipynb`](./demo.ipynb). It loads `Salesforce/blip2-flan-t5-xl` from Transformers, captions an image, and answers a visual question.

**Environment (suggested)**
```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install -U transformers accelerate pillow safetensors
```

**Run**
1. Open the notebook, set `IMAGE_PATH` to a local file (or URL).  
2. Run the caption cell.  
3. Set a `question` string and run the VQA cell.

---

## Resource Links
1. Paper: https://arxiv.org/abs/2301.12597  
2. BLIP‑2 in LAVIS (Salesforce): https://github.com/salesforce/LAVIS/tree/main/projects/blip2  
3. 🤗 Model Card (Flan‑T5 XL): https://huggingface.co/Salesforce/blip2-flan-t5-xl  
4. 🤗 Model Card (OPT 2.7B): https://huggingface.co/Salesforce/blip2-opt-2.7b  
5. Colab‑style starter (community): https://colab.research.google.com/github/salesforce/LAVIS/blob/main/docs/source/tutorials/BLIP2_captioning.ipynb

---
## Citations
1. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv preprint arXiv:2301.12597

2. Salesforce Research. (2023). BLIP-2 in LAVIS: Bootstrapping Language-Image Pre-training Implementation. Available at: https://github.com/salesforce/LAVIS/tree/main/projects/blip2

3. Salesforce Research & Hugging Face. (2023). BLIP-2 Flan-T5 XL Model Card. Available at: https://huggingface.co/Salesforce/blip2-flan-t5-xl

4. Salesforce Research & Hugging Face. (2023). BLIP-2 OPT 2.7B Model Card. Available at: https://huggingface.co/Salesforce/blip2-opt-2.7b

5.Salesforce Research Community. (2023). BLIP-2 Captioning Tutorial (Colab). Available at: https://colab.research.google.com/github/salesforce/LAVIS/blob/main/docs/source/tutorials/BLIP2_captioning.ipynb


---
