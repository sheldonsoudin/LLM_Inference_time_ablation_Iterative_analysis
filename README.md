# LLM Inference-time Ablation Iterative Analysis

## Introduction

This project investigates inference-time ablation in large language models by comparing a scratch-trained GPT-style model against the pretrained EleutherAI Pythia-410M. The goal is to examine whether zeroing specific internal components at inference time reveals differences in how each model organizes and relies on its internal computations. In particular, the experiment focuses on ablations of attention and MLP submodules at early, middle, and late positions in the network, and measures the effect of each intervention on benchmark accuracy

A few aspects of the original proposal were simplified during implementation. Instead of comparing multiple pretrained models and a wider benchmark suite, the final experiment used only Pythia-410M as the pretrained reference model and evaluated performance only on HellaSwag and MMLU. 

The main purpose of this project is educational and it was done with the supervision of PhD candidate Maab Elrashid, and for Professor Mirco Ravanelli’s class Comp432(Machine Learning), at Concordia University in the 2026 winter semester. 

## Experiment Setup and Tools Used

The repository includes the project files, such as model.py, data.py, and train.py. 
The experiments were run from a Jupyter notebook in Google Colab, with Google Drive mounted for storage and checkpoint management. The Colab workflow also included  pulling the GitHub repository, installing dependencies with pip, running training from the notebook, resuming from saved checkpoints, and testing generation from the final saved model.

The main tools and libraries used in the Colab notebook were Google Colab, Google Drive, Git, PyTorch, Hugging Face Transformers, datasets, and huggingface_hub. The training script also used mixed-precision training utilities such as torch.autocast and GradScaler.

The experiment was carried out using Google Colab Pro+ and paid Google Drive storage. Training was performed on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU and took approximately 7 hours.

## Dataset Description (data.py)

The dataset is DCLM-Baseline 1.0, a large web-text corpus built mainly from Common Crawl and released by ML Foundations as a research baseline for language model pretraining. It contains plain-text web documents and is intended for large-scale training.

The data pipeline streams the mlfoundations/dclm-baseline-1.0 dataset, tokenizes it with the GPT-2 tokenizer, and converts the text into fixed-length 1024-token training chunks for causal language modeling. It uses a streaming IterableDataset, inserts EOS tokens between documents, and builds input-target pairs by shifting tokens for next-token prediction

## Pretraining From Scratch (model.py and train.py)

The scratch-trained model is a custom GPT-style decoder-only transformer with 24 layers, 16 heads, 1024 embedding dimensions, 4096 hidden dimensions in the MLP, and a 1024-token context window. It includes fused QKV self-attention, tied token/output embeddings, custom LayerNorm, GELU activations, Flash Attention support, and an alternating pattern of dense and locally sparse attention layers. Its design draws from nanoGPT, Raschka’s LLMs-from-Scratch, and Brown et al. (GPT-3)

The scratch model was trained using train.py. Training was run for 25,000 steps with a batch size of 4 and 8 gradient accumulation steps, targeting 819.2 million tokens in total, with checkpoints saved every 1,000 steps

## Pythia 410 Million Parameters

Pythia-410M is a transformer-based causal language model from EleutherAI’s Pythia suite, a family of models designed for interpretability research. According to the model card, Pythia-410M has about 405 million total parameters (302 million non-embedding parameters), uses a GPT-NeoX architecture with 24 layers, model width 1024, and 16 attention heads, Biderman et al. describe the broader Pythia project as a research-oriented setup for studying how language models develop across scale and training rather than as a chat-style product model. 

## Ablation Protocol 

In this experiment I compare two models: a scratch-trained GPT and EleutherAI/Pythia-410M. For each model, I ran a baseline and 12 single module ablations, totaling in 13 conditions per model. The ablations are organized by layer position, that is early, middle and late. They are also organized by module type: attention QKV projection, attention output projection, down/projection.  Each ablation is done by a lesion-style intervention, meaning that I save the target module’s parameters, zero them out in place, run the evaluation, then restore the original weights so the next ablation starts from the untouched model. That makes this an isolated ablation rather than retraining after removing a component. This kind of protocol is consistent with standard ablation logic in neural-network analysis, where one disables a component and measures the downstream change in behavior. (arXiv)

## Evaluations Benchmarks 

HellaSwag and MMLU are the two benchmarks used to evaluate the effect of an ablation. 

First, HellaSwag was introduced as a common sense inference benchmark built with adversarial filtering (arXiv). It is a continuation benchmark where the model chooses the most plausible ending from four points. In this experiment, I used 100 shuffled validation examples scoring each ending by the model’s continuation log-probability and selecting the highest-scoring option. 

Second, this experiment  runs MMLU, a multiple-choice benchmark designed to test broad knowledge and reasoning across many academic subjects. In this expereiemt, I only used three subjects; high_school_biology, high_school_mathematics, and computer_security; with 10 questions per subject. The benchmark scores answer options A/B/C/D by continuation likelihood and picks the best one. The underlying MMLU benchmark is the “Massive Multitask Language Understanding” test introduced by Hendrycks et al. (arXiv)

After each run, accuracy for each benchmark and also computes delta versus that model’s own baseline are computed. Results are then saved to CSV and visualized with bar charts

## Results and Visualisations

Full tabulated results are available on collab notebook. 



This first figure (above)  is a bar chart showing the average HellaSwag accuracy after different model ablations. The bars on the left correspond to the Pythia model, while the bars on the right correspond to the scratch GPT model. Overall, the scratch GPT model consistently underperforms Pythia. A few other notable patterns stand out in this visualization: Pythia’s baseline accuracy is the highest, Pythia’s early-layer ablations produce the largest performance drops, especially from early_attn_out through early_mlp_up, and the scratch GPT model shows little change across most ablations.



This second figure (above) is a bar chart showing the average MMLU accuracy after different model ablations. As in the previous figure, the bars on the left correspond to the Pythia model, while the bars on the right correspond to the scratch GPT model. Unlike the HellaSwag results, the MMLU accuracies are much more similar across the two models and across the ablation settings. Most bars cluster around roughly the same range, and the large overlapping error bars suggest that the differences between ablations are small and not very stable. Overall, this visualization indicates that neither model shows a strong or consistent sensitivity to any single ablation on MMLU, so the benchmark does not reveal a clear separation in internal component importance the way HellaSwag does. This may be due to the relatively small sample size used for MMLU. 

The two figures above show the average change in accuracy relative to each model’s own baseline for both HellaSwag and MMLU across the different ablation settings. The first panel corresponds to HellaSwag, and the second panel corresponds to MMLU. In each panel, the blue bars represent Pythia-410M and the orange bars represent scratch GPT. Values below zero indicate that an ablation hurt performance, while values above zero indicate that an ablation slightly improved performance. The dashed horizontal line at zero marks no change from baseline.

Again, the clearest pattern appears in the HellaSwag panel. Pythia shows several substantial negative drops, especially for the early-layer ablations, with early_mlp_down, early_mlp_up, and early_attn_out producing the largest decreases relative to baseline. This suggests that Pythia relies heavily on these early components for HellaSwag performance. 

In contrast, the scratch GPT model remains much closer to zero across most ablations, with only very small positive or negative changes, indicating that its performance is comparatively insensitive to the removal of individual components. 

In the MMLU panel, both models stay much closer to zero overall, and the large error bars suggest that most differences are small and unstable. Overall, these figures support the earlier conclusion that Pythia has clearer functional specialization, especially in early layers on HellaSwag, whereas the scratch GPT model appears flatter and less sensitive to the tested ablations.

## Analysis of results
The conclusions of this experiment are as follows: On HellaSwag in particular, Pythia-410M performs better overall than the scratch GPT model, suggesting stronger pretrained representations. Pythia is also much more sensitive to early-layer ablations, especially in the early MLP components. This is likely related to its stronger pretrained structure: because it is the better-performing model, disrupting key components has a larger impact on overall performance. In addition, HellaSwag shows the clearest and most consistent ablation pattern, whereas the scratch GPT model changes very little across most ablations. This likely indicates that the scratch GPT has a less organized or less specialized internal structure than Pythia.

## Challenges 
Further testing and larger samples would be needed to confirm these results. As the project progressed, the scope of the experiment kept expanding, making it difficult to explore every direction. To obtain more meaningful conclusions, it would have been preferable to use a larger MMLU sample size and run more trials overall. The main reasons for not extending the experiments further were limitations in compute and time, especially given that this was primarily a learning project. Training the scratch model also required several attempts to get right, and, from a practical standpoint, time had to be balanced against other coursework, projects, and study commitments.

## Sources

### Ablation
Meyes et al. (2019), Ablation Studies in Artificial Neural Networks.  https://arxiv.org/abs/1901.08644 
Michel, Paul, et al. “Are Sixteen Heads Really Better than One?” ArXiv.org, 4 Nov. 2019, http://arxiv.org/abs/1905.10650 

### Benchmarks
Zellers et al. (2019), HellaSwag: Can a Machine Really Finish Your Sentence? https://arxiv.org/abs/1905.07830 
Hendrycks et al. (2021), MMLU benchmark. https://arxiv.org/abs/2009.03300 
CAIS MMLU dataset card (cais/mmlu). https://huggingface.co/datasets/cais/mmlu 

### Pretrained Model (Loaded)
Biderman et al. (2023), “Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.” Available: https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf 
EleutherAI Pythia-410M model card. https://huggingface.co/EleutherAI/pythia-410m 

### Model Architecture (Trained from Scratch)
Karpathy, nanoGPT: fused QKV, weight tying, Flash Attention, bias flag.
Raschka, LLMs-from-Scratch: custom LayerNorm, transparent GELU.
Brown et al. (GPT-3): alternating dense/sparse attention, 2048 context, scale√k.

### Data Processing
Hugging Face Datasets: loading and streaming. https://huggingface.co/docs/datasets/loading , https://huggingface.co/docs/datasets/stream 
DCLM Baseline dataset (ML Foundations). https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
Karpathy, nanoGPT: contiguous token buffer, chunking into (block_size + 1). https://github.com/karpathy/nanoGPT 
Brown et al. (GPT-3), Radford et al. (GPT-2): causal LM, next-token prediction with shifted targets. GPT-2 tokenizer (vocab_size = 50257).

### Training Loop 
Karpathy, nanoGPT : training loop structure, gradient accumulation, cosine LR schedule. https://github.com/karpathy/nanoGPT/blob/master/train.py 
PyTorch:  optimizer step, backward pass, gradient clipping. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
PyTorch AMP: torch.autocast, GradScaler. https://pytorch.org/docs/stable/amp.html 
Hugging Face Transformers: GPT-2 tokenizer loading. https://huggingface.co/docs/transformers/main_classes/tokenizer 

### AI Resources
These resources were used in review and debugging of code, as well as for searching for pertinent information and sources. 
OpenAI’s ChatGPT 
Anthropic’s Claude 

### Educational Resources
These resources were used to help in designing and implementing this project:

Supervision by PhD Candidate Maab Elrashid
Laboratories from COMP432, Mirco Ravanelli, Concordia University, 2026.
S. Kublik and Shubham Saboo, Gpt-3. O’Reilly Media, 2022.
S. Raschka, Build a Large Language Model (From Scratch). Manning, 2024.
J. Alammar and Maarten Grootendorst, Hands-On Large Language Models. “O’Reilly Media, Inc.,” 2024



