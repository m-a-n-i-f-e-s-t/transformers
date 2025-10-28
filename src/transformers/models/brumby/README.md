# Brumby-14b-base

**Model Developer**: [Manifest AI](https://manifestai.com/)

**Number of Parameters**: 14B

**Release Page**: https://manifestai.com/articles/release-brumby-14b/

## Model Overview

**Brumby-14b-base** is a completely attention-free LLM whose performance is competitive with state-of-the-art models. This 
model, which we call **Brumby-14B-Base**, has a familiar Transformer-style architecture, except it uses
[power retention](https://manifestai.com/articles/release-power-retention/) 
layers instead of attention layers. It is available [on Huggingface](todo).
Here is how it compares to other models at a similar scale on popular benchmarks.

### Evaluation Results
| Task/Model     | Brumby-14B-base | falcon-mamba-7b | mamba-codestral-7b-v0.1 | nvidia-nemotron-nano-12b-v2-base | qwen3-14b-base | glm-4.5-air-base | mistral-nemo-base-2407  |   |   |
|----------------|-----------------|-----------------|-------------------------|----------------------------------|----------------|------------------|-------------------------|---|---|
| ARC            | 0.89            | 0.74            | 0.47                    | 0.93                             | 0.94           | 0.92             | 0.76                    |   |   |
| GSM8K          | 0.88            | 0.52            | 0.23                    | 0.84                             | 0.84           | 0.83             | 0.54                    |   |   |
| GSM8K Platinum | 0.87            | 0.54            | 0.24                    | 0.87                             | 0.88           | 0.85             | 0.57                    |   |   |
| HELLASWAG      | 0.77            | 0.8             | 0.7                     | 0.82                             | 0.81           | 0.85             | 0.83                    |   |   |
| MMLU           | 0.71            | 0.6             | 0.46                    | 0.78                             | 0.78           | 0.77             | 0.64                    |   |   |
| MMLU Pro       | 0.36            | 0.23            | 0.19                    | 0.53                             | 0.55           | 0.51             | 0.35                    |   |   |
| MBPP           | 0.57            | 0.4             | 0.48                    | 0.71                             | 0.75           | 0.73             | 0.54                    |   |   |
| MATH           | 0.62            | 0.19            | 0.12                    | 0.26                             | 0.54           | 0.47             | 0.2                     |   |   |



### Training


The training budget for this model was $4,000, trained 60 hours on a cluster of 32 H100s. (For comparison, training 
an LLM of this scale from scratch typically costs ~$200k.) We were able to achieve this low cost thanks to the 
mathematical similarity between attention and power retention. We used a technique we call _retraining_, which 
repurposes the weights of a pretrained Transformer as an initialization for power retention.


The initial weights for Brumby-14B-Base came from Qwen3-14B-Base. The jumps in loss correspond to changes in the 
underlying training distribution, following the three-phase dataset of
[Nemotron Nano](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf).
After 3000 steps of training, it reaches the same training loss on this data as Qwen3-14B-Base.
This trend is mirrored by performance on downstream evaluations.


So what is power retention? Similar to attention, power retention is a layer that takes $Q,K,V \in R^{t \times d}$ as 
inputs, and gives $Y \in R^{t \times d}$ as an output. It also accepts a gating signal $g \in R^t$.
It is a “true” RNN, in that every prediction can be influenced by information arbitrarily far back in the past.
The state of the RNN is a matrix $S \in R^{d \times D}, which is updated according to

$$
S_t = g_t S_{t-1} + V_t \phi_p(K_t)^T \qquad Y_t = S_t Q_t
$$

The function $\phi_p: R^d \to R^D$ is related to the tensor power (the “power” in power retention).
The power $p$ controls the dimension $D$, giving us a hyperparameter to scale the state size of the RNN,
just as one might use the width to scale the parameter count.
For our experiments, power $p=2$ resulted in the optimal state size for the model.

What makes this a retention layer, as opposed to just a recurrent layer, is that it also has an attention form. 
This second formulation is critical to any hardware-efficient implementation. If you want to learn more about the 
power retention layer and how to implement it efficiently, see our preprint paper and related blog post. Our 
hardware-efficient power retention kernels are available open-source and can be installed with `pip install retention`.

### Reproduce Evaluation

To reproduce evaluation results, first install the latest version of [lm-evaluation-harnesss](https://github.com/EleutherAI/lm-evaluation-harness), then run:

```
lm_eval --model hf --model_args "pretrained=manifestai/brumby-14b-base,trust_remote_code=True" --tasks "gsm8k" --batch_size 8
```

Note that the huggingface generation implementation relies on compiling triton kernels and is thus slow to start and inefficient, we are planning to rollout efficient inference kernels in the coming weeks.


### Coming soon

**Fast long-context inference:** Our fastest power retention inference kernels are hundreds of times faster than 
equivalent attention kernels on long contexts. We will update the architecture to incorporate these fast kernels.

**Long-context SFT:** A finetune of Brumby-14B-Base at context length 1,000,000 is no more expensive (per token) than a 
finetune of Qwen3-14B-Base at context length 10,000. We will release a long context SFT toolkit so that anyone can 
perform these long-context finetunes, unlocking new capabilities for LLMs in domains like search and coding.

**VLLM integration:** A robust inference engine is an essential complement to any SOTA LLM. We are developing kernels 
to integrate power retention with VLLM. Expect to see both unmatched inference speeds and reduced memory 
requirements, allowing more users to fit on each GPU.

**The Brumby Band:** Brumby-14B-Base is just the first of a coming family of models. In the coming weeks, we will 
retrain and release power retention base models at a variety of scales, from as small as 1B parameters up to >100B 
parameters.