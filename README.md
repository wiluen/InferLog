# InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching

Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: 1) the privacy risks associated with commercial LLMs, driving the adoption of locally deployment, and 2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system.

In this study, we present InferLog, the first LLM inference optimization method for online log parsing. InferLog accelerate inference by designing 1) A prefix-aware ICL refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. 2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-relate configuration for dynamic log parsing workloads. The experimental results based on Loghub-2k dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
![overview of InferLog](https://github.com/wiluen/InferLog/blob/main/resource/overivew.png)

## Insight
Reusing prefix KV caching of ICL part, not limited to common instruct.

<img src="https://github.com/wiluen/InferLog/blob/main/resource/insight.png" alt="Insight of InferLog" width="800px" style="text-align: center">


## start

### Environment Version
- vLLM(v0.6.3.post1)

### Install dependencies (with python 3.9)
> pip install -r requirements.txt

### Benchmark
We use loghub2k as the dataset and employ in-context learning technique to find demonstration examples based on similarity in the candidate set. The raw logs are available for downloading at [https://github.com/logpai/loghub]

### Main File
- src/prefix_cache_reusing: implementation of Prefix-Aware ICL Refinement(PAIR) strategy to refine the ICL examples to improve prefix cache hit rate.
- src/fast_config_tuning: implementation of configuration tuning base on AttMAML and SMBO.
