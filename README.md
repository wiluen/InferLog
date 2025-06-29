# InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching

Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: 1) the privacy risks associated with commercial LLMs, driving the adoption of locally deployment, and 2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system.

In this study, we present InferLog, the first LLM inference optimization method for online log parsing. InferLog accelerate inference by designing 1) A prefix-aware ICL refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. 2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-relate configuration for dynamic log parsing workloads. The experimental results based on Loghub-2k dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.

<img src="https://github.com/wiluen/InferLog/blob/main/resource/overivew.png" alt="Overview of InferLog" width="500px" style="text-align: center">

## Insight
Reusing prefix KV caching of ICL part, not limited to common instruct.

<img src="https://github.com/wiluen/InferLog/blob/main/resource/insight.png" alt="Insight of InferLog" width="500px" style="text-align: center">


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

### ICL-oriented KV Cache Reusing
InferLog proposes Prefix-aware ICL Refinement(PAIR) policy to refine the log parsing prompt by adjusting the contents and order of ICL examples. For each current request, InferLog first identifies the examples of historical requests with the highest prefix cache hit rate probability, then performs modifying and reordering operations to update the current examples to align with it. By doing so, the ICL component can significantly boost the prefix cache hit rate

#### Main code:
- inferlog.py: implement of PAIR
- postprocess.py: post processing for log parsing
- prepare_icl.py: selecting ICL for distict log messages in Loghub-2k. The main code is from [https://github.com/logpai/logparser/blob/main/logparser/DivLog/DivLog.py]
- prompt.zip: the processed results

#### Run
Download the open-source model and launch it using vLLM
```
vllm serve qwen/Qwen2.5-14B-Instruct --port 8001 --enable-prefix-caching
```
Run the command to perform log parsing on a datset and get inference performance and parsing accuracy
```
python inferlog.py --dataset 'Spark' --port 8001 --capacity 200 --concurrency 100 --enable_pair
```

### Fast Inference Config Tuning
To achieve a fast and tailored configuration optimization, we introduce Attention mechinism in Model-Agnostic Meta-Learning to identify optimal initial parameters of the model for the target tuning tasks and then adopt Sequential Model-based Optimization (SMBO) to update the meta-model through fewshot learning with newly collected data to rapidly recommend best
configuration in a few iterations.

#### Configuration parameters
<img src="https://github.com/wiluen/InferLog/blob/main/resource/conf.png" alt="Critical Config in vLLM" width="600px" style="text-align: center">

#### Main code:
- data_collect.py: collect data using LHS or random
- maml.py: original implement of MAML algorithm
- attmaml.py: implement of our Attention MAML algorithm
- attmamldrop_bo.py: AttMAML with MC dropout for SMBO
- env.py: Interaction with environment(apply config and conduct testing to get metrics)
- tuner.py: implement of SMBO, it can use the kernel of 'gp' or 'meta'


#### Steps:
1. Collect meta data using LHS sampling algorithm (data_collect.py)
2. Conduct training for AttMAML and save model (attmaml.py)
3. Online tuning (tuner.py and attmamldrop_bo.py)

#### Dataset
perform LHS to sample configuration and measure the performance of LLM inference for log parsing based on Qwen2.5-14B and vLLM




