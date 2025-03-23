## ICL-oriented KV Cache Reusing
InferLog proposes Prefix-aware ICL Refinement(PAIR) policy to refine the log parsing prompt by adjusting the contents and order of ICL examples. For each current request, InferLog first identifies the examples of historical requests with the highest prefix cache hit rate probability, then performs modifying and reordering operations to update the current examples to align with it. By doing so, the ICL component can significantly boost the prefix cache hit rate

## PAIR
![pair](https://github.com/wiluen/InferLog/blob/main/resource/pair.png)

## Main code:
- inferlog.py: implement of PAIR
- postprocess.py: post processing for log parsing
- prepare_icl.py: selecting ICL for distict log messages in Loghub-2k. The main code is from [https://github.com/logpai/logparser/blob/main/logparser/DivLog/DivLog.py]
- prompt.zip: the processed results

## Run
Download the open-source model and launch it using vLLM
```
vllm serve qwen/Qwen2.5-14B-Instruct --enable-prefix-caching
```
Run the command to perform log parsing on a datset and get inference performance and parsing accuracy
```
python inferlog.py --dataset 'Spark' --port 8080 --capacity 200 --concurrency 10 --enable_pair
```



