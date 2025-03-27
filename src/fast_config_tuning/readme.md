
## Fast Inference Config Tuning
To achieve a fast and tailored configuration optimization, we introduce Attention mechinism in Model-Agnostic Meta-Learning to identify optimal initial parameters of the model for the target tuning tasks and then adopt Sequential Model-based Optimization (SMBO) to update the meta-model through fewshot learning with newly collected data to rapidly recommend best
configuration in a few iterations.

## Configuration parameters
<img src="https://github.com/wiluen/InferLog/blob/main/resource/conf.png" alt="Critical Config in vLLM" width="600px" style="text-align: center">

## Main code:
- data_collect.py: collect data using LHS or random
- maml.py: original implement of MAML algorithm
- attmaml.py: implement of our Attention MAML algorithm
- attmamldrop_bo.py: AttMAML with MC dropout for SMBO
- env.py: Interaction with environment(apply config and conduct testing to get metrics)
- tuner.py: implement of SMBO, it can use the kernel of 'gp' or 'meta'


## Steps:
1. Collect meta data using LHS sampling algorithm (data_collect.py)
2. Conduct training for AttMAML and save model (attmaml.py)
3. Online tuning (tuner.py and attmamldrop_bo.py)

## Dataset
perform LHS to sample configuration and measure the performance of LLM inference for log parsing based on Qwen2.5-14B and vLLM
