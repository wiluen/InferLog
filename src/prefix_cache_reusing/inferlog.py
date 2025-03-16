import asyncio
import time
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import logging
import argparse
import re
import json
import random
import csv
from collections import deque
import datetime
import os
from collections import OrderedDict
from postprocess import post_process,match_icl,reorder_and_replace,pair_elements
# from evaluator import evaluate, evaluate_all_datasets
from collections import Counter
from datetime import datetime
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def count_prefix_matches(a, b):
    """Targeting the common prefix, find the historical ICL that best matches the current ICL"""
    # a: current icl,  b:target icl
    a=pair_elements(a)
    b=pair_elements(b)
    match_count=0
    for k_b, v_b in b:
        matched = False
        for i, (k_a, v_a) in enumerate(a):
            if v_b == v_a: 
                match_count += 1 
                a.pop(i)  
                matched = True
                break  
        if not matched: 
            break
    return match_count

def count_normal_matches(a,b):
    """Targeting the maximum number of demo matches, find the historical ICL that best matches the current ICL"""
    # a:current icl, b:target_icl
    indexes = [1, 3, 5, 7, 9]
    cur_elements = set(a[i] for i in indexes if i < len(a))
    tar_elements = set(b[i] for i in indexes if i < len(b))
    match_count = len(cur_elements & tar_elements)
    return match_count

def pair_elements(a):
    return [(a[i], a[i + 1]) for i in range(0, len(a), 2)]

def reorder_and_replace(a, b):
    """Adjust the order of current ICL based on the prefix of target ICL"""
    a=pair_elements(a)
    b=pair_elements(b)
    remaining_a = a[:]
    result = []
    for k_b, v_b in b:
        matched = False
        for i, (k_a, v_a) in enumerate(remaining_a):
            if v_b == v_a:  
                result.append((k_b, v_b))  
                remaining_a.pop(i) 
                matched = True
                break  
        if not matched:
            break
    result.extend(remaining_a)
    result_flat=[result[0][0],result[0][1],
                 result[1][0],result[1][1],
                 result[2][0],result[2][1],
                 result[3][0],result[3][1],
                 result[4][0],result[4][1]]
    return result_flat


def cache_hit(key):
    global cache
    """Cache hit, move the accessed elements to the end of the queue"""
    cache.move_to_end(key) 
    cache[key]+=1
    return cache
   

def put(key):
    global cache
    """Insert elements, and if the cache is full, remove the oldest unused elements"""
    if len(cache) >= capacity:
        cache.popitem(last=False) 
    cache[key] = 0 
    return cache
    

def match_icl(cur_icl):
    global cache
    """Traverse OrderedDict, return matching ICL and count"""
    max_count = 0
    match_list = None

    for hist_icl in cache.keys(): 
        common_elements_count = count_prefix_matches(cur_icl, hist_icl)
        if common_elements_count==5:
            return hist_icl,5
        elif common_elements_count > max_count:
            max_count = common_elements_count
            match_list = hist_icl
    return match_list,max_count

def ICL_reorder(cur_icl):
    global cache
    """
    ICL reordering, divided into four situations
    1. Direct hit, operation: cache hit
    2. The demo is consistent, but the order is different. Operation: cache hit, adjust order
    3. There is no match, operation: Join the team
    4. Matched some, operation: Join the team, adjust the order
    """
    if len(cache)==0:
        # the first
        put(tuple(cur_icl))
        reorder_icl=cur_icl
    else:
        if tuple(cur_icl) in cache:
            # already exsitng
            cache_hit(tuple(cur_icl))
            reorder_icl=cur_icl 
        else:
            match_list,max_count=match_icl(cur_icl)
            if max_count==5:
                # match 5, reorder
                cache_hit(tuple(match_list))
                reorder_icl=match_list 
            elif max_count==0:
                # no match,add
                put(tuple(cur_icl))
                reorder_icl=cur_icl
            else:
                # match some,add,reorder
                reorder_icl=reorder_and_replace(cur_icl,match_list)
                put(tuple(reorder_icl))
    return reorder_icl



def evaluatePA(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['template'])
    if length == 0: return 0
    correct = 0
    for i in range(length):
        if result['template'][i] == groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0]:
            correct += 1
        # else:
        #     with open(f'exp/wrong_{dataset}.txt','a') as f:
        #         f.write('divlog:'+result['template'][i]+'\n')
        #         f.write('groundtruth:'+groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0]+'\n')
                
    return correct/length

# correctly identified templates over total num of identified template
def evaluatePTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx] not in oracle_tem_dict:
          oracle_tem_dict[groundtruth['EventTemplate'][idx]] = [groundtruth['Content'][idx]]
        else: oracle_tem_dict[groundtruth['EventTemplate'][idx]].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if result['template'][idx] not in result_tem_dict:
          result_tem_dict[result['template'][idx]] = [result['log'][idx]]
        else: result_tem_dict[result['template'][idx]].append(result['log'][idx])

    correct_num = 0
    for key in result_tem_dict.keys():
        if key not in oracle_tem_dict: continue
        else:
          if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1
    
    return correct_num/len(result_tem_dict)

# correctly identified templates over total num of oracle template
def evaluateRTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx] not in oracle_tem_dict:
          oracle_tem_dict[groundtruth['EventTemplate'][idx]] = [groundtruth['Content'][idx]]
        else: oracle_tem_dict[groundtruth['EventTemplate'][idx]].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if result['template'][idx] not in result_tem_dict:
          result_tem_dict[result['template'][idx]] = [result['log'][idx]]
        else: result_tem_dict[result['template'][idx]].append(result['log'][idx])

    correct_num = 0
    for key in oracle_tem_dict.keys():
        if key not in result_tem_dict: continue
        else:
          if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1
    
    return correct_num/len(oracle_tem_dict)

# calculate grouping accuracy
def evaluateGA(groundtruth, result):
    # load logs and templates
    compared_list = result['log'].tolist()

    # select groundtruth logs that have been parsed
    parsed_idx = []
    for idx, row in groundtruth.iterrows():
        if row['Content'] in compared_list:
            parsed_idx.append(idx)
            compared_list.remove(row['Content'])

    if not (len(parsed_idx) == 2000):
        print(len(parsed_idx))
        print("Wrong number of groundtruth logs!")
        return 0

    groundtruth = groundtruth.loc[parsed_idx]

    # grouping
    groundtruth_dict = {}
    for idx, row in groundtruth.iterrows():
        if row['EventTemplate'] not in groundtruth_dict:
            # create a new key
            groundtruth_dict[row['EventTemplate']] = [row['Content']]
        else: 
            # add the log in an existing group
            groundtruth_dict[row['EventTemplate']].append(row['Content'])

    result_dict = {}
    for idx, row in result.iterrows():
        if row['template'] not in result_dict:
            # create a new key
            result_dict[row['template']] = [row['log']]
        else: 
            # add the log in an existing group
            result_dict[row['template']].append(row['log'])

    # sorting for comparison
    for key in groundtruth_dict.keys():
        groundtruth_dict[key].sort()

    for key in result_dict.keys():
        result_dict[key].sort()

    # calculate grouping accuracy
    count = 0
    for parsed_group_list in result_dict.values():
        for gt_group_list in groundtruth_dict.values():
            if parsed_group_list == gt_group_list:
                count += len(parsed_group_list)
                break

    return count / 2000

async def process_stream(stream):
    stream_message = ''
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
            stream_message += chunk.choices[0].delta.content
        if chunk.choices[0].finish_reason is not None:
            break
    # logging.info(f'====RESPONSE: {stream_message}')
    # logging.info(f'total tokens={total_tokens}')
    return first_token_time, total_tokens,stream_message

async def make_request(client, request_arrive_time,output_tokens, item,request_timeout,log_template):
    idx=item[1]
    prompt_icl=item[0]
    cur_icl_content=prompt_icl.split('\n')
    cur_icl=cur_icl_content[:10]
    global cache

    qa=[]
    query=cur_icl_content[-1]
    if rerank:
        t1=time.time()
        reorder_icl=ICL_reorder(cur_icl)
        t2=time.time()
        # with open("/home/wyl/streamlog/overhead", "a") as file:
        #     file.write(f"{t2-t1}\n")

    else:
        reorder_icl=cur_icl

    for i in range(5):
        qa.append({'query':reorder_icl[2*i],'answer':reorder_icl[2*i+1]})
    
    instruction="I want you to act like an expert in log parsing. I will give you a log message wrapped by backticks. Your task is to identify all the dynamic variables in logs, replace them with {placeholder}, and output a static log template. Please only print the input log's template wrapped by backticks. "
    icl_message="Here are some log messages and their templates to help you understand how to parse logs: \n"
    for demo in qa:
        icl_message+= f"{demo['query']}\n{demo['answer']}\n"
    messages=[{"role": "system", "content": instruction},
                {"role": "user", "content": icl_message},
                {"role": "assistant", "content": "Sure, I can help you with log parsing based on the examples you provided. "},
        ]    
    messages.append({"role": "user", "content": 'Log message: '+query})

    start_time = time.time()    
    try:
        stream = await client.chat.completions.create(
            model="qwen/Qwen2.5-14B-Instruct",
            messages=messages,
            max_tokens=output_tokens,
            temperature=0,
            stream=True
        )
     
        
        first_token_time, total_tokens,stream_message = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        end_time = time.time()
        standard_template = post_process(stream_message) 
        log_template[int(idx)]=standard_template 
        e2e_latency=end_time-request_arrive_time
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft,e2e_latency
    

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None
        

async def worker(client, request_arrive_time,semaphore, queue, results, output_tokens, request_timeout,log_template):
    while True:
        async with semaphore:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            logging.info(f"====STARTING PROCESS REQ {item[1]}")
            result = await make_request(client,request_arrive_time, output_tokens,item, request_timeout,log_template)
            if result:
                results.append(result)
            # else:
                # logging.warning(f"Request {item} failed")
            queue.task_done() 
            # logging.info(f"====REQ DONE {item}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

async def run_benchmark(concurrency, request_timeout, output_tokens, vllm_url, api_key):
    client = AsyncOpenAI(base_url=vllm_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []
   
    file_list = [f for f in os.listdir(workload_folder)] 
    sorted_file_list = sorted(file_list, key=extract_number)
    for icl_file in sorted_file_list[:requests]:
        idx=icl_file.split('_')[-1].split('.')[0]
        file_path = os.path.join(workload_folder, icl_file)
        with open(file_path, "r") as file:
            prompt = file.read()
            await queue.put([prompt,idx])
    
    log_template = [None for _ in range(len(file_list))]

    request_arrive_time=time.time()

    for _ in range(concurrency):
        await queue.put(None)

    workers = [asyncio.create_task(worker(client, request_arrive_time,semaphore, queue, results, output_tokens, request_timeout,log_template)) 
               for _ in range(concurrency)]
    
    start_time = time.time()
    await asyncio.gather(*workers)
    end_time = time.time()

    # calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _,_ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _,_ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _,_ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft,_ in results if ttft is not None]
    e2e_latency_list = [e2e for _, _, _,_,e2e in results if e2e is not None]
    tpot_list = [x / y for x, y in zip(latencies, [tokens for tokens, _, _, _,_ in results if tokens is not None])]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0
    avg_e2e = sum(e2e_latency_list) / len(e2e_latency_list) if e2e_latency_list else 0

    # calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    tpot_percentiles = [calculate_percentile(tpot_list, p) for p in percentiles]
    e2e_percentiles = [calculate_percentile(e2e_latency_list, p) for p in percentiles]

    # write template
    with open(temp_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['template'])
        for element in log_template:
            writer.writerow([element])


    filename=f'run/{dataset}.csv' 
    filename_acc = f'run/accuracy_{dataset}.csv'
    new_row = {
        "dataset": dataset,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "latency_average": avg_latency,
        "latency_p95": latency_percentiles[1],
        "tokens_per_second_average": avg_tokens_per_second,
        "tokens_per_second_p95": tps_percentiles[1],
        "time_to_first_token_average": avg_ttft,
        "time_to_first_token_p95": ttft_percentiles[1],
        "time_per_output_token_average": avg_tpot,     
        "time_per_output_token_p95": tpot_percentiles[1],
        "total_time": total_elapsed_time,
        "throughput(rps)": requests/total_elapsed_time
    }
    dt_object = datetime.fromtimestamp(end_time)
    date_time_format = "%Y-%m-%d %H:%M"
    formatted_date_time = dt_object.strftime(date_time_format)
    with open(filename,'a') as f:
        metrics=f'{formatted_date_time},{dataset},{requests},{concurrency},{avg_latency},{latency_percentiles[1]},{avg_tokens_per_second},{tps_percentiles[1]},{avg_ttft},{ttft_percentiles[1]},{avg_tpot},{tpot_percentiles[1]},{total_elapsed_time},{requests/total_elapsed_time}\n'
        f.write(metrics)

    if evaluate: 
    # 保存日志和模板
        with open(log_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            log_data = list(reader)[1:]
        with open(temp_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            log_template = list(reader)[1:]
        with open(result_path, 'w', newline='') as csvfile:    
            writer = csv.writer(csvfile)
            writer.writerow(['log', 'template'])
            for log, template in zip(log_data, log_template):
                writer.writerow([log[0],template[0]])

        if not os.path.exists(filename_acc):
            df = pd.DataFrame(columns=['Dataset', 'Parsing Accuracy', 'Precision Template Accuracy', 'Recall Template Accuracy', 'Grouping Accuracy'])
        else:
            df=pd.read_csv(filename_acc)
     
        df_groundtruth = pd.read_csv(groundtruth_path)
        df_parsedlog = pd.read_csv(result_path)
        PA = evaluatePA(df_groundtruth, df_parsedlog)
        PTA = evaluatePTA(df_groundtruth, df_parsedlog)
        RTA = evaluateRTA(df_groundtruth, df_parsedlog)
        GA = evaluateGA(df_groundtruth, df_parsedlog)
        print("{}:\t PA:\t{:.6f}\tPTA:\t{:.6f}\tRTA:\t{:.6f}\tGA:\t{:.6f}".format(dataset, PA, PTA, RTA, GA))
        if dataset not in df['Dataset'].values:
            df.loc[len(df)] = [dataset, PA, PTA, RTA, GA]
        else:
            df.loc[df['Dataset'] == dataset, 'Parsing Accuracy'] = PA
            df.loc[df['Dataset'] == dataset, 'Precision Template Accuracy'] = PTA
            df.loc[df['Dataset'] == dataset, 'Recall Template Accuracy'] = RTA
            df.loc[df['Dataset'] == dataset, 'Grouping Accuracy'] = GA
       
        df.to_csv(filename_acc,mode='a', index=False, float_format="%.6f")
        
    return new_row
   

def print_results(results):
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Online Log Parsing with vLLM")
    parser.add_argument("--dataset", type=str, required=True, help="log dataset")
    parser.add_argument("--port", type=str, required=True, help="vllm port")
    parser.add_argument("--capacity", type=str, required=True, help="capacity of ICL Table")
    parser.add_argument("--concurrency", type=str, required=True, help="requests concurrency")
    parser.add_argument("--enable_pair", type=str, required=True, help="enable PAIR,if false, return to prefix caching")
    args = parser.parse_args() 
    dataset=args.dataset
    port=args.port
    capacity=args.capacity
    concurrency=args.concurrency
    naive_output=[]
    workload_folder=f'prompt/{dataset}'
    result_path=f'benchmark/{dataset}_divlog_evaluate.csv'
    groundtruth_path=f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv'
    temp_path=f'benchmark/{dataset}_divlog_template.csv'
    log_path=f'benchmark/{dataset}_divlog_test.csv'
    requests=2000
    rerank=True
    evaluate=True
    cache=OrderedDict()
    request_timeout=300
    output_tokens=100
    vllm_url=f"http://localhost:{port}/v1"
    api_key="EMPTY"
    results = asyncio.run(run_benchmark(concurrency, request_timeout, output_tokens, vllm_url, api_key))
    print_results(results)



 