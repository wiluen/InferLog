import yaml
import subprocess
import random
import time
import os
from pyDOE import lhs
import numpy as np
from setconfig import get_command


n_token=14
n_seq=6
n_delay=4

def get_rand():
    rand_list=[]
    for i in range(15):
        a = random.randint(0, n_token)
        b = random.randint(0, n_seq)
        c = random.randint(0, n_delay)
        rand_list.append([a,b,c])
    return rand_list

def get_lhs():
    n_dimensions = 3
    n_samples = 100
    n_token_bounds = [0, n_token]
    n_seq_bounds = [0, n_seq]
    n_delay_bounds = [0, n_delay]
    samples = lhs(n_dimensions, samples=n_samples)
    mapped_samples = np.zeros(samples.shape)
    mapped_samples[:, 0] = samples[:, 0] * (n_token_bounds[1] - n_token_bounds[0]) + n_token_bounds[0]
    mapped_samples[:, 1] = samples[:, 1] * (n_seq_bounds[1] - n_seq_bounds[0]) + n_seq_bounds[0]
    mapped_samples[:, 2] = samples[:, 2] * (n_delay_bounds[1] - n_delay_bounds[0]) + n_delay_bounds[0]
    samples_list = mapped_samples.tolist()
    return samples_list

def is_port_in_use(port):
    result = subprocess.run(['lsof', '-i', ':' + str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode() != ''

def check_port(port, interval=3):
    while True:
        if not is_port_in_use(port):
            print(f"Port {port} is free, executing subsequent program.")
            break
        else:
            print(f"Port {port} is in use, checking again in {interval} seconds.")
        time.sleep(interval)


def run(i,j,k,iter,dataset):
    env = os.environ.copy()
    card=int(iter%5)+1
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    print("CUDA_VISIBLE_DEVICES in card:",card)
    random_port = random.randint(8030, 8050)
    command = get_command(i,j,k,iter,dataset,random_port)
    split_strings = command.split(' ')
    comm = [f"{s}" for s in split_strings] 
    process_vllm = subprocess.Popen(
        comm,
        env=env
    )
   
    start_time = time.time()
    while process_vllm.poll() is None and (time.time() - start_time) < 50:
        time.sleep(5)
        print(f'Checking if vllm is ready...,left time:{50-time.time()+start_time} seconds')

    if process_vllm.poll() is not None:
        print(f"vllm failed to start in iteration {i + 1}")

    print('vllm ready')
    capacity = 200
    concurrency = 200
    enable_pair = "True" 
    process_benchmark = subprocess.Popen(
        ["python", "inferlog.py","--dataset",dataset,"--port",str(random_port),"--capacity",str(capacity),"concurrency",str(concurrency),"enable_pair",enable_pair]
    )
    process_benchmark.wait()
    time.sleep(10)
    print(f"Terminating vllm in iteration {i + 1}")
    process_vllm.terminate()  # Send SIGTERM signal to vllm to gracefully terminate the process

    # If vllm does not respond, use kill() to force terminate
    try:
        process_vllm.wait(timeout=10)  # Wait 3 seconds for vllm to exit
    except subprocess.TimeoutExpired:
        print("vllm did not exit gracefully, killing the process")
        process_vllm.kill()  # Force terminate vllm

    print(f"Finished iteration {dataset} {iter}\n, wait for 10s ")
    time.sleep(40)

if __name__=='__main__':
    dataset='HPC'
    generate_conf=get_rand()
    iter=0
    for d in generate_conf:
        print('config:',d)
        i=int(d[0])
        j=int(d[1])
        k=int(d[2])    
        run(i,j,k,iter,dataset)
        iter+=1
    