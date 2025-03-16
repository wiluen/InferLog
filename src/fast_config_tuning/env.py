import yaml
import subprocess
import random
import time
import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)   

def command_str(i,j,k,model,dataset,random_port):
    with open('config.yml', 'r') as file:
        config= yaml.safe_load(file)
    command = [
        "vllm", "serve", model,"--enable-prefix-caching","--port",str(random_port)
    ]
    config_values = {}
    max_token_name='max-num-batched-tokens'
    max_seq_name='max-num-seqs'
    delay_name='scheduler-delay-factor'

    max_token_value=config[max_token_name]['values'][i]
    max_seq_value=config[max_seq_name]['values'][j]
    delay_value=config[delay_name]['values'][k]

    config_values[max_token_name]=max_token_value
    config_values[max_seq_name]=max_seq_value
    config_values[delay_name]=delay_value

    command.append(f"--max-model-len") 
    command.append(str(max_token_value))
    command.append(f"--{max_token_name}") 
    command.append(str(max_token_value))
    command.append(f"--{max_seq_name}") 
    command.append(str(max_seq_value))
    command.append(f"--{delay_name}") 
    command.append(str(delay_value))

    filename=f'run/test_{dataset}.csv'

    now=time.time()
    dt_object = datetime.fromtimestamp(now)
    date_time_format = "%Y-%m-%d %H:%M:%S"
    formatted_date_time = dt_object.strftime(date_time_format)
    with open(filename,'a') as f:
        conf=f'{i},{j},{k},{formatted_date_time},{max_token_value},{max_seq_value},{delay_value},'
        f.write(conf)
    return command

def read_performance(dataset):
    csv_file_path=f'run/test_{dataset}.csv'
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()
        
        if lines:  
            last_line = lines[-1].strip()  
            data = last_line.split(',')  
            
            if len(data) > 1:  
                second_last_data = data[-2]  
                return second_last_data
            else:
                return "no data found"
        else:
            return "file is empty"

def get_command(i,j,k,dataset,random_port):
    model="qwen/Qwen2.5-14B-Instruct"
    command = command_str(i,j,k,model,dataset,random_port)
    return ' '.join(command)

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


def run(i,j,k,dataset):
    env = os.environ.copy()
    card=random.randint(1,5)
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    print("CUDA_VISIBLE_DEVICES in card:",card)
    random_port = random.randint(8030, 8040)
    command = get_command(i,j,k,dataset,random_port)
    split_strings = command.split(' ')
    yy = [f"{s}" for s in split_strings]
    print(yy)
    process_vllm = subprocess.Popen(
        yy,
        env=env
    )
    start_time = time.time()
    while process_vllm.poll() is None and (time.time() - start_time) < 50:
        time.sleep(5)
        print(f'Checking if vllm is ready...,left time:{50-time.time()+start_time} seconds')

    if process_vllm.poll() is not None:
        print(f"vllm failed to start in iteration {i + 1}")

    print('vllm ready')

    process_benchmark = subprocess.Popen(
        ["python", "inferlog.py","--dataset",dataset,"--port",str(random_port)]
    )

    process_benchmark.wait()
    time.sleep(10)
    print(f"Terminating vllm in iteration {i + 1}")
    process_vllm.terminate() 

    try:
        process_vllm.wait(timeout=10)  
    except subprocess.TimeoutExpired:
        print("vllm did not exit gracefully, killing the process")
        process_vllm.kill()  

    print(f"Finished iteration {dataset}\n, wait for 10s ")
    time.sleep(30)

def get_similar(task):
    df = pd.read_csv('log_feature.csv')
    data_dict = {row['dataset']: np.array([row['feature1'],row['feature2'],row['feature3']]) for index, row in df.iterrows()}
    feature = None
    if task in data_dict:
        feature = data_dict[task]
    else:
        print(f"No data found for task: {task}")
    spark = data_dict.get('Spark', None)
    mac = data_dict.get('Mac', None)
    hdfs = data_dict.get('HDFS', None)
    bgl = data_dict.get('BGL', None)
    hadoop = data_dict.get('Hadoop', None)
    zookeeper = data_dict.get('Zookeeper', None)
    openstack = data_dict.get('OpenStack', None)
    proxifier = data_dict.get('Proxifier', None)
    apache = data_dict.get('Apache', None)
    healthapp = data_dict.get('HealthApp', None)
    linux = data_dict.get('Linux', None)
    openssh = data_dict.get('OpenSSH', None)
    windows = data_dict.get('Windows', None)
    thunderbird = data_dict.get('Thunderbird', None)
    android = data_dict.get('Android', None)
    hpc = data_dict.get('HPC', None)
    
    if task=='Spark':feature=spark
    if task=='Mac':feature=mac
    if task=='HDFS':feature=hdfs
    if task=='BGL':feature=bgl
    if task=='Hadoop':feature=hadoop
    if task=='Zookeeper':feature=zookeeper
    if task=='OpenStack':feature=openstack
    if task=='Proxifier':feature=proxifier
    if task=='Apache':feature=apache
    if task=='HealthApp':feature=healthapp
    if task=='Linux':feature=linux
    if task=='OpenSSH':feature=openssh
    if task=='Windows':feature=windows
    if task=='Thunderbird':feature=thunderbird
    if task=='Android':feature=android
    if task=='HPC':feature=hpc

    meta_train=['Spark', 'Mac', 'HDFS','BGL','Hadoop','OpenStack','Zookeeper','Proxifier']  #meta-train #2
    tasks = np.array([feature,spark, mac, hdfs,bgl,hadoop,openstack,zookeeper,proxifier]) # meta-train #2
    cos_sim_matrix = cosine_similarity(tasks)

    cos_similar=cos_sim_matrix[0][1:]
    print(cos_similar)
    combined = list(zip(cos_similar, meta_train))
    combined.sort(key=lambda pair: pair[0], reverse=True)
    top_three = combined[:3]
    top_three_names = [name for _, name in top_three]
    print('top-3 similar task:',top_three_names)
    return top_three_names

def get_initial_point(task):
    top_task=get_similar(task)
    top1_task=top_task[0]
    top2_task=top_task[1]
    top3_task=top_task[2]
    best_meta_data=pd.read_csv('best_record.csv')
    task1_best_res=np.array(best_meta_data.loc[best_meta_data['dataset'] == str(top1_task), ['conf1','conf2','conf3']].values)
    task2_best_res=np.array(best_meta_data.loc[best_meta_data['dataset'] == str(top2_task), ['conf1','conf2','conf3']].values)
    task3_best_res=np.array(best_meta_data.loc[best_meta_data['dataset'] == str(top3_task), ['conf1','conf2','conf3']].values)
    concatenated_array = np.concatenate([task1_best_res, task2_best_res,task3_best_res], axis=0)
    print('initial point:',concatenated_array)
    return concatenated_array





