from random import sample
import json
from tqdm import tqdm
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch,copy,time
from sklearn.cluster import KMeans  
os.environ["PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS"] = "2000"


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
    return selected_items

def getDppIndex(log_emb_list, 
                item_size,    # log dataset size
                split_ratio):

    max_length = int(item_size * split_ratio)
    feature_vectors = np.array(log_emb_list) 

    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T) 
    t = time.time()
    result = dpp(similarities, max_length)
    result.sort()
    print('DPP algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    return result

def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def DPPsplit(log_list, groundtruth_template, candidate_idx):
    cand_logs = [log_list[idx] for idx in candidate_idx]
    cand_templates = [groundtruth_template[idx] for idx in candidate_idx]
    test_idx = []
    for i in range(len(log_list)):
      if i not in candidate_idx: 
         test_idx.append(i)
    test_idx.sort()
    test_logs = [log_list[idx] for idx in test_idx]
    test_templates = [groundtruth_template[idx] for idx in test_idx]
    return test_logs, cand_logs, test_templates, cand_templates

class ModelParser():
    def __init__(self,dataset):
        self.dataset=dataset
        self.map_path=f"lookupmap/{self.dataset}_lookupmap.json"
        self.cand_ratio = 0.1
        self.split_method ='DPP'
        self.emb_path=f"embedding_log2k/{self.dataset}.json"
        self.log_path = f"dataset/{self.dataset}/{self.dataset}_2k.log_structured_corrected.csv"
        self.order_method="KNN"
        self.permutation='ascend'
        self.log_test, self.log_cand, self.gt_test, self.gt_cand = self.splitCandidates(self.log_path, self.cand_ratio, self.split_method)
        self.lookUpMap = self.buildLookupMap(self.map_path)


    def extractCsvContent(self, groundtruth_path):
        dataframe = pd.read_csv(groundtruth_path)
        content_list = dataframe['Content'].values.tolist()
        return content_list

  # extract groundtruth templates from log_structured.csv file
    def extractCsvTemplate(self, groundtruth_path):
        dataframe = pd.read_csv(groundtruth_path)
        template_list = dataframe['EventTemplate'].values.tolist()
        return template_list
    
    def splitCandidates(self, groundtruth_path, cand_ratio, method="random"):
        log_list = self.extractCsvContent(groundtruth_path)
        groundtruth_template = self.extractCsvTemplate(groundtruth_path)
        if method == "random":
            self.map_path += '_random.json'
          # split randomly
            log_test, log_cand, gt_test, gt_cand = train_test_split(log_list, groundtruth_template, test_size=cand_ratio, random_state=42)
        elif method == "DPP":
          # split with diversity
            file = open(self.emb_path, "r")
            emb_map = json.load(file)
            file.close()
            log_embs = []
            for log in log_list:
                log_embs.append(emb_map[log])
            print(f"length of log embs is {len(log_embs)}")
            candidate_idx = getDppIndex(log_embs, 2000, cand_ratio)
            log_test, log_cand, gt_test, gt_cand = DPPsplit(log_list, groundtruth_template, candidate_idx)
            log_test = log_test + log_cand
            gt_test = gt_test + gt_cand
        
        return log_test, log_cand, gt_test, gt_cand
    
    def generateLuMap(self, look_up_map_path):
        # get embeddings from embedding json file
        print('Generating lookup map for {} ...'.format(self.dataset))
        with open(self.emb_path, "r") as file:
            emb_map = json.load(file)
        test_embs = [emb_map[log] for log in self.log_test]
        cand_embs = [emb_map[log] for log in self.log_cand]
        lookUpMap = {}
        for test_idx in tqdm(range(len(self.log_test))):
            dis_dict = {}
            for cand_idx in range(len(self.log_cand)):
                dis_dict[cosine_similarity(test_embs[test_idx], cand_embs[cand_idx])] = cand_idx
            sorted_list = []
            for key in sorted(dis_dict, reverse=True): 
                sorted_list.append(dis_dict[key])
            lookUpMap[self.log_test[test_idx]] = sorted_list
        with open(look_up_map_path, 'w') as file:
            file.write(json.dumps(lookUpMap))
        return lookUpMap

    def getNearest(self, log, N=5):
        cand_list = self.lookUpMap[log]
        if self.order_method == 'random':
            return sample(cand_list, N)
        # return the idexes of most similar N log candidates
        elif self.order_method == 'KNN':
            shift = 0
            result = cand_list[0:N]
            while log in result:
                shift += 1
                result = cand_list[shift:N+shift]
            if self.permutation == 'ascend':
                return result
            elif self.permutation == 'descend':
                result.reverse()
                return result
            elif self.permutation == 'random':
                result = sample(result, N)
                return result

    def buildLookupMap(self, map_path):
        if (os.path.exists(map_path)): 
            print("Loading look up map of {} ...".format(self.dataset))
            with open(map_path, "r") as file:
                return json.load(file)
        else: 
            return self.generateLuMap(map_path)

    
    def generatePrompt2txt(self, line_idx,log, nearest_num=5):
        idxes = self.getNearest(log, nearest_num)
        prompt=""
        # backward iteration
        for i in range(len(idxes)-1,-1,-1):
            prompt = prompt + "Log message: `" + self.log_cand[idxes[i]].strip() + \
                '`\nLog template: `' + self.gt_cand[idxes[i]].strip() + '`\n'  
            similarist_gt = self.gt_cand[idxes[0]]
        prompt=prompt+'`'+log+'`'
        with open(f'prompt/{self.dataset}/icl_{line_idx}.txt', 'w', encoding='utf-8') as file:
            file.write(prompt) 


if __name__ == "__main__":        
    datasets = ["HDFS", "Spark", "BGL", "Windows", "Linux", "Android", "Mac", "Hadoop", "HealthApp", "OpenSSH", "Thunderbird", "Proxifier", "Apache", "HPC", "Zookeeper", "OpenStack"]
    for dataset in datasets:
        mp=ModelParser(dataset)
        print('generating ICL for dataset:',dataset)
        path = f'prompt/{dataset}'
        try:
            os.makedirs(path, exist_ok=True) 
            print(f"path {path} create successful")
        except OSError as error:
            print(f"path {path} create fail: {error}")
        for line_idx in tqdm(range(len(mp.log_test))): 
            log = mp.log_test[line_idx]
            mp.generatePrompt2txt(line_idx,log, nearest_num=5)
