import pickle
import argparse
import torch
import datetime
import scipy

import time
from typing import List , Tuple, Dict
from sentence_transformers import SentenceTransformer
from queue import Empty, Full
from torch.multiprocessing import Lock, Process, current_process, Manager, set_start_method
from util.my_edit_distance import CustomLevin
from util.hangle_util import *
from os.path import dirname, join, abspath
from typing import List, Dict, Callable
from tqdm import tqdm


def load_embedding_data(embedding_path : str):
    # (<vector>', <백터>)
    with open(embedding_path , 'rb') as f:
        data = pickle.load(f)
    return data

def split_jaso_to_string(levin, target : str):
    return ''.join([ ''.join(_) for _ in levin.decompose_string(target)])

def post_process_levinstein(levin, query, candidates):
    aux = dict()
    for cand in candidates:
        aux[split_jaso_to_string(levin,cand[0])] = cand[0]
    query_decompose = split_jaso_to_string(levin, query)
    edit_distance_pool = list(aux.keys())        
    edit_distance_outputs = [\
        [_[0], levin.get_string_distance(_[0], query)] \
            for _ in candidates]
    edit_distance_outputs.sort(key = lambda x: x[1])
    return edit_distance_outputs[0][0]


def predict(model, levin, target : str, sentences, sentence_embeddings, topk : int = 30, debug : bool = False) -> str:

    queries = [target]
    query_embeddings = model.encode(queries)
    ret = []

    for query, query_embedding in zip(queries, query_embeddings):
        
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        candidates = []
        
        for idx, distance in results[0:topk]:
            target = sentences[idx]
            cosine_score = (1-distance)
            candidates.append([target, cosine_score])
        candidates.sort(key = lambda x : -x[1])

        first_sbert_ret = candidates[0][0]
        
        
        # pp_ret = post_process_levinstein(levin, query, candidates)
        # ret.append(pp_ret)
        ret.append(first_sbert_ret) 
 

        if debug:

            print('#' ,query)
            print('\t # sbert ret #')
            for idx, distance in results[0:topk]:
                target = sentences[idx]
                cosine_score = (1-distance)
                print([target, cosine_score])
          
            print('\t # edit ret #')
            for edit_output in edit_distance_outputs:
                print(edit_output)
            print('output: {} '.format(edit_distance_outputs[0][0]))        
    
    # print(ret[0], ret[1], ret)
    return ret[0]



def multi_inference(embedding_type_name, tasks_to_accomplish, tasks_finished, gpu_idx, src_sentences, src_sentence_embeddings) -> bool:
    '''
    param 
        - embedding_type_name : str
            - setnece transformer 에서 제공하는 embedding 종류 
        - tasks_to_accomplish : Queue
            - sentence(str) 이 담긴 queue
        - tasks_finished : Queue
            - inference 결과 로그를 담는 queue
        - src_sentences : List[str]
            - source pool sentence List
        - src_sentence_embeddgins
            - source pool settence vector 들의 List
    desc
        - 해당하는 emedding_type_name로 sentence encoder를 로딩
        - tasks_to_accomplish 에서 job(str)을 가져옴
        - src_sentence_embeddings 를 보고 가장 유사한 embeddig을 찾고 매핑된 src_sentence를 구한다

    '''
    GPU_NUM = gpu_idx # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    model = SentenceTransformer(embedding_type_name)
    
    
    levin = CustomLevin()
    while True:
        try:
            if tasks_to_accomplish.empty():
                raise Empty
            task = tasks_to_accomplish.get_nowait()

        except Empty:
            break
        else:
            ret = []
            # ret : [(sentence_embeddings, concat_synom) ... ]

            for i in tqdm(task):
            # i -> 한 프로세서에 배분된 target sentence의 리스트
                target = i
                sbert_inference = predict(model, levin, target, src_sentences, src_sentence_embeddings)               
                ret.append([i, sbert_inference])
            for data in ret:
                # target = data[0]
                # inference = data[1]
                # sbert_only_inference = data[2]
                tasks_finished.put(data)
            
            print(' {} : END pushing {} data to finished_queue'.format(current_process().name, len(ret)))
            time.sleep(0.5)

            
    return True


def main(args):

    '''
    embedding_path에 있는 data 
        - [sentence#1, vecetor#1]
        - [sentence#2, vecetor#2]
        ...
    '''
    print('> START  ')
    print('> parameter  ')
    for k, v in args._get_kwargs():
        print('> {} : {}'.format(k, v))
    print('')
    print('> Action ')
    # 0. sentence_embeddings 준비
    embedding_type_name = args.embedding_type_name
    topk = args.topk
    target_data_path = args.target_data_path
    ground_data_path = args.ground_data_path
    source_embedding_path = args.source_embedding_path
    number_of_processes = args.gpu_num
    


    # 1. source pool loading
    # [(vector#1, sentence#1), (vector#2, sentence#2) ... ]
    source_pool = load_embedding_data(source_embedding_path)    
    src_embeddings = [_[0] for _ in source_pool]     
    src_sentences = [_[1] for _ in source_pool] 
    # 2. target data split
    target_data_list = [_.strip() for _ in open(target_data_path , mode='r', encoding='utf-8')]
    number_of_processes = number_of_processes if number_of_processes < len(target_data_list) else len(target_data_list)    
    num_of_tasks = len(target_data_list)//number_of_processes    
    tasks = [target_data_list[_ * num_of_tasks : (_ + 1) * num_of_tasks] for _ in range(number_of_processes)]
    
    # 3. queue 준비
    tasks_to_accomplish = Manager().Queue()
    tasks_finished = Manager().Queue()
    for task in tasks:
        tasks_to_accomplish.put(task)
    
    processes = []
    # process 생성
    # target에 대해서 encoding 하고 가장 벡터가 유사한 것을 찾는다.
    for i in range(number_of_processes):
        p = Process(target = multi_inference \
                    , args = (embedding_type_name, tasks_to_accomplish, tasks_finished, i+1, src_sentences, src_embeddings,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


    # 결과 파일로 저장 
    store_target = []
    while not tasks_finished.empty():
        store_target.append(tasks_finished.get_nowait())
    
    if ground_data_path:
        gt_data = [_.strip() for _ in open(ground_data_path, mode ='r', encoding= 'utf-8')]
        for idx, val in enumerate(zip(store_target, gt_data)):
            store_target[idx].append(val[-1])
    


    time_tag= datetime.datetime.now().strftime('%Y%m%d%H%S')


    head_line = ['<target>', '<inference>', '<ground_truth>']
    with open(args.output_data_path, mode= 'w' , encoding='utf-8') as wdesc:
        
        wdesc.writelines('\t'.join(head_line))
        wdesc.writelines('\n')

        for i in store_target:
            # target = data[0]
            # inference = data[1]
            # sbert_only_inference = data[2]
            line = '\t'.join(i)
            wdesc.writelines(line)
            wdesc.writelines('\n')
            
    print('> FINISH - result file : {}'.format(args.output_data_path))
    return True



if __name__ == "__main__":

    '''
    target_data_path의 data 들이 source_data_pool과 그 embedding에서 어디에 가장 가까운지
    를 output_data_path에 놔둔다
    '''
    MAIN_DIR = abspath('')
    MAIN_DATA_DIR = join(MAIN_DIR , 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_data_path",
                        type=str,
                        default= join(MAIN_DATA_DIR, 'target_data_path.tsv'),
                        help="target data file path")


    parser.add_argument("--ground_data_path",
                        type=str,
                        default= join(MAIN_DATA_DIR, 'ground_truth_path.tsv'),
                        help="ground file path")

    parser.add_argument("--output_data_path",
                        type=str,
                        default= join(join(MAIN_DIR, 'result'), 'output.tsv'),
                        help="test output file path")

    parser.add_argument("--gpu_num",
                        default=1,
                        type=bool,
                        required=False,
                        help="gpu_number")


    parser.add_argument("--embedding_type_name",
                        default='xlm-r-large-en-ko-nli-ststb',
                        type=str,
                        required=False,
                        help="embedding type name in huggingface")

    parser.add_argument("--source_embedding_path", 
                        type=str,
                        help="pickled data")

    parser.add_argument("--topk",
                        default=10,
                        type=int,
                        required=False,
                        help="topk")
    args = parser.parse_args()

    LIMIT_PROC_UNIT = torch.cuda.device_count() 
    print ('> Available devices ', LIMIT_PROC_UNIT)
    args.gpu_num = args.gpu_num if LIMIT_PROC_UNIT >= args.gpu_num else LIMIT_PROC_UNIT
    print ('> current used devices ', args.gpu_num)
    
    # distiluse-base-multilingual-cased
    set_start_method('spawn')

    strt_time = time.time()   
    main(args)
    end_time = time.time()
    print('> END, COST_TIME : {}'.format(end_time-strt_time))    
