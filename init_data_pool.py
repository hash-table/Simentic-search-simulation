

import datetime
import time
import pickle
import torch
import argparse
from os.path import dirname, join, abspath
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from util.hangle_util import *
from typing import List, Tuple
from queue import Empty
try:
    from torch.multiprocessing import Lock, Process, current_process, Manager, set_start_method
except:
    from multiprocessing import Lock, Process, current_process, Manager







def make_embedding(tasks_to_accomplish, tasks_finished, gpu_idx, embedding_type_name ) -> List[List[Tuple]]:
    GPU_NUM = gpu_idx # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    model = SentenceTransformer(embedding_type_name )
    
    
    while True:
        try:
            if tasks_to_accomplish.empty():
                raise Empty

            task = tasks_to_accomplish.get_nowait()
        except Empty:
            break
        else:
            ret = []

            for i in tqdm(task):
                sentence_embeddings = tuple(model.encode([i])[0].tolist())
                ret.append((sentence_embeddings, i))
            print(' {} : embedding making finished'.format(current_process().name))
            for data in ret:
                print("{} put : {}".format(current_process().name, data[1]))
                tasks_finished.put(data)
            
            print(' {} : END pushing embed-sent {} data to finished_queue'.format(current_process().name, len(ret)))
            time.sleep(0.5)

            
    return True


def main(args):
    args.embedding_output_path = join(args.EMBED_DATA_DIR, args.output_path) 
    print('> START  ')
    print('> parameter  ')
    for k, v in args._get_kwargs():
        print('> {} : {}'.format(k, v))
    print('')
    print('> Action ')
    pool_path = args.data_path
    GPU_NUM = args.gpu_num
    embedding_type_name = args.embedding_type_name
    output_embedding_data = args.embedding_output_path 

    sym_line_list = [ _.strip() for _ in open(pool_path, mode = 'r' , encoding= 'utf-8')]
    
    number_of_processes = GPU_NUM if GPU_NUM < len(sym_line_list) else len(sym_line_list)
    num_of_tasks = len(sym_line_list)//number_of_processes    
    
    tasks = [sym_line_list[_ * num_of_tasks : (_ + 1) * num_of_tasks] for _ in range(number_of_processes)]
    
    
    tasks_to_accomplish = Manager().Queue()
    for task in tasks:
        tasks_to_accomplish.put(task)
    tasks_finished = Manager().Queue()
    processes = []
    # creating processes
    for i in range(number_of_processes):
        p = Process(target = make_embedding, args = (tasks_to_accomplish, tasks_finished, i,embedding_type_name , ))
        processes.append(p)
        p.start()
    store_target = []
    
    for p in processes:
        p.join()

    while not tasks_finished.empty():
        store_target.append(tasks_finished.get_nowait())
    
    with open(output_embedding_data, 'wb') as f:
        pickle.dump(store_target, f, pickle.HIGHEST_PROTOCOL)

    
    
    
    return True






if __name__ == "__main__":
    MAIN_DIR = abspath('')
    MAIN_DATA_DIR = join(MAIN_DIR , 'data')
    EMBED_DATA_DIR = join(MAIN_DIR , 'embedding')
    time_tag= datetime.datetime.now().strftime('%Y%m%d%H%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default= join(MAIN_DATA_DIR, 'source_pool.tsv'),
                        required=False,

                        help="target source file path")

    parser.add_argument("--embedding_type_name",
                        default='xlm-r-large-en-ko-nli-ststb',
                        type=str,
                        required=False,
                        help="embedding type name in sentence-transformer")

    parser.add_argument("--gpu_num",
                        default=1,
                        type=bool,
                        required=False,
                        help="gpu_number")

    parser.add_argument("--output_path",
                        type=str,
                        default= join(EMBED_DATA_DIR, '{}'.format(time_tag)),
                        required=False,
                        help="output file path")


    args = parser.parse_args()

    args.MAIN_DIR = abspath('')
    args.MAIN_DATA_DIR = join(MAIN_DIR , 'data')
    args.EMBED_DATA_DIR = join(MAIN_DIR , 'embedding')
    
    set_start_method('spawn')
    # 현재 가용 GPU 보기 
    LIMIT_PROC_UNIT = torch.cuda.device_count() 
    print ('Available devices ', LIMIT_PROC_UNIT)
    args.gpu_num = args.gpu_num if LIMIT_PROC_UNIT >= args.gpu_num else LIMIT_PROC_UNIT
    strt_time = time.time()   
    main(args)
    end_time = time.time()
    print('> END, COST_TIME : {}'.format(end_time-strt_time))    
