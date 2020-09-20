# Simentic search simulation with Sentence Embedding

- huggingface에서 sentencebert 계열의 embedding 들을 이용해 simentic search 수행
- input_data_path 를 source_pool_data_path 안에서 simentic search 
- source_pool_data에 대해 가장 가까운 sentence로 mapping (cosine simliarity)
- ground truth_path 에 대한 결과와 비교
- edit-distance, romanizer 등을 이용한 후처리 로직 적용 후 테스트

# Init 

1. clone the repo
2. build image in repo
        
        docker build -t sbert-search-img -f ./container/Dockerfile .

3. run container in repo
        
        docker run -itd --name sbert-search-app --network host -e --runtime=nvidia sbert-search-img:latest


# Run 

1. Make embedding vector based on source data file

        python init_data_pool.py [--data_path DATA_PATH]
                             [--embedding_type_name EMBEDDING_TYPE_NAME]
                             [--gpu_num GPU_NUM] [--output_path OUTPUT_PATH]
    
        - DATA_PATH : source 로 사용할 one-line corpus
        - EMBEDDING_TYPE_NAME : Sentence transformers 에서 지원하는 type name
        - GPU_NUM : source embedding에 사용할 gpu 수
        - OUTPUT_PATH : Embedding 결과를 pickle로 저장할 위치 

        EX) python init_data_pool.py --output_path source_pool_xlm-r-large-en-ko-nli-ststb

2. Simentic search  

        python main.py [--target_data_path TARGET_DATA_PATH]
                    [--ground_data_path GROUND_DATA_PATH]
                    [--output_data_path OUTPUT_DATA_PATH] [--gpu_num GPU_NUM]
                    [--embedding_type_name EMBEDDING_TYPE_NAME]
                    [--source_embedding_path SOURCE_EMBEDDING_PATH] [--topk TOPK]

        - TARGET_DATA_PATH : 검색할 대상, one-line-corpus
        - GROUND_DATA_PATH : 검색할 대상에 대한 정답 셋, one-line-corpus
        - OUTPUT_DATA_PATH : 검색 대상에 대한 추론값 및 Ground truth와의 비교 결과를 저장할 위치 
        - GPU_NUM : simentic search에 사용할 gpu 수
        - EMBEDDING_TYPE_NAME : 검색 대상을 encoding할 Embedding 방법 (* init_data_pool.py와 같은 EMBEDDING_TYPE_NAME 사용)
        - SOURCE_EMBEDDING_PATH : Source pool의 Embedding 정보들이 저장되어 있는 경로 (* init_data_pool.py 로 생성)
        - TOPK : simentic search에서 사용할 topk 

         EX) python main.py --source_embedding_path ./embedding/source_pool_xlm-r-large-en-ko-nli-ststb

3. Simentic search example (result.output.tsv)


    <target>	<inference>	<ground_truth>
    파란백	블루 보틀	파란 가방
    디옹	dior	dior
    hi	안녕 하세요	안녕 하세요
    bye	안녕히 가세요	안녕히 가세요
    나이키 에어맥스	나이키	나이키
    애호박	채소	채소
    이거 별로야 바꿔줘	환불	환불
    언제 도착해?	배송조회	배송조회
    빨간가방	레드백	레드백

