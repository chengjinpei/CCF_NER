class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 10 + 1  # 实体的种类

        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 20 * 3

        self.train_epoch = 10
        self.sequence_length = 512  # BERT的输入MAX_LEN

        self.learning_rate = 1e-4  # 下接结构的学习率
        self.embed_learning_rate = 5e-5  # BERT的微调学习率
        self.batch_size = 2

        # BERT预训练模型的存放地址
        self.bert_file = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/pretrained_model/BERT/bert_model.ckpt'
        self.bert_config_file = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/pretrained_model/BERT/bert_config.json'
        self.vocab_file = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/pretrained_model/BERT/vocab.txt'

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file  = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/ensemble/source_file/'
        self.ensemble_result_file = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/ensemble/result_file/'

        # 存放的模型名称，用以预测
        #self.checkpoint_path = "F:/5.Github项目代码/11.Python深度学习/chapter8/data/model/runs_0/1620797096/model_0.5720_0.4779-8012"  #
        self.checkpoint_path = "F:/5.Github项目代码/11.Python深度学习/chapter8/data/model/runs_0/1620797096/model_0.5296_0.6694-4006"
        self.model_dir = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/model'  # 模型存放地址
        self.new_data_process_quarter_final = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/process_data/'  # 数据预处理的结果路径
        self.source_data_dir = 'F:/5.Github项目代码/11.Python深度学习/chapter8/data/clear_csv_data/'  # 原始数据集

        # self.model_type = 'idcnn'  # 使用idcnn
        self.model_type = 'bilstm'  # 使用bilstm
        self.lstm_dim = 256
        self.dropout = 0.5
        self.use_origin_bert = True  # True:使用原生bert, False:使用动态融合bert

