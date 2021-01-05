class Config(object):
    backbone = 'tf_efficientnet_b5_ns'
    num_classes = 43 #
    # loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    #
    input_size = 456
    train_batch_size = 11  # batch size
    val_batch_size = 6
    test_batch_size = 1
    optimizer = 'sgd'
    lr = 0.00001  # adam 0.00001
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0]
    num_workers = 4  # how many workers for loading data
    max_epoch = 50
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 50
    save_interval = 1
    min_save_epoch=1
    #
    log_dir = 'log/'
    train_val_data = './garbage_classify/70'
    raw_json = './garbage_classify/garbage_classify_rule.json'
    train_list='./dataset/label.txt'
    val_list='./dataset/val.txt'
    #
    checkpoints_dir = 'ckpt/'
    cut_prob=0.6
