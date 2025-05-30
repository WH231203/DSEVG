import torch

def pre_trained_model_to_finetune(checkpoint, args):
    checkpoint = checkpoint['model']
    # only delete the class_embed since the finetuned dataset has different num_classes
    num_layers = args.dec_layers + 1 if args.two_stage else args.dec_layers
    for l in range(num_layers):
        del checkpoint["class_embed.{}.weight".format(l)]
        del checkpoint["class_embed.{}.bias".format(l)]
    
    return checkpoint




#####################0709加：用于初始化模型，而不删除分类层的权重
def load_init_weight(model, args):
    checkpoint = torch.load(args.init_weight)
    # pretrained_dict = checkpoint['state_dict']
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert (len([k for k, v in pretrained_dict.items()]) != 0)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    del checkpoint  # dereference seems crucial
    torch.cuda.empty_cache()
    return model
#####################0709加：用于初始化模型，而不删除分类层的权重