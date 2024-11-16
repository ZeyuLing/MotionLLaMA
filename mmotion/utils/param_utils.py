def get_parameter_stats(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0

    return total_params, trainable_params, trainable_ratio