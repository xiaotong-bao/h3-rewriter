"""Copy a pretrained policy adapter into an exact, frozen KL reference."""
def freeze_reference(model):
    copied = 0
    for name, parameter in model.named_parameters():
        if '.default.' in name:
            reference = model.get_parameter(name.replace('.default.', '.ref.'))
            reference.data = parameter.detach().clone()
            reference.requires_grad_(False)
            copied += 1
    assert copied, 'No pretrained policy adapter parameters found'
    return copied
