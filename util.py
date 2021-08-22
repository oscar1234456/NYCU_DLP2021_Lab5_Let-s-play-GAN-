import torch.nn as nn
# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     m.weight.data.normal_( 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_( 1.0, 0.02)
    #     m.bias.data.fill_(0)
        # nn.init.constant_(m.bias.data, 0)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)