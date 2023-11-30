import torch
import torch.optim as optim


def center_and_rescale(dat):
    # centers points on the origin and makes std in each dim = 1
    means = np.mean(dat, 0)  # this is litterally the center of mass, assuming all particles have equal mass

    dat2 = (dat - means[None, :]) / 250
    return dat2


def save_checkpoint(save_name, model, z, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    z_ts = torch.stack(z, dim=0)  # Concatenates a sequence of tensors along a new dimension.
    torch.save(z_ts, save_name + '_latents.pt')
    torch.save(state, save_name + '.pth')
    print('model saved to {}'.format(save_name))


def load_checkpoint(save_name, model, optimizer):
    z_ts = torch.load(save_name + '_latents.pt')
    z_lst = []
    for i in range(z_ts.size()[0]):
        z_lst.append(z_ts[i, :])
    if model is None:
        pass
    else:
        model_CKPT = torch.load(save_name + '.pth')
        model.load_state_dict(model_CKPT['state_dict'])  # load weights into model
        # model.cuda()
        # optimizer = optim.Adam(model.parameters())
        print('loading checkpoint!')
        # optimizer.load_state_dict(model_CKPT['optimizer'])

    return model, z_lst, optimizer
