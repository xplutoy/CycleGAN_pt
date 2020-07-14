import torch as T
import torchvision as tv
from torch.autograd import Variable

from data import valid_A, valid_B
from networks import ResnetGenerator

epoch = 100

# data
save_dir = './results/'

# network
input_nc = 3
output_nc = 3
ngf = 64
ndf = 64

netG_A = ResnetGenerator(input_nc, output_nc, ngf).cuda()
netG_B = ResnetGenerator(input_nc, output_nc, ngf).cuda()

netG_A.eval()
netG_B.eval()

if epoch >= 1:
    checkpoint = T.load(save_dir + 'ckpt_{}.ptz'.format(epoch))
    netG_A.load_state_dict(checkpoint['G_A'])
    netG_B.load_state_dict(checkpoint['G_B'])

batch = 0
for (A, _), (B, _) in zip(valid_A, valid_B):
    batch += 1

    a_real = Variable(A, volatile=True).cuda()
    b_fake = netG_A(a_real)
    a_rec = netG_B(b_fake)

    b_real = Variable(B, volatile=True).cuda()
    a_fake = netG_B(b_real)
    b_rec = netG_A(a_fake)

    tv.utils.save_image(T.cat([
        a_real.data * 0.5 + 0.5,
        b_fake.data * 0.5 + 0.5,
        a_rec.data * 0.5 + 0.5,
        b_real.data * 0.5 + 0.5,
        a_fake.data * 0.5 + 0.5,
        b_rec.data * 0.5 + 0.5], 0),
        save_dir + 'valid_images/{}_{}.png'.format(epoch, batch), 3)
