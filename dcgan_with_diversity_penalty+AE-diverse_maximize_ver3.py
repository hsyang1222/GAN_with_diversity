from __future__ import print_function
from dcgan import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
inception_model_score.lazy_mode(True)
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


import easydict
args = easydict.EasyDict({
    'dataset':'cifar10',
    'dataroot':'../dataset',
    'workers':2,
    'batchSize':2048,
    'imageSize':64,
    'nz':100,
    'ngf':64,
    'ndf':64,
    'niter':300,
    'lr':0.0002,
    'beta1':0.5,
    'cuda':True,
    'dry_run':False,
    'ngpu':1,
    'netG':'',
    'netD':'',
    'netE':'',
    'manualSeed':None,
    'classes':None,
    'outf':'result_image',
    'AEiter' : 1,
    'z_add':0.8,
    'lambda_diverse':0.05,
    'lambda_uniform' : 1
})


#opt = parser.parse_args()
opt = args
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, #download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:2" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 100, 4, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
        

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netE = Encoder(ngpu).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)


criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


train_loader = dataloader
print(train_loader.dataset)
import hashlib
real_images_info_file_name = inception_model_score.trainloaderinfo_to_hashedname(train_loader)
if os.path.exists('../../inception_model_info/' + real_images_info_file_name) : 
    print("Using exist inception model info from :", real_images_info_file_name)
    inception_model_score.load_real_images_info('../../inception_model_info/' + real_images_info_file_name)
else : 
    inception_model_score.model_to('cuda')

    #put real image
    for each_batch in train_loader : 
        X_train_batch = each_batch[0]
        inception_model_score.put_real(X_train_batch)

    #generate real images info
    inception_model_score.lazy_forward(batch_size=64, device='cuda', real_forward=True)
    inception_model_score.calculate_real_image_statistics()
    #save real images info for next experiments
    inception_model_score.save_real_images_info('../../inception_model_info/' + real_images_info_file_name)
    print("Save inception model info to :", real_images_info_file_name)
    #offload inception_model
    inception_model_score.model_to('cpu')
    
import wandb
wandb.init(project='GAN_with_diversit_maximize', config=opt)
config = wandb.config

mse = torch.nn.MSELoss()


'''
# skip AE
import tqdm
for epoch in tqdm.tqdm(range(config.AEiter), desc="AE"):
    loss_sum = 0.
    for i, data in enumerate(dataloader, 0):
        real_cuda = data[0].to(device)
        batch_size = real_cuda.size(0)
        
        latent_vector = netE(real_cuda)
        latent_4dim = latent_vector.view(batch_size,nz,1,1)
        repaint = netG(latent_4dim)
        
        mse_loss = mse(repaint, real_cuda)
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        mse_loss.backward()
        optimizerE.step()
        optimizerG.step()
        
        loss_sum += mse_loss.item()
        
del netE
'''     
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        z1 = torch.rand(batch_size, nz, 1, 1, device=device)

        fake_z1 = netG(z1)
        label.fill_(fake_label)
        predict = netD(fake_z1.detach())

        errD_fake = criterion(predict, label) 
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        if epoch % 10 == 0 :
            inception_model_score.put_fake(fake_z1.detach().cpu())
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        
        z2 = torch.rand(batch_size, nz, 1, 1, device=device)
        
        fake_z1 = netG(z1)
        fake_z2 = netG(z2)
        
        loss_maximize_div = torch.mean(torch.abs(fake_z1 - fake_z2)) # to maximize
        diff_fakes_z1z2 = torch.mean(torch.abs(fake_z1 - fake_z2), dim=(1,2,3)).detach()
        
        z_1dot5 = ((z1 + z2) / 2).detach()
        fake_z1dot5 = netG(z_1dot5)
        
        diff_fake_z1dot5_z1 = torch.mean(torch.abs(fake_z1.detach() - fake_z1dot5), dim=(1,2,3))
        diff_fake_z1dot5_z2 = torch.mean(torch.abs(fake_z2.detach() - fake_z1dot5), dim=(1,2,3))
        loss_uniform_diff = mse(diff_fake_z1dot5_z1, diff_fakes_z1z2/2) + mse(diff_fake_z1dot5_z2, diff_fakes_z1z2/2) # to uniform
  
        fake_z1 = netG(z1)
        label.fill_(real_label)
        predict = netD(fake_z1)
        loss_label_g = criterion(predict, label)
        
        errG = loss_label_g - config.lambda_diverse * loss_maximize_div + config.lambda_uniform * loss_uniform_diff
        errG.backward()
        D_G_z2 = predict.mean().item()
        optimizerG.step()

    if epoch % 10 == 0:
        netG = netG.to('cpu')
        netD = netD.to('cpu')
        inception_model_score.model_to(device)

        #generate fake images info
        inception_model_score.lazy_forward(batch_size=64, device=device, fake_forward=True)
        inception_model_score.calculate_fake_image_statistics()
        metrics = inception_model_score.calculate_generative_score()
        inception_model_score.clear_fake()

        #onload all GAN model to cpu and offload inception model to gpu
        netG = netG.to(device)
        netD = netD.to(device)
        inception_model_score.model_to('cpu')
        
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, DivMaxLoss : %.4f, DivUniformLoss ; %.4f'
          % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, loss_maximize_div.item(), loss_uniform_diff.item()))
        
        print("\t\tFID : %.4f, IS_f : %.4f, P : %.4f, R : %.4f, D : %.4f, C : %.4f" 
              %(metrics['fid'], metrics['fake_is'], metrics['precision'], metrics['recall'], metrics['density'], metrics['coverage']))
       
        vutils.save_image(real_cpu,
                '%s/real_samples.png' % opt.outf,
                normalize=True)
        fake_z1 = netG(fixed_noise)
        vutils.save_image(fake_z1.detach(),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)
        
        fake_z2 = netG(fixed_noise + config.z_add)
        vutils.save_image(fake_z2.detach(),
                '%s/fake2_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)
        
        fake_np = vutils.make_grid(fake_z1.detach().cpu(), nrow=32).permute(1,2,0).numpy()
        fake2_np = vutils.make_grid(fake_z2.detach().cpu(), nrow=32).permute(1,2,0).numpy()
        
        wandb.log({
            "epoch" : epoch,
            "Loss_D": errD.item(),
            "Loss_G": errG.item(),
            "D(real)": D_x,
            "D(G(z))-before D train": D_G_z1,
            "D(G(z))-after D train": D_G_z2,
            "DivMaxLoss" : loss_maximize_div.item(),
            "DivUniformLoss" : loss_uniform_diff.item(),
            "fid" : metrics['fid'],
            'fake_is':metrics['fake_is'],
            "precision":metrics['precision'],
            "recall":metrics['recall'],
            "density":metrics['density'],
            "coverage":metrics['coverage'],
            "G(z) " : [wandb.Image(fake_np, caption='fixed z image')],
            "G(z + div_add) " : [wandb.Image(fake2_np, caption='fixed z + add image')],
        })

        if opt.dry_run:
            break
    # do checkpointing
    '''
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    
    for i, data in enumerate(dataloader, 0):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        vutils.save_image(fake.detach(),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)
    '''
