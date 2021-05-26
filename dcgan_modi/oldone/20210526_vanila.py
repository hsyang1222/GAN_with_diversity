from __future__ import print_function
import generative_model_score
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
import lpips

import easydict
args = easydict.EasyDict({
    'dataset':'cifar10',
    'dataroot':'../../dataset',
    'workers':4,
    'batchSize':2048,
    'imageSize':32,
    'nz':100,
    'ngf':64,
    'ndf':64,
    'niter':500,
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
    'AEiter' : 0,
    'z_add':0.8,
    'lambda_diverse': 0.0,
    'lambda_uniform' : 0,
    'try_div_chance' : 0,
    'device':'cuda:3',
    'name' : 'vanila',
    'report_every' : 10,
    'keep_try_over' : 0.8,
    'mul_alpha_param' : 0.06
})

lpips_model = loss_fn = lpips.LPIPS(net='alex', spatial=True)

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

device = torch.device(opt.device if opt.cuda else "cpu")
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
            #nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
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
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            # state size. 1x1x1
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
            #nn.Conv2d(ndf * 8, 100, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, nz, 2, 1, 0, bias=False),
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


real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


train_loader = dataloader
print(train_loader.dataset)
real_images_info_file_name = inception_model_score.trainloaderinfo_to_hashedname(train_loader)
real_image_info_path = '../../../inception_model_info/'+real_images_info_file_name

if os.path.exists( real_image_info_path) : 
    print("Using exist inception model info from :",real_image_info_path)
    inception_model_score.load_real_images_info(real_image_info_path)
else : 
    inception_model_score.model_to(device)

    #put real image
    for each_batch in train_loader : 
        X_train_batch = each_batch[0]
        inception_model_score.put_real(X_train_batch)

    #generate real images info
    inception_model_score.lazy_forward(batch_size=64, device=device, real_forward=True)
    inception_model_score.calculate_real_image_statistics()
    #save real images info for next experiments
    inception_model_score.save_real_images_info(real_image_info_path)
    print("Save inception model info to :", real_images_info_file_name)
    #offload inception_model
    inception_model_score.model_to('cpu')
    
import wandb
wandb.init(project='GAN_mul_alpha', name=opt.name, config=opt)
config = wandb.config


mse = torch.nn.MSELoss()

import tqdm
for epoch in tqdm.tqdm(range(config.AEiter), desc="AE"):
    loss_sum = 0.
    for i, data in enumerate(dataloader, 0):
        real_cuda = data[0].to(device)
        batch_size = real_cuda.size(0)
        
        latent_vector = netE(real_cuda)
        real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
        repaint = netG(real_latent_4dim)
        
        mse_loss = mse(repaint, real_cuda)
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        mse_loss.backward()
        optimizerE.step()
        optimizerG.step()
        
        loss_sum += mse_loss.item()
                
    print(epoch, loss_sum)
    
mul_alpha = torch.tensor([1.], requires_grad=True, device=device)
optimizerM = torch.optim.Adam([mul_alpha], lr=0.001) 

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device) #real_latent_4dim.detach()

for epoch in range(opt.niter + 1):
    for i, data in enumerate(tqdm.tqdm(dataloader, desc='batch')):
        real_cuda = data[0].to(device)
        batch_size = real_cuda.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cuda.dtype, device=device)
        
        #############################
        # generate original_feature
        #############################
        '''
        latent_vector = netE(real_cuda)
        real_latent_4dim = latent_vector.view(batch_size,nz,1,1)
        repaint = netG(real_latent_4dim)
  
        mse_loss = mse(repaint, real_cuda)
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        mse_loss.backward()
        optimizerE.step()
        optimizerG.step()
    
        real_feature = real_latent_4dim.detach()
        '''
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        output_real = netD(real_cuda)
        errD_real = criterion(output_real, label)
        errD_real.backward()
        D_x = output_real.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        '''
        real_feature_noised  = (1-mul_alpha)*real_feature + mul_alpha*noise
        
        repaint_real = netG(real_feature)
        '''
        fake = netG(noise)
        
        if config.report_every > 0 and epoch % config.report_every == 0 :
            inception_model_score.put_fake(fake.detach().cpu())
        
        label.fill_(fake_label)
        #output_repaint_real = netD(repaint_real)
        output_fake = netD(fake)
        errD_fake = criterion(output_fake, label) #+ criterion(output_repaint_real, label)
        D_G_z1 = output_fake.mean().item()
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        
        #real_feature_noised  = (1-mul_alpha)*real_feature + mul_alpha*noise
        #repaint_real = netG(real_feature)
        fake = netG(noise)

        #maximize_diff = torch.mean(torch.abs(repaint_real - fake))

        output_fake = netD(fake)
        #output_repaint_real = netD(repaint_real)

        #maximize_output_fake = torch.mean(output_repaint_real - output_fake)

        #mul_alpha_loss = -(config.mul_alpha_param * maximize_diff + maximize_output_fake)
        #optimizerM.zero_grad()
        #mul_alpha_loss.backward(retain_graph = True)
        #optimizerM.step()
        
        errG = criterion(output_fake, label) #+ criterion(output_repaint_real, label)
        errG.backward()
        D_G_z2 = output_fake.mean().item()
        optimizerG.step()
        

    if config.report_every > 0 and epoch % config.report_every == 0:
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=True)

        #real_feature_noised  = (1-mul_alpha)*fixed_noise + mul_alpha*noise
        fake2 = netG(fixed_noise+1e-6)
        vutils.save_image(fake2.detach(),
            '%s/fake2_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=True)
  
        fake_np = vutils.make_grid(fake.detach().cpu(), nrow=32).permute(1,2,0).numpy()
        fake2_np = vutils.make_grid(fake2.detach().cpu(), nrow=32).permute(1,2,0).numpy()
        
        netG = netG.to('cpu')
        netD = netD.to('cpu')
        
        lpips_model.to(device)
        ex_d = lpips_model.forward(fake, fake2).mean()
        lpips_model.to('cpu')
        
        inception_model_score.model_to(device)

        #generate fake images info
        inception_model_score.lazy_forward(batch_size=64, device=device, fake_forward=True)
        inception_model_score.calculate_fake_image_statistics()
        metrics = inception_model_score.calculate_generative_score()
        inception_model_score.clear_fake()

        #onload all GAN model to cpu and offload inception model to gpu
        inception_model_score.model_to('cpu')
        netG = netG.to(device)
        netD = netD.to(device)
        
        
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,))
        
        print("\t\tFID : %.4f, IS_f : %.4f, P : %.4f, R : %.4f, D : %.4f, C : %.4f, LPIPS : %.4f"
              %(metrics['fid'], metrics['fake_is'], metrics['precision'], 
                metrics['recall'], metrics['density'], metrics['coverage'], ex_d))
        

        wandb.log({
            "epoch" : epoch,
            "Loss_D": errD.item(),
            "Loss_G": errG.item(),
            "D(real)": D_x,
            "D(G(z))-before D train": D_G_z1,
            "D(G(z))-after D train": D_G_z2,
            "fid" : metrics['fid'],
            'fake_is':metrics['fake_is'],
            "precision":metrics['precision'],
            "recall":metrics['recall'],
            "density":metrics['density'],
            "coverage":metrics['coverage'],
            "G(z) " : [wandb.Image(fake_np, caption='fixed z image')],
            "G(z + div_add) " : [wandb.Image(fake2_np, caption='fixed z + %.4f' % mul_alpha)],
            'LPIPS' : ex_d,
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
