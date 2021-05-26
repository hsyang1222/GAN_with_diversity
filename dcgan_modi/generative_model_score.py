import torch
from scipy.stats import entropy
from scipy import linalg
import numpy as np
import prdc
import pickle


class GenerativeModelScore:
    def __init__(self):
        
        self.inception_model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.inception_model.forward = self._forward
        self.inception_model.eval()

        self.real_images = None
        self.fake_images = None
        
        
        self.real_predict_softmax = None
        self.real_feature = None
        
        self.fake_predict_softmax = None
        self.fake_feature = None
        
        self.lazy = False
        
    def _forward(self, x):
        import torchvision

        if x.size(1) != 3 : 
            x = self.inception_model._transform_input(x)

        resize = torchvision.transforms.Resize((299, 299))
        x = resize(x)

        # N x 3 x 299 x 299
        x = self.inception_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception_model.avgpool(x)
        # N x 2048 x 1 x 1
        feature = x.detach()
        x = self.inception_model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.inception_model.fc(x)
        # N x 1000 (num_classes)
        return x, feature

    def predict_to_inception_score(self, predict, splits=1):
        preds = torch.softmax(predict, dim=1).numpy()

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)
    
    
    def trainloaderinfo_to_hashedname(self, train_loader) : 
        import hashlib
        dataset_info = str(train_loader.dataset).split('\n')
        name, datapoints, split = dataset_info[0:3]
        transform = dataset_info[4:]
        except_root_info = [name, datapoints, split] + transform
        name = hashlib.md5(str(except_root_info).encode()).hexdigest() + '.pickle'
        return name
        

    def feature_to_mu_sig(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    def model_to(self, device) : 
        self.inception_model = self.inception_model.to(device)
    
    def lazy_mode(self, tf) :
        self.lazy = tf
    
    def put_real(self, real_images) : 
        if self.lazy : 
            if self.real_images is None : 
                self.real_images = real_images
            else : 
                self.real_images = torch.cat([self.real_images, real_images])
        else : 
            self.real_forward(real_images)
            
            
    def real_forward(self, real_images):
        real_predict_softmax, real_feature = self.analysis_softmax_and_feature(real_images)
        if self.real_predict_softmax is None : 
            self.real_predict_softmax = real_predict_softmax.detach().cpu()
            self.real_feature = real_feature.detach().cpu()
        else : 
            self.real_predict_softmax = torch.cat([self.real_predict_softmax, real_predict_softmax.detach().cpu()])
            self.real_feature = torch.cat([self.real_feature, real_feature.detach().cpu()])    
                
  
        
    def put_fake(self, fake_images) : 
        if self.lazy : 
            if self.fake_images is None : 
                self.fake_images = fake_images
            else : 
                self.fake_images = torch.cat([self.fake_images, fake_images])
        else:
            self.fake_forward(fake_images)
            
    
    def fake_forward(self, fake_images) : 
        fake_predict_softmax, fake_feature = self.analysis_softmax_and_feature(fake_images)
        if self.fake_predict_softmax is None : 
            self.fake_predict_softmax = fake_predict_softmax.detach().cpu()
            self.fake_feature = fake_feature.detach().cpu()
        else : 
            self.fake_predict_softmax = torch.cat([self.fake_predict_softmax, fake_predict_softmax.detach().cpu()])
            self.fake_feature = torch.cat([self.fake_feature, fake_feature.detach().cpu()])
    
    
                  
    def lazy_forward(self, batch_size=64, shuffle=True, num_workers=4, real_forward=False, fake_forward=False, device='cpu') : 
        
        assert self.lazy, "lazy_forward only run in lazy mode. call lazy_mode() first."
        
        from torch.utils.data import  TensorDataset, DataLoader
        import tqdm 
        
        if real_forward : 
            real_dataset = TensorDataset(self.real_images)
            real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            print("generate real images info")
            for real_images in tqdm.tqdm(real_loader) : 
                self.real_forward(real_images[0].to(device))
        
        if fake_forward : 
            fake_dataset = TensorDataset(self.fake_images)
            fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            print("generate fake images info")
            for fake_images in tqdm.tqdm(fake_loader) :
                self.fake_forward(fake_images[0].to(device))
            
    def save_real_images_info(self, file_name='real_images_info.pickle') : 
        with open(file_name, 'wb') as f :
            pickle.dump((self.real_inception_score, self.real_feature_np, (self.real_mu, self.real_sigma)), f)
    
    def load_real_images_info(self, file_name='real_images_info.pickle'):
        with open(file_name, 'rb') as f:
            (self.real_inception_score, self.real_feature_np, (self.real_mu, self.real_sigma)) = \
                pickle.load(f)
            
    def calculate_real_image_statistics(self) : 
        self.real_inception_score = self.predict_to_inception_score(self.real_predict_softmax)[0]
        self.real_feature_np = self.real_feature.view(-1, 2048).numpy()
        self.real_mu, self.real_sigma = self.feature_to_mu_sig(self.real_feature_np)        
    
    def calculate_fake_image_statistics(self) : 
        self.fake_inception_score = self.predict_to_inception_score(self.fake_predict_softmax)[0]
        self.fake_feature_np = self.fake_feature.view(-1, 2048).numpy()
        self.fake_mu, self.fake_sigma = self.feature_to_mu_sig(self.fake_feature_np)
    
    def clear_fake(self) : 
        self.fake_images = None
        self.fake_predict_softmax = None
        self.fake_feature = None
        self.fake_mu, self.fake_sigma = None, None
        
    def calculate_generative_score(self):
        fid = self.calculate_frechet_distance(self.real_mu, self.real_sigma, self.fake_mu, self.fake_sigma)
        real_pick = np.random.permutation(self.real_feature_np)[:10000]
        fake_pick = np.random.permutation(self.fake_feature_np)[:10000]
        metrics = prdc.compute_prdc(real_features=real_pick, fake_features=fake_pick, nearest_k=5)
        metrics['fid'] = fid
        metrics['real_is'] = self.real_inception_score
        metrics['fake_is'] = self.fake_inception_score
        return metrics
    

    def analysis_softmax_and_feature(self, images):
        return self.inception_model(images)
