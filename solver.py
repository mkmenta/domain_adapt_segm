import datetime
import time

import os
import re
import torch
import torch.nn.functional as F

from models.da_discriminators import FeatureDiscriminator, PixelwiseFeatureDiscriminator
from models.segmenter_baseline import Segmenter
from models.stargan import Generator, Discriminator
from utils.metrics import update_cm, compute_metrics, softIoULoss, print_metrics
from utils.segmentation2rgb import segmentation2rgb
from utils.visualizer import Visualizer


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, mix_loader, source_loader, mix_loader_val, source_loader_val,
                 target_loader=None, target_loader_val=None):
        """Initialize configurations."""

        # Data loader.
        if mix_loader_val is not None:
            self.mix_loader = mix_loader
            self.source_loader = source_loader
            self.mix_loader_val = mix_loader_val
            self.source_loader_val = source_loader_val
            self.target_loader = target_loader
            self.target_loader_val = target_loader_val
        else:
            self.mix_loader = mix_loader
            self.source_loader = source_loader
            self.target_loader = source_loader_val
        self.image_size = 32
        self.n_classes = self.source_loader.dataset.n_classes

        # Model configurations generator and discriminator
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_num_down = config.d_num_down
        self.df_num_down = config.df_num_down
        self.g_num_init = config.g_num_init
        self.g_num_down = config.g_num_down
        self.g_num_up = config.g_num_up
        self.df_num_up = config.df_num_up
        self.lambda_cls = config.lambda_cls
        self.lambda_cycle = config.lambda_cycle
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_g = 1.
        self.lambda_loss_disc = 1.
        self.lambda_fdom = config.lambda_fdom
        self.lambda_ffeat = config.lambda_ffeat
        self.lambda_frf = config.lambda_frf

        if self.lambda_cycle == 0:
            self.lambda_cls = 0.
            self.lambda_g = 0.
            self.lambda_loss_disc = 0.

        # Model configuration segmenter
        self.s_conv_dim = config.s_conv_dim
        self.s_repeat_num = config.s_repeat_num
        self.s_num_init = config.s_num_init
        self.s_num_down = config.s_num_down
        self.s_num_up = config.s_num_up
        self.lambda_segm = config.lambda_segm
        self.fake_segm = config.fake_segm
        self.da_type = config.da_type

        self.drop_g = config.drop_g
        self.drop_d = config.drop_d

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.patience = config.patience
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.s_lr = config.s_lr
        self.df_lr = config.df_lr
        self.lr_decay = config.lr_decay
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.segm_criterion = softIoULoss()
        self.oracle_cond = config.oracle_cond
        self.load_pretrained = config.load_pretrained
        self.modules_pretrained = config.modules_pretrained
        self.df_source_only = config.df_source_only
        self.df_move_one = config.df_move_one

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir

        # Step size.
        self.log_step = config.log_step
        self.lr_update_step = config.lr_update_step
        self.val_step = config.val_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard and config.mode == 'train':
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(conv_dim=self.g_conv_dim, repeat_num=self.g_repeat_num, num_down=self.g_num_down,
                           num_up=self.g_num_up, num_init=self.g_num_init, drop=self.drop_g)
        self.D = Discriminator(image_size=self.image_size, conv_dim=self.d_conv_dim, repeat_num=self.d_num_down,
                               drop=self.drop_d)
        self.S = Segmenter(conv_dim=self.s_conv_dim, repeat_num=self.s_repeat_num, num_down=self.s_num_down,
                           num_up=self.s_num_up, num_init=self.s_num_init, drop=self.drop_g,
                           in_channels=self.G.bottleneck_dim)

        if self.da_type == 'uncond':
            self.Df = FeatureDiscriminator(inplanes=self.G.bottleneck_dim, seg_nclasses=0, num_ups_feat=0,
                                           num_downs=self.df_num_down, drop=self.drop_d)
        elif self.da_type == 'input_cond':
            self.Df = FeatureDiscriminator(inplanes=self.G.bottleneck_dim, seg_nclasses=2, num_ups_feat=self.g_num_down,
                                           num_downs=self.df_num_down, drop=self.drop_d)
        elif self.da_type == 'output_cond':
            self.Df = PixelwiseFeatureDiscriminator(inplanes=self.G.bottleneck_dim, num_ups=self.df_num_up,
                                                    drop=self.drop_d)
        else:
            self.Df = None

        if self.load_pretrained is not None:
            for m_str in self.modules_pretrained:
                m = getattr(self, m_str)
                self.restore_model(m, m_str, self.load_pretrained)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.s_optimizer = torch.optim.Adam(self.S.parameters(), self.s_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.S, 'S')
        self.G.to(self.device)
        self.D.to(self.device)
        self.S.to(self.device)

        if self.Df is not None:
            self.df_optimizer = torch.optim.Adam(self.Df.parameters(), self.df_lr, [self.beta1, self.beta2])
            self.print_network(self.Df, 'D')
            self.Df.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    @staticmethod
    def restore_model(model, model_str, log_dir):
        """Restore the trained generator and discriminator."""
        # print('Loading the trained models')
        path = os.path.join(log_dir, 'best-{}.ckpt'.format(model_str))
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Visualizer(self.log_dir, name='visual_results')

    def update_lr(self, g_lr, d_lr, s_lr, df_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.s_optimizer.param_groups:
            param_group['lr'] = s_lr
        if self.Df is not None:
            for param_group in self.df_optimizer.param_groups:
                param_group['lr'] = df_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.s_optimizer.zero_grad()

        if self.Df is not None:
            self.df_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    @staticmethod
    def label2onehot2D(labels, C):
        # labels.shape = BSxHxW
        # C = number of classes
        labels = labels.unsqueeze(1)
        one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3)).to(labels.device)
        one_hot = one_hot.scatter_(1, labels.data, 1)
        return one_hot

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def save_model(self, label):
        G_path = os.path.join(self.log_dir, '{}-G.ckpt'.format(label))
        D_path = os.path.join(self.log_dir, '{}-D.ckpt'.format(label))
        S_path = os.path.join(self.log_dir, '{}-S.ckpt'.format(label))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.S.state_dict(), S_path)
        G_optim_path = os.path.join(self.log_dir, '{}-G_optim.ckpt'.format(label))
        D_optim_path = os.path.join(self.log_dir, '{}-D_optim.ckpt'.format(label))
        S_optim_path = os.path.join(self.log_dir, '{}-S_optim.ckpt'.format(label))
        torch.save(self.g_optimizer.state_dict(), G_optim_path)
        torch.save(self.d_optimizer.state_dict(), D_optim_path)
        torch.save(self.s_optimizer.state_dict(), S_optim_path)

        if self.Df is not None:
            Df_path = os.path.join(self.log_dir, '{}-Df.ckpt'.format(label))
            torch.save(self.Df.state_dict(), Df_path)
            Df_optim_path = os.path.join(self.log_dir, '{}-Df_optim.ckpt'.format(label))
            torch.save(self.df_optimizer.state_dict(), Df_optim_path)

        print('Saved model checkpoints into {}...'.format(self.log_dir))

    def tb_images(self, x, c_org, epoch, mode):
        with torch.no_grad():
            x_fake, _ = self.G(x, 1 - c_org)
            x_cycle, _ = self.G(x_fake, c_org)
            s = self.S(self.G(x, torch.ones(x.size(0), 1).to(self.device))[1])
            x_id, _ = self.G(x, c_org)
            self.logger.image_summary(mode=mode, epoch=epoch, label='image',
                                      images=self.denorm(x))
            self.logger.image_summary(mode=mode, epoch=epoch, label='translation',
                                      images=self.denorm(x_fake))
            self.logger.image_summary(mode=mode, epoch=epoch, label='cycle',
                                      images=self.denorm(x_cycle))
            self.logger.image_summary(mode=mode, epoch=epoch, label='identity',
                                      images=self.denorm(x_id))
            self.logger.image_summary(mode=mode, epoch=epoch, label='segmentation',
                                      images=segmentation2rgb(s.argmax(1), n_labels=2))
            print('Saved real and fake images...')

    def train(self):
        """Train StarGAN within a single dataset."""
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        # Fetch fixed inputs for debugging.
        mix_iter = iter(self.mix_loader)
        x_fixed, c_org_fixed, _ = next(mix_iter)
        x_fixed = x_fixed.to(self.device)
        c_org_fixed = c_org_fixed.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        s_lr = self.s_lr
        df_lr = self.df_lr

        # Start training from scratch or resume training.
        start_iters = 0
        """
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        """

        # Start training.
        print('Start training...')
        start_time = time.time()
        epoch = 0
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, c_org, gt_real = next(mix_iter)
            except:
                mix_iter = iter(self.mix_loader)
                x_real, c_org, gt_real = next(mix_iter)

            # Fetch source images and masks
            try:
                x_source, gt_source = next(source_iter)
                if x_source.size(0) < self.batch_size:
                    raise Exception
            except:
                source_iter = iter(self.source_loader)
                x_source, gt_source = next(source_iter)

            # Fetch target images and masks
            try:
                x_target, gt_target = next(target_iter)
                if x_target.size(0) < self.batch_size:
                    raise Exception
            except:
                target_iter = iter(self.target_loader)
                x_target, gt_target = next(target_iter)

            x_real = x_real.to(self.device)  # Input images.
            x_source = x_source.to(self.device)
            x_target = x_target.to(self.device)
            gt_source = gt_source.to(self.device)
            gt_target = gt_target.to(self.device)
            gt_real = gt_real.to(self.device)
            c_org = c_org.to(self.device)  # Original domain labels.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            d_loss, loss_log = self.D_losses(x_real, c_org)
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # =================================================================================== #
            #                             3. Train the feature discriminator                      #
            # =================================================================================== #
            if self.Df is not None:
                # Backward and optimize.
                df_loss, log = self.Df_losses(x_source, x_target, gt_source, gt_target, x_real, c_org, gt_real)
                loss_log.update(log)
                self.reset_grad()
                df_loss.backward()
                self.df_optimizer.step()

            # =================================================================================== #
            #                               4. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                g_loss, _, log = self.G_losses(x_real, c_org, gt_real, x_source, gt_source, x_target, gt_target)
                loss_log.update(log)
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                self.s_optimizer.step()

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss_log.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                # Log in tensorboard
                if self.use_tensorboard:
                    self.logger.scalar_summary(mode='train', epoch=i + 1, **loss_log)

            if (i + 1) % self.val_step == 0:
                print('Epoch {} finished'.format(epoch))
                # Translate fixed images for debugging.
                self.tb_images(x_fixed, c_org_fixed, epoch, 'train')

                # Validation
                print('Validation...')
                g_loss_val = self.validation(epoch)
                self.G.train()
                self.D.train()
                if self.Df is not None:
                    self.Df.train()
                self.S.train()

                # Compute patience and save best model
                if epoch == 0 or g_loss_val < es_best:
                    es_best = g_loss_val
                    print('Found new best model.')
                    self.save_model('best')
                    curr_pat = 0
                else:
                    curr_pat += 1
                    print('Patience {}/{}'.format(curr_pat, self.patience))

                if curr_pat > self.patience:
                    print('Early stopping')
                    break

                # Save last model
                self.save_model('last')
                epoch += 1

                # Decay learning rates.
                g_lr *= self.lr_decay
                d_lr *= self.lr_decay
                s_lr *= self.lr_decay
                df_lr *= self.lr_decay
                self.update_lr(g_lr, d_lr, s_lr, df_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        self.logger.close()

    # =================================================================================== #
    #                                         D                                           #
    # =================================================================================== #
    def D_losses(self, x_real, c_org):
        # Compute loss with real images.
        out_src, out_cls = self.D(x_real)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_org)

        # Compute loss with fake images.
        x_fake, _ = self.G(x_real, 1 - c_org)
        out_src, out_cls = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = self.lambda_loss_disc * (
                d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp)
        return d_loss, {'D/loss_real': d_loss_real.item(),
                        'D/loss_fake': d_loss_fake.item(),
                        'D/loss_cls': d_loss_cls.item(),
                        'D/loss_gp': d_loss_gp.item(),
                        'D/loss': d_loss.item()}

    # =================================================================================== #
    #                                         Df                                           #
    # =================================================================================== #
    def Df_losses(self, x_source, x_target, gt_source, gt_target, x_real, c_org, gt_real):
        # ============================== Source vs. target ================================== #
        if self.lambda_fdom > 0:
            # Features
            _, h_source = self.G(x_source, torch.ones(x_source.size(0), 1).to(self.device))
            if self.df_source_only:
                x_fake, _ = self.G(x_source, torch.zeros(x_source.size(0), 1).to(self.device))
                _, h_target = self.G(x_fake, torch.ones(x_source.size(0), 1).to(self.device)) # be careful!! zeros vs ones
            else:
                _, h_target = self.G(x_target, torch.ones(x_target.size(0), 1).to(self.device))  # be careful!! zeros vs ones

            # Interpolation features for GP
            alpha = torch.rand(h_source.size(0), 1, 1, 1).to(self.device)
            h_hat = (alpha * h_source.data + (1 - alpha) * h_target.data).requires_grad_(True)

            # Segmentations (conditioning case)
            if self.da_type in ['input_cond', 'output_cond']:
                if self.oracle_cond:
                    s_source_sm = self.label2onehot2D(gt_source, self.n_classes)
                    if self.df_source_only:
                        s_target_sm = s_source_sm.clone()
                    else:
                        s_target_sm = self.label2onehot2D(gt_target, self.n_classes)
                else:
                    s_source_sm = F.softmax(self.S(h_source), 1).detach()
                    s_target_sm = F.softmax(self.S(h_target), 1).detach()
                s_hat_sm = (alpha * s_source_sm.data + (1 - alpha) * s_target_sm.data).requires_grad_(True)
            else:
                s_source_sm = None
                s_target_sm = None
                s_hat_sm = None

            # Forward Df passes
            if self.da_type == 'output_cond':
                _, df_source_dom = self.Df(h_source.detach())
                _, df_target_dom = self.Df(h_target.detach())
                _, df_h_hat_dom = self.Df(h_hat)

                df_source_dom = (df_source_dom * s_source_sm).view(s_source_sm.shape[0], self.n_classes, -1).sum(2) \
                                / s_source_sm.view(s_source_sm.shape[0], self.n_classes, -1).sum(2)
                df_target_dom = (df_target_dom * s_target_sm).view(s_target_sm.shape[0], self.n_classes, -1).sum(2) \
                                / s_target_sm.view(s_target_sm.shape[0], self.n_classes, -1).sum(2)
                df_h_hat_dom = (df_h_hat_dom * s_hat_sm).view(s_hat_sm.shape[0], self.n_classes, -1).sum(2) \
                               / s_hat_sm.view(s_hat_sm.shape[0], self.n_classes, -1).sum(2)
            else:
                _, df_source_dom = self.Df(h_source.detach(), s_source_sm)
                _, df_target_dom = self.Df(h_target.detach(), s_target_sm)
                _, df_h_hat_dom = self.Df(h_hat, s_hat_sm)

            df_loss_fdom_source = - torch.mean(df_source_dom)
            df_loss_fdom_target = torch.mean(df_target_dom)
            df_loss_fdom_gp = self.gradient_penalty(df_h_hat_dom, h_hat)

            df_loss_fdom = df_loss_fdom_source + df_loss_fdom_target + self.lambda_gp * df_loss_fdom_gp
        else:
            df_loss_fdom = df_loss_fdom_gp = torch.zeros(1, requires_grad=True).to(self.device)

        # ================================ Real vs. fake ==================================== #
        if self.lambda_frf > 0:
            # Features
            if self.df_source_only:
                _, h_real = self.G(x_source, torch.ones(x_source.size(0), 1).to(self.device))
                x_fake, _ = self.G(x_source, torch.zeros(x_source.size(0), 1).to(self.device))
                _, h_fake = self.G(x_fake, torch.ones(x_real.size(0), 1).to(self.device))  ## careful here!
            else:
                _, h_real = self.G(x_real, torch.ones(x_real.size(0), 1).to(self.device))  ## careful: c_org
                x_fake, _ = self.G(x_real, 1 - c_org)
                _, h_fake = self.G(x_fake, torch.ones(x_real.size(0), 1).to(self.device))  # c_org

            # Interpolation features for GP
            alpha = torch.rand(h_real.size(0), 1, 1, 1).to(self.device)
            h_hat = (alpha * h_real.data + (1 - alpha) * h_fake.data).requires_grad_(True)

            # Segmentations (conditioning case)
            if self.da_type in ['input_cond', 'output_cond']:
                if self.oracle_cond:
                    s_real_sm = self.label2onehot2D(gt_real if not self.df_source_only else gt_source, self.n_classes)
                    s_fake_sm = s_real_sm.clone()
                else:
                    s_real_sm = F.softmax(self.S(h_real), 1).detach()
                    s_fake_sm = F.softmax(self.S(h_fake), 1).detach()
                s_hat_sm = (alpha * s_real_sm.data + (1 - alpha) * s_fake_sm.data).requires_grad_(True)
            else:
                s_real_sm = None
                s_fake_sm = None
                s_hat_sm = None

            # Forward Df passes
            if self.da_type == 'output_cond':
                df_real_rf, _ = self.Df(h_real.detach())
                df_fake_rf, _ = self.Df(h_fake.detach())
                df_hat_rf, _ = self.Df(h_hat)

                # df_real_rf = (df_real_rf * s_real_sm).view(s_real_sm.shape[0], self.n_classes, -1).sum(2) \
                #              / s_real_sm.view(s_real_sm.shape[0], self.n_classes, -1).sum(2)
                # df_fake_rf = (df_fake_rf * s_fake_sm).view(s_fake_sm.shape[0], self.n_classes, -1).sum(2) \
                #              / s_fake_sm.view(s_fake_sm.shape[0], self.n_classes, -1).sum(2)
                # df_hat_rf = (df_hat_rf * s_hat_sm).view(s_hat_sm.shape[0], self.n_classes, -1).sum(2) \
                #             / s_hat_sm.view(s_hat_sm.shape[0], self.n_classes, -1).sum(2)
            else:
                df_real_rf, _ = self.Df(h_real.detach(), s_real_sm)
                df_fake_rf, _ = self.Df(h_fake.detach(), s_fake_sm)
                df_hat_rf, _ = self.Df(h_hat, s_hat_sm)

            df_loss_frf_real = - torch.mean(df_real_rf)
            df_loss_frf_fake = torch.mean(df_fake_rf)
            df_loss_frf_gp = self.gradient_penalty(df_hat_rf, h_hat)

            df_loss_frf = df_loss_frf_fake + df_loss_frf_real + self.lambda_gp * df_loss_frf_gp
        else:
            df_loss_frf = df_loss_frf_gp = torch.zeros(1, requires_grad=True).to(self.device)

        # =================================== Df loss ======================================= #
        df_loss = self.lambda_fdom * df_loss_fdom + self.lambda_frf * df_loss_frf

        return df_loss, {'Df/loss_dom': df_loss_fdom.item(),
                         'Df/loss_gp_dom': df_loss_fdom_gp.item(),
                         'Df/loss_rf': df_loss_frf.item(),
                         'Df/loss_gp_rf': df_loss_frf_gp.item(),
                         'Df/loss': df_loss.item()}

    # =================================================================================== #
    #                                         G                                           #
    # =================================================================================== #
    def G_losses(self, x_real, c_org, gt_real, x_source, gt_source, x_target, gt_target):
        # ============== Translation vs. real and translation classification ================ #
        x_fake, f_real = self.G(x_real, 1 - c_org)
        out_src, out_cls = self.D(x_fake)
        g_loss_fake = - torch.mean(out_src)
        g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, 1 - c_org)

        # ===================================== Cycle ======================================= #
        x_cycle, f_fake = self.G(x_fake, c_org)
        g_loss_cycle = torch.mean(torch.abs(x_real - x_cycle))

        # ================================= Identity loss =================================== #
        x_id, _ = self.G(x_real, c_org)
        id_loss = torch.mean(torch.abs(x_real - x_id))

        # ================================= Segmentation ==================================== #
        _, h_source = self.G(x_source, torch.ones(x_source.size(0), 1).to(self.device))
        s_source = self.S(h_source)

        if self.fake_segm:
            x_fake_target, _ = self.G(x_source, torch.zeros(x_source.size(0), 1).to(self.device))
            _, h_fake_target = self.G(x_fake_target, torch.zeros(x_source.size(0), 1).to(self.device))  # careful ones
            s_fake = self.S(h_fake_target)
            sg_loss_segm_aux = self.segm_criterion(s_fake, gt_source)
        else:
            sg_loss_segm_aux = 0.

        sg_loss_segm = self.segm_criterion(s_source, gt_source) + sg_loss_segm_aux

        # ============================= L1 feature matching ================================= #
        ge_loss_ffeat = F.l1_loss(f_fake, f_real)

        # ============================== Source vs. target ================================== #
        if self.lambda_fdom > 0 and self.Df is not None:
            if self.df_source_only:
                x_fake, _ = self.G(x_source, torch.zeros(x_source.size(0), 1).to(self.device))
                _, h_target = self.G(x_fake, torch.zeros(x_fake.size(0), 1).to(self.device))  ## be careful
            else:
                _, h_target = self.G(x_target, torch.ones(x_target.size(0), 1).to(self.device))  ## careful here! zeros vs ones
            s_target = self.S(h_target)

            if self.da_type in ['input_cond', 'output_cond']:
                if self.oracle_cond:
                    s_source_sm = self.label2onehot2D(gt_source, self.n_classes)
                    if self.df_source_only:
                        s_target_sm = s_source_sm.clone()
                    else:
                        s_target_sm = self.label2onehot2D(gt_target, self.n_classes)
                else:
                    s_source_sm = F.softmax(s_source, 1).detach()
                    s_target_sm = F.softmax(s_target, 1).detach()
            else:
                s_source_sm = None
                s_target_sm = None

            # Feature adversarial loss (source/target)
            if self.da_type == 'output_cond':
                _, df_source_dom = self.Df(h_source)
                _, df_target_dom = self.Df(h_target)

                df_source_dom = (df_source_dom * s_source_sm).view(s_source_sm.shape[0], self.n_classes, -1).sum(2) \
                                / s_source_sm.view(s_source_sm.shape[0], self.n_classes, -1).sum(2)
                df_target_dom = (df_target_dom * s_target_sm).view(s_target_sm.shape[0], self.n_classes, -1).sum(2) \
                                / s_target_sm.view(s_target_sm.shape[0], self.n_classes, -1).sum(2)
            else:
                _, df_source_dom = self.Df(h_source, s_source_sm)
                _, df_target_dom = self.Df(h_target, s_target_sm)

            if not self.df_move_one:
                ge_loss_fdom = (df_source_dom.mean(0) - df_target_dom.mean(0)).mean()  # (df_source_dom.mean(0) - df_target_dom.mean(0)).abs().mean()
            else:
                ge_loss_fdom = - df_target_dom.mean()

        else:
            ge_loss_fdom = torch.zeros(1, requires_grad=True).to(self.device)

        # ================================ Real vs. fake ==================================== #
        if self.lambda_frf > 0 and self.Df is not None:
            # Features
            if self.df_source_only:
                _, h_real = self.G(x_source, torch.ones(x_source.size(0), 1).to(self.device))
                x_fake, _ = self.G(x_source, torch.zeros(x_source.size(0), 1).to(self.device))
                _, h_fake = self.G(x_fake, torch.ones(x_fake.size(0), 1).to(self.device))  ##  careful here!
            else:
                # Features
                _, h_real = self.G(x_real, torch.ones(x_source.size(0), 1).to(self.device))  ## c_org
                x_fake, _ = self.G(x_real, 1 - c_org)
                _, h_fake = self.G(x_fake, torch.ones(x_source.size(0), 1).to(self.device))  ## (1 - c_org)

            # Segmentations (conditioning case)
            if self.da_type in ['input_cond', 'output_cond']:
                if self.oracle_cond:
                    s_real_sm = self.label2onehot2D(gt_real if not self.df_source_only else gt_source, self.n_classes)
                    s_fake_sm = s_real_sm.clone()
                else:
                    s_real_sm = F.softmax(self.S(h_real), 1).detach()
                    s_fake_sm = F.softmax(self.S(h_fake), 1).detach()
            else:
                s_real_sm = None
                s_fake_sm = None

            # Forward Df passes
            if self.da_type == 'output_cond':
                df_real_rf, _ = self.Df(h_real)
                df_fake_rf, _ = self.Df(h_fake)

                # df_real_rf = (df_real_rf * s_real_sm).view(s_real_sm.shape[0], self.n_classes, -1).sum(2) \
                #              / s_real_sm.view(s_real_sm.shape[0], self.n_classes, -1).sum(2)
                # df_fake_rf = (df_fake_rf * s_fake_sm).view(s_fake_sm.shape[0], self.n_classes, -1).sum(2) \
                #              / s_fake_sm.view(s_fake_sm.shape[0], self.n_classes, -1).sum(2)
            else:
                df_real_rf, _ = self.Df(h_real, s_real_sm)
                df_fake_rf, _ = self.Df(h_fake, s_fake_sm)

            if not self.df_move_one:
                ge_loss_frf = (df_real_rf.mean(0) - df_fake_rf.mean(0)).mean()  # (df_real_rf.mean(0) - df_fake_rf.mean(0)).abs().mean()
            else:
                ge_loss_frf = - df_fake_rf.mean()

        else:
            ge_loss_frf = torch.zeros(1, requires_grad=True).to(self.device)

        # =================================== G loss ======================================== #
        g_loss = self.lambda_g * g_loss_fake + self.lambda_cycle * g_loss_cycle + \
                 self.lambda_cls * g_loss_cls + self.lambda_segm * sg_loss_segm + \
                 self.lambda_id * id_loss + self.lambda_fdom * ge_loss_fdom + \
                 self.lambda_frf * ge_loss_frf + self.lambda_ffeat * ge_loss_ffeat

        return g_loss, s_source, {'G/loss_fake': g_loss_fake.item(),
                                  'G/loss_cycle': g_loss_cycle.item(),
                                  'G/loss_cls': g_loss_cls.item(),
                                  'G/loss': g_loss.item(),
                                  'S/loss_segm': sg_loss_segm.item(),
                                  'G/loss_id': id_loss.item(),
                                  'Ge/loss_ffeat': ge_loss_ffeat.item(),
                                  'Ge/loss_fdom': ge_loss_fdom.item(),
                                  'Ge/loss_frf': ge_loss_frf.item()}

    def validation(self, epoch):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.G.eval()
        self.D.eval()
        if self.Df is not None:
            self.Df.eval()
        self.S.eval()

        mix_iter = iter(self.mix_loader_val)
        source_iter = iter(self.source_loader_val)
        target_iter = iter(self.target_loader_val)

        # Evaluate segmentation
        metrics = {'S/loss_segm': 0,
                   'iou': [],
                   'accuracy': [],
                   'G/loss': 0,
                   'G/loss_fake': 0,
                   'G/loss_cycle': 0,
                   'G/loss_cls': 0,
                   'G/loss_id': 0,
                   'Ge/loss_fdom': 0,
                   'Ge/loss_frf': 0,
                   'Ge/loss_ffeat': 0}
        cm = torch.zeros(2, 2).float().cuda()
        i = 0
        with torch.no_grad():
            while True:

                # =================================================================================== #
                #                                 1. Preprocessing                                    #
                # =================================================================================== #
                # Fetch real images and labels.
                try:
                    x_real, c_org, gt_real = next(mix_iter)
                except:
                    print("mix_iter shouldn't have raised this exception in validation")

                # Fetch source images and masks
                try:
                    x, gt = next(source_iter)
                    if x.size(0) < self.batch_size:
                        raise Exception
                    x_source, gt_source = x, gt
                except:
                    break

                # Fetch target images and masks
                try:
                    x, gt = next(target_iter)
                    if x.size(0) < self.batch_size:
                        raise Exception
                    x_target, gt_target = x, gt
                except:
                    break

                x_real = x_real.to(self.device)  # Input images.
                x_source = x_source.to(self.device)
                x_target = x_target.to(self.device)
                gt_source = gt_source.to(self.device)
                gt_target = gt_target.to(self.device)
                gt_real = gt_real.to(self.device)
                c_org = c_org.to(self.device)  # Original domain labels.

                # =================================================================================== #
                #                               4. Generator                                #
                # =================================================================================== #
                _, s_source, loss_log = self.G_losses(x_real, c_org, gt_real, x_source, gt_source, x_target, gt_target)

                cm = update_cm(cm, s_source, gt_source)

                # =================================================================================== #
                #                                 5. Miscellaneous                                    #
                # =================================================================================== #
                for k in loss_log:
                    metrics[k] += loss_log[k]
                i += 1
        metrics = compute_metrics(cm, metrics)
        metrics['G/loss_es'] = metrics['G/loss'] - self.lambda_g * metrics['G/loss_fake'] - self.lambda_fdom * metrics['Ge/loss_fdom'] - self.lambda_frf * metrics['Ge/loss_frf']
        pattern = re.compile("(?!iou|accuracy).*")
        metrics.update({k: v / i for k, v in metrics.items() if pattern.match(k)})

        # Log metrics
        self.logger.scalar_summary(mode='val', epoch=epoch, **metrics)

        # Log visualization
        x_target = x_target.to(self.device)
        self.tb_images(x_target, torch.zeros(x_target.size(0), 1).to(self.device), epoch, 'val')

        return metrics['G/loss_es']

    def test(self, which_dataset, condition_target):
        """Test segmentation."""

        if which_dataset == 'source':
            loader = self.source_loader
        else:
            loader = self.target_loader

        # Load the trained generator.
        self.restore_model(self.G, 'G', self.log_dir)
        self.restore_model(self.S, 'S', self.log_dir)

        # Load the trained generator.
        self.G.eval()
        self.S.eval()

        # Evaluate segmentation
        metrics = {'loss_segm': 0,
                   'iou': 0,
                   'accuracy': 0}
        cm = torch.zeros(2, 2).float().cuda()

        with torch.no_grad():
            for i, (x, gt) in enumerate(loader):
                # Prepare input images and target masks.
                x = x.to(self.device)
                gt = gt.to(self.device)

                # Segment images
                condition = 1. if condition_target == 'source' else 0.
                _, h = self.G(x, condition * torch.ones(x.size(0), 1).to(self.device))

                s = self.S(h)
                metrics['loss_segm'] += self.segm_criterion(s, gt).item()

                # Update metrics
                cm = update_cm(cm, s, gt)

        metrics['loss_segm'] /= len(loader)

        # Compute metrics
        metrics = compute_metrics(cm, metrics)

        print_metrics('TEST ' + which_dataset + ': ', metrics)
