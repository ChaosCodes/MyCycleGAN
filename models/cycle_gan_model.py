import torch
import itertools
import torchvision.models
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # TODO
        # specify the training losses you want to print out.
        #self.loss_names = ['D_C', 'D_B', 'G', 'G_A_B', 'G_A_C', 'G_B_B', 'G_B_C', 'G_C_B', 'G_C_C', 'cycle_A',
        #                   'cycle_B', 'cycle_C']
        self.loss_names = ['D_C', 'D_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_A_BC', 'rev_A']
        visual_names_B = ['real_B', 'fake_B_BC', 'rev_B']
        visual_names_C = ['real_C', 'fake_C_BC', 'rev_C']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B + visual_names_C  # combine visualizations for A and B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_BC', 'D_B', 'D_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_BC']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_BC (G), D_A (D_Y), D_B (D_X)
        self.netG_BC = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_BC_rev = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                             not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.vgg = torchvision.models.vgg19_bn(pretrained=True)\
                       .to(torch.device(f'cuda:{self.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'))
            self.vgg.eval()

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_BC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_C.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A_BC = self.netG_BC(self.real_A)  # G_BC(A)
        self.fake_B_BC = self.netG_BC(self.real_B)  # G_BC(B)
        self.fake_C_BC = self.netG_BC(self.real_C)  # G_BC(C)

        self.rev_A = self.netG_BC_rev(self.real_A)  # G_BC(A)
        self.rev_B = self.netG_BC_rev(self.real_B)  # G_BC(B)
        self.rev_C = self.netG_BC_rev(self.real_C)  # G_BC(C)

    def backward_D_basic(self, netD, real, fake_A, fake_B, fake_C):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake_A.detach())
        loss_D_fake_A = self.criterionGAN(pred_fake, False)
        # Fake
        pred_fake = netD(fake_B.detach())
        loss_D_fake_B = self.criterionGAN(pred_fake, False)
        # Fake
        pred_fake = netD(fake_C.detach())
        loss_D_fake_C = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + (loss_D_fake_A + loss_D_fake_B + loss_D_fake_C) * 0.6) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_C(self):
        """Calculate GAN loss for discriminator D_C"""
        fake_A = self.fake_B_pool.query(self.fake_A_BC)
        fake_B = self.fake_B_pool.query(self.fake_B_BC)
        fake_C = self.fake_B_pool.query(self.fake_C_BC)
        self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_C, fake_A, fake_B, fake_C)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_B_pool.query(self.fake_A_BC)
        fake_B = self.fake_B_pool.query(self.fake_B_BC)
        fake_C = self.fake_B_pool.query(self.fake_C_BC)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_A, fake_B, fake_C)

    def backward_G(self):
        """Calculate the loss for generators G_BC"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rev_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rev_B, self.real_B) * lambda_B
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_C = self.criterionCycle(self.rev_C, self.real_C) * lambda_B

        vgg_loss_A = self.criterionIdt(self.vgg(self.real_A), self.vgg(self.fake_A_BC))
        vgg_loss_B = self.criterionIdt(self.vgg(self.real_B), self.vgg(self.fake_B_BC))
        vgg_loss_C = self.criterionIdt(self.vgg(self.real_C), self.vgg(self.fake_C_BC))
        self.loss_vgg = vgg_loss_A + vgg_loss_B + vgg_loss_C
        # GAN loss D_B(G_BC(A))
        self.loss_G_A_B = self.criterionGAN(self.netD_B(self.fake_A_BC), True)
        self.loss_G_A_C = self.criterionGAN(self.netD_C(self.fake_A_BC), True)
        # GAN loss D_B(G_BC(B))
        self.loss_G_B_B = self.criterionGAN(self.netD_B(self.fake_B_BC), True)
        self.loss_G_B_C = self.criterionGAN(self.netD_C(self.fake_B_BC), True)
        # GAN loss D_B(G_BC(B))
        self.loss_G_C_B = self.criterionGAN(self.netD_B(self.fake_C_BC), True)
        self.loss_G_C_C = self.criterionGAN(self.netD_C(self.fake_C_BC), True)
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A_B + self.loss_G_A_C + self.loss_G_B_B + self.loss_G_B_C + self.loss_G_C_B +\
                      self.loss_G_C_C + self.loss_cycle_A + self.loss_cycle_B + self.loss_cycle_C + self.loss_vgg * lambda_idt
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_BC

        #self.set_requires_grad([self.netD_B, self.netD_C], False)  # Ds require no gradients when optimizing Gs
        #self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        #self.backward_G()             # calculate gradients for G_A and G_B
        #self.optimizer_G.step()       # update G_A and G_B's weights
        # D_B and D_C
        self.set_requires_grad([self.netD_B, self.netD_C], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        self.backward_D_B()      # calculate gradients for D_B
        self.backward_D_C()      # calculate graidents for D_C
        self.optimizer_D.step()  # update D_B and D_C's weights
