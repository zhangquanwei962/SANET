import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
import os
import model
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from dataset import *
from utils import adjust_learning_rate





def main():
    # cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='content',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='style',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth')

    parser.add_argument('--save-dir', type=str, metavar='<dir>', default='./experiments', 
                        help='Directory to save trained models, default=./experiments')
    parser.add_argument('--log-dir', type=str, metavar='<dir>', default='./logs', 
                        help='Directory to save logs, default=./logs')

    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', '-maxiter' , type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--style_weight', type=float, default=3.0)
    parser.add_argument('--content_weight', type=float, default=100.0)
    parser.add_argument('--n_threads', type=int, default=5)
    parser.add_argument('--save_model_interval', type=int, default=1000)
    parser.add_argument('--start_iter', type=float, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--log-image-every', type=int, metavar='<int>', default=100, 
                        help='Period for loging generated images, non-positive for disabling, default=100')
    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # directory trained models
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    # directory for logs
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

  

    vgg = model.vgg
    decoder = model.decoder
    
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    network = model.Net(vgg, decoder, args.start_iter)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    print(f'# Minibatch-size: {args.batch_size}')
    # print(f'# epoch: {args.epoch}')
    print('')

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam([
                                {'params': network.decoder.parameters()},
                                {'params': network.transform.parameters()}], lr=args.lr)

    if(args.start_iter > 0):
        optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))
    
    writer = SummaryWriter(log_dir=str(log_dir))

    for i in tqdm(range(args.start_iter, args.max_iter)):
        adjust_learning_rate(optimizer, args.lr, args.lr_decay, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s, l_identity1, l_identity2, g = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # write logs

        writer.add_scalar('Loss/Loss', loss.item(), i + 1)
        writer.add_scalar('Loss/Loss_content', loss_c.item(), i + 1)
        writer.add_scalar('Loss/Loss_style', loss_s.item(), i + 1)
        if args.log_image_every > 0 and ((i + 1) % args.log_image_every == 0 or i == 0 or (i + 1) == args.max_iter):
            writer.add_image('Image/Content', denormalzation(content_images[0], device), i + 1)
            writer.add_image('Image/Style', denormalzation(style_images[0], device), i + 1)
            writer.add_image('Image/Generated', denormalzation(g[0], device), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                            i + 1))
            state_dict = network.transform.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                            i + 1))
            state_dict = optimizer.state_dict()
            torch.save(state_dict,
                    '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                            i + 1))
    writer.close()

if __name__ == '__main__':
    main()