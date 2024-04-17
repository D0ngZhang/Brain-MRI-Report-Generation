import torch.distributed as dist
from blip.blip import *
import nibabel as nib
from dataset import data_set
import torch
import argparse
from torch.utils.data import DataLoader
import logger
import metrics

from option import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()

if Narval:
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    parser.add_argument("--dist", type=bool, default=1, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dataset_path", type=str, default="/lustre07/scratch/uanazodo/dominic/ERG/data/high-field",
                        help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    torch.distributed.init_process_group(backend="nccl")
elif Mist:
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    parser.add_argument("--dist", type=bool, default=1, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--img_path", type=str,
                        default="/gpfs/fs0/scratch/u/uanazodo/uanazodo/dominic/ERG/datasets/high-field/301-preprocessed",
                        help="path of the dataset")
    parser.add_argument("--report_path", type=str,
                        default="/gpfs/fs0/scratch/u/uanazodo/uanazodo/dominic/ERG/datasets/high-field/reports",
                        help="path of the report")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    torch.distributed.init_process_group(backend="nccl")
elif Sockeye:
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    parser.add_argument("--dist", type=bool, default=1, help="distribute or regular")
    parser.add_argument("--local_rank", default=os.environ['LOCAL_RANK'], type=int)
    parser.add_argument("--hq_dataset_path", type=str, default="/project/st-zjanew-1/li/mri/merge_segmented/merge_segmented", help="path of the dataset")
    parser.add_argument("--lq_dataset_path", type=str, default="/project/st-zjanew-1/li/mri/low-field", help="path of the low-quality dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    torch.distributed.init_process_group(backend="nccl")
elif Colab:
    parser.add_argument("--dist", type=bool, default=0, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--hq_dataset_path", type=str, default=r"/content/test_mri", help="path of the dataset")
    parser.add_argument("--lq_dataset_path", type=str, default=r"/content/test_mri_low", help="path of the low-quality dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
elif Local_Li:
    parser.add_argument("--dist", type=bool, default=0, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--hq_dataset_path", type=str, default=r"E:\merge_segmented\merge_segmented", help="path of the dataset")
    parser.add_argument("--lq_dataset_path", type=str, default=r"E:\low-field", help="path of the low-quality dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
elif Local_Dong:
    parser.add_argument("--img_path", type=str,
                        default="E:/data/301", help="path of the dataset")
    parser.add_argument("--report_path", type=str, default="E:/reports", help="path of the diagnosis reports")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
else:
    raise ValueError("Invalid option")


parser.add_argument("--mode", type=str, default="train", help="training or validation or testing")
parser.add_argument("--max_length", type=int, default=200, help="maximum length of the tokens")
parser.add_argument("--prompt", type=str, default="A brain MRI.")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=800, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--decay", type=float, default=0.05, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")

parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--width", type=int, default=192, help="image width")
parser.add_argument("--height", type=int, default=192, help="image height")
parser.add_argument("--depth", type=int, default=16, help="image depth / slice number")

parser.add_argument("--patch_size", type=int, default=16, help="the side length of patch")
parser.add_argument("--patch_depth", type=int, default=2, help="the depth of patch")
parser.add_argument("--transformer_depth", type=int, default=12, help="number of transformer layers")
parser.add_argument("--embed_dim", type=int, default=256, help="bert dim")

parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=200, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=0.02, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=0.04, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)


torch.autograd.set_detect_anomaly(True)
world_size = torch.cuda.device_count()
if world_size > 1:
    multi_gpu = True
else:
    multi_gpu = False

print("GPU num: ", world_size)
print("torch distribution", torch.distributed.is_available())

if Narval or Mist or Sockeye:
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)
else:
    device = torch.device("cuda")
print("setting devices")

model = blip_pretrain(opt).to(device)

# if opt.epoch != 0:
#     model.load_state_dict(torch.load("saved_models/model_%d.pth" % opt.epoch)['model'])
#     print("Pretrained models loaded")

if multi_gpu:
    if opt.dist:
        model = DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                                          broadcast_buffers=False)
        print("Distributed parallel training")
    else:
        model = DataParallel(model)
        print("Data parallel training")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=opt.decay)

total_dataset = data_set(opt)
data_size = len(total_dataset)
validation_size = round(0.05 * data_size)

train_dataset, test_dataset = torch.utils.data.random_split(total_dataset,[data_size - validation_size, validation_size])

if Narval or Mist or Sockeye:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.n_cpu, pin_memory=True, sampler=train_sampler)
else:
    dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu, pin_memory=True)
print("data prepared")
log_train = logger.Train_Logger(os.getcwd(), "train_log")
log_val = logger.Train_Logger(os.getcwd(), "val_log")
log_test = logger.Train_Logger(os.getcwd(), "test_log")
print("Start training")

os.makedirs("images/training", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)
os.makedirs("images/testing", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)



for epoch in range(0, 100):

    epoch_training_loss = metrics.LossAverage()
    epoch_mlm_loss = metrics.LossAverage()
    epoch_ita_loss = metrics.LossAverage()
    epoch_itm_loss = metrics.LossAverage()
    epoch_lm_loss = metrics.LossAverage()

    for i, imgs in enumerate(dataloader):
        imgs_t1 = imgs["t1"]  # t1
        imgs_t2 = imgs["t2"]  # t2
        reports = imgs["report"]
        # reports_tokens = imgs["report_tokens"]
        # text_ids = imgs["report-id"]  # c1 style
        # text_masks = imgs["report-mask"]

        imgs_t1, imgs_t2 = imgs_t1.float().to(device), imgs_t2.float().to(device)

        alpha = 0.4 * min(1, (epoch * len(dataloader) + i) / (2 * len(dataloader)))
        imgs = torch.cat((imgs_t1, imgs_t2), 1)
          # ------------------
          #  Train Generators
          # ------------------
        # loss_mlm, loss_ita, loss_itm, loss_lm
        loss_mlm, loss_ita, loss_itm, loss_lm = model(imgs, reports, alpha)
        loss = loss_mlm + loss_ita + loss_itm + loss_lm

        loss_mlm_detached = loss_mlm.detach().clone()
        loss_ita_detached = loss_ita.detach().clone()
        loss_itm_detached = loss_itm.detach().clone()
        loss_lm_detached = loss_lm.detach().clone()
        loss_detached = loss.detach().clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

        epoch_training_loss.update(loss_detached.item(), imgs_t1.shape[0])
        epoch_mlm_loss.update(loss_mlm_detached.item(), imgs_t1.shape[0])
        epoch_ita_loss.update(loss_ita_detached.item(), imgs_t1.shape[0])
        epoch_itm_loss.update(loss_itm_detached.item(), imgs_t1.shape[0])
        epoch_lm_loss.update(loss_lm_detached.item(), imgs_t1.shape[0])

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Total loss: %f, mlm loss: %f, ita loss: %f, itm loss: %f, lm loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_detached.item(), loss_mlm_detached.item(), loss_ita_detached.item(), loss_itm_detached.item(), loss_lm_detached.item()
            )
        )

    if multi_gpu:
        state_encoder = {'epoch': epoch, 'model': model.module.state_dict()}
    else:
        state_encoder = {'epoch': epoch, 'model': model.state_dict()}
    torch.save(state_encoder, "saved_models/model_%d.pth" % epoch)


    temp_log_train = OrderedDict({'Total loss': epoch_training_loss.avg, 'MLM loss': epoch_mlm_loss.avg, 'ITA loss': epoch_ita_loss.avg,
                                  'ITM loss': epoch_itm_loss.avg, 'LM loss': epoch_lm_loss.avg,})
    log_train.update(epoch, temp_log_train)

    os.makedirs("images/validation/%d" % epoch, exist_ok=True)
    if Narval or Mist or Sockeye:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.n_cpu, pin_memory=True, sampler=test_sampler)
    else:
        test_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.n_cpu, pin_memory=True)
    print("Start validation")
    for i, imgs in enumerate(test_dataloader):
        imgs_t1 = imgs["t1"]  # t1
        imgs_t2 = imgs["t2"]  # t2
        reports = imgs["report"]
        affines = imgs["affine"]

        imgs_t1, imgs_t2 = imgs_t1.float().to(device), imgs_t2.float().to(device)
        # text_ids, text_masks = text_ids.long().to(device), text_masks.long().to(device)

        imgs = torch.cat((imgs_t1, imgs_t2), 1)
        # ------------------
        #  Train Generators
        # ------------------

        if Narval or Mist or Sockeye:
            captions = model.module.generate(imgs, sample=False, num_beams=1, max_length=200, min_length=10, repetition_penalty=1.1)
        else:
            captions = model.generate(imgs, sample=False, num_beams=1, max_length=200, min_length=10,
                                      repetition_penalty=1.1)

        for j in range(imgs_t1.shape[0]):
            input = imgs_t1[j, 0, ...] * 0.5 + 0.5

            label = reports[j]
            prediction = captions[j]

            input = input.detach().cpu().numpy()


            affine = affines[j].numpy().astype('float64')

            input = input.transpose(1, 2, 0)

            input = nib.Nifti1Image(input, affine)

            nib.save(input, "images/validation/%d/%d_%d.nii" % (epoch, i, j))
            with open("images/validation/%d/%d_%d.txt" % (epoch, i, j), 'w', encoding='utf-8') as file:
                # 将"label"和"prediction"及其对应的值写入文件
                file.write(f'label: {label}\n')
                file.write(f'prediction: {prediction}\n')
