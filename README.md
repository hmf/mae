## Masked Autoencoders: A PyTorch Implementation

Original: https://github.com/facebookresearch/mae


<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.

* This repo is a modification on the [DeiT repo](https://github.com/facebookresearch/deit). Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Catalog

- [x] Visualization demo
- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-B</th>
<th valign="bottom">ViT-L</th>
<th valign="bottom">ViT-H</th>
<th valign="bottom">ViT-H<sub>448</sub></th>
<td valign="bottom" style="color:#C0C0C0">prev best</td>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1K (no external data)</td>
<td align="center">83.6</td>
<td align="center">85.9</td>
<td align="center">86.9</td>
<td align="center"><b>87.8</b></td>
<td align="center" style="color:#C0C0C0">87.1</td>
</tr>
<td colspan="5"><font size="1"><em>following are evaluation of the same model weights (fine-tuned in original ImageNet-1K):</em></font></td>
<tr>
</tr>
<tr><td align="left">ImageNet-Corruption (error rate) </td>
<td align="center">51.7</td>
<td align="center">41.8</td>
<td align="center"><b>33.8</b></td>
<td align="center">36.8</td>
<td align="center" style="color:#C0C0C0">42.5</td>
</tr>
<tr><td align="left">ImageNet-Adversarial</td>
<td align="center">35.9</td>
<td align="center">57.1</td>
<td align="center">68.2</td>
<td align="center"><b>76.7</b></td>
<td align="center" style="color:#C0C0C0">35.8</td>
</tr>
<tr><td align="left">ImageNet-Rendition</td>
<td align="center">48.3</td>
<td align="center">59.9</td>
<td align="center">64.4</td>
<td align="center"><b>66.5</b></td>
<td align="center" style="color:#C0C0C0">48.7</td>
</tr>
<tr><td align="left">ImageNet-Sketch</td>
<td align="center">34.5</td>
<td align="center">45.3</td>
<td align="center">49.6</td>
<td align="center"><b>50.9</b></td>
<td align="center" style="color:#C0C0C0">36.0</td>
</tr>
<td colspan="5"><font size="1"><em>following are transfer learning by fine-tuning the pre-trained MAE on the target dataset:</em></font></td>
</tr>
<tr><td align="left">iNaturalists 2017</td>
<td align="center">70.5</td>
<td align="center">75.7</td>
<td align="center">79.3</td>
<td align="center"><b>83.4</b></td>
<td align="center" style="color:#C0C0C0">75.4</td>
</tr>
<tr><td align="left">iNaturalists 2018</td>
<td align="center">75.4</td>
<td align="center">80.1</td>
<td align="center">83.0</td>
<td align="center"><b>86.8</b></td>
<td align="center" style="color:#C0C0C0">81.2</td>
</tr>
<tr><td align="left">iNaturalists 2019</td>
<td align="center">80.5</td>
<td align="center">83.4</td>
<td align="center">85.7</td>
<td align="center"><b>88.3</b></td>
<td align="center" style="color:#C0C0C0">84.1</td>
</tr>
<tr><td align="left">Places205</td>
<td align="center">63.9</td>
<td align="center">65.8</td>
<td align="center">65.9</td>
<td align="center"><b>66.8</b></td>
<td align="center" style="color:#C0C0C0">66.0</td>
</tr>
<tr><td align="left">Places365</td>
<td align="center">57.9</td>
<td align="center">59.4</td>
<td align="center">59.8</td>
<td align="center"><b>60.3</b></td>
<td align="center" style="color:#C0C0C0">58.0</td>
</tr>
</tbody></table>

### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


# Changes

## INESC Environment

<!--- cSpell:disable --->
```shell
:~$ cd Desktop/notepad/vpn/
:~$ sudo openvpn --config inesctec_202103.ovpn
:~$ ssh ubuntu@10.61.14.231
:~$ ssh -X ssh -X 10.61.4.52
```
<!--- cSpell:enable --->

# Avoid overriding CUDA installation

The default dev container installation configuration adds the UDa libraries and tools. The additional OS installation updates the packages. When this is done, some CUDA libraries are updated. The OS then has the incorrect libraries:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ apt list --installed | grep -i 12.2

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

dbus-user-session/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
dbus/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libcudnn8-dev/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libcudnn8/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libdbus-1-3/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libnode72/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed,automatic]
libnvjpeg-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
libnvjpeg-dev-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
nodejs-doc/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 all [installed,automatic]
nodejs/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed]
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 
```
<!--- cSpell:enable --->


We add the following commands by pinning the packages to avoid the changes in their version:

<!--- cSpell:disable --->
```
sudo apt-mark hold cuda-toolkit libcudnn8-dev libcudnn8
sudo apt-get upgrade -y
```
<!--- cSpell:enable --->


Check CUDA access:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/ijepa (test_1) $  python has_cuda.py 
has_cuda = True
device = cuda
n_gpu = 1
```
<!--- cSpell:enable --->

## Requirements

* timm==0.3.2- fix is needed to work with PyTorch 1.8.1+
* PIL
* Matplotlib
* pip3 install timm==0.4.5

## Code corrections/changes

Try with running the code. First download the model checkpoint. We see that we need the data paths set up. These are set set for `data`. 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ mkdir checkpoints
vscode ➜ /workspaces/mae (test_1) $ cd checkpoints/
vscode ➜ /workspaces/mae/checkpoints (test_1) $ wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth
--2024-02-16 11:06:03--  https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth
Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 18.154.41.12, 18.154.41.96, 18.154.41.57, ...
Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|18.154.41.12|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 346326087 (330M) [binary/octet-stream]
Saving to: ‘mae_finetuned_vit_base.pth’

mae_finetuned_vit_base.pth        100%[==========================================================>] 330.28M  29.3MB/s    in 12s     

2024-02-16 11:06:16 (27.5 MB/s) - ‘mae_finetuned_vit_base.pth’ saved [346326087/346326087]
vscode ➜ /workspaces/mae/checkpoints (test_1) $ cd ..


vscode ➜ /workspaces/mae (test_1) $ mkdir data
vscode ➜ /workspaces/mae (test_1) $ cd data/

```
<!--- cSpell:enable --->

To test the code, we need the `imagenet-1k` dataset. We need to sign-up and register at: http://image-net.org/download. The answer should have been sent via e-mail within 5 days of the request. Here is the recorded message:


> You have submitted a request at Fri Feb 16 03:29:35 2024. We are reviewing your request. When we approve your request, we will notify you by email. You should expect to hear from us in 5 work days.

No access was given. 

We need to update the code. Code originally written for PyTorch v1.x. Some references for the update:

1. [Tips and Tricks for Upgrading to PyTorch 2.0](https://towardsdatascience.com/tips-and-tricks-for-upgrading-to-pytorch-2-3127db1d1f3d)
  1. https://pytorch.org/docs/stable/generated/torch.compile.html
  1. [TorchDynamo](https://github.com/pytorch/torchdynamo/tree/0b8aaf340dad4777a080ef24bf09623f1aa6f3dd)
  1. [FC Graph](https://pytorch.org/docs/stable/fx.html)
  1. [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
  1. [Triton](https://github.com/openai/triton)
  1. Learn from
     1. [PyTorch documentation](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)
     1. [2022 PyTorch Conference](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)
     1. [TDS post](https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26)
  1. PyTorch Lightening
  1. "One of the nice things about PyTorch 2 is that it is fully backward compatible."
  1. Example of a ViT using [TIMM](https://pypi.org/project/timm/) and [Automatic mixed precision (AMP)](https://pytorch.org/docs/stable/amp.html)
     1. Use of AMP important for speedup (nearly 50% better)
  1. [Minifier](https://pytorch.org/functorch/stable/notebooks/minifier.html)


Here is the command used to used to train the model using the previously downloaded checkpoint and assuming the data is in the `./data` path:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ./data

```
<!--- cSpell:enable --->

Here are the errors we got for this test:

1. [ModuleNotFoundError: No module named 'torch._six'](https://github.com/microsoft/DeepSpeed/issues/2845)
   1. Convert from ```from torch._six import inf```
   1. to `from torch import inf`
1. `FileNotFoundError: [Errno 2] No such file or directory: './data/train'`
1. `FileNotFoundError: Couldn't find any class folder in ./data/train`

Next we tried to get the ImageNet dataset from HuggingFace. For a full account of what was tried, see [these notes](./imagenet1k.md). To summarize we had to:

1. Download [test_images.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/test_images.tar.gz?download=true) from the HuggingFace site
1. Download the training and validation data sets from the links in [this page](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4):
   1. wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
   1. wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
1. Split the training dataset into small data sets because we did not have enough space
1. Copy each split extract the tar files within and then execute the command in [this page](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to create the class folders;
1. Copy the validation data set and execute the bash script in the [same page](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to create the class folders
1. Whenever we extracted data, we had to remove the archives to free space. In the last training split, we had to copy the archive to a temporary folder and extracted the data from there;
1. Copied and extracted the test dataset last. Did not fit so 362 images were lost. Could not find the labels for these images.


These are the set of folders that are expected by the code:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/train
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/test
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/val
```

Make sure you can launch the dev container (Docker). The container must also bind to the share.  

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ ls /mnt/data02/data/cache/imagenet-1k/
test  train  val
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /mnt/data
```
<!--- cSpell:enable --->

The following erro occurs if the files are not split into class folders (training has 1300 per class, and we have 1000 classes):

<!--- cSpell:disable --->
```shell
Traceback (most recent call last):
  File "/workspaces/mae/main_finetune.py", line 356, in <module>
    main(args)
  File "/workspaces/mae/main_finetune.py", line 174, in main
    dataset_val = build_dataset(is_train=False, args=args)
  File "/workspaces/mae/util/datasets.py", line 24, in build_dataset
    dataset = datasets.ImageFolder(root, transform=transform)
  File "/home/vscode/.local/lib/python3.10/site-packages/torchvision/datasets/folder.py", line 309, in __init__
    super().__init__(
  File "/home/vscode/.local/lib/python3.10/site-packages/torchvision/datasets/folder.py", line 144, in __init__
    classes, class_to_idx = self.find_classes(self.root)
  File "/home/vscode/.local/lib/python3.10/site-packages/torchvision/datasets/folder.py", line 218, in find_classes
    return find_classes(directory)
  File "/home/vscode/.local/lib/python3.10/site-packages/torchvision/datasets/folder.py", line 42, in find_classes
    raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
FileNotFoundError: Couldn't find any class folder in /mnt/data/val.
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
[10:08:13.470830] Sampler_train = <torch.utils.data.distributed.DistributedSampler object at 0x7fe534a3e500>
Traceback (most recent call last):
  File "/workspaces/mae/main_finetune.py", line 356, in <module>
    main(args)
  File "/workspaces/mae/main_finetune.py", line 259, in main
    model.to(device)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1152, in to
    return self._apply(convert)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 802, in _apply
    module._apply(fn)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 802, in _apply
    module._apply(fn)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 825, in _apply
    param_applied = fn(param)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1150, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/cuda/__init__.py", line 302, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ nvidia-smi 
Failed to initialize NVML: Unknown Error
```
<!--- cSpell:enable --->

Required a rebuild of the Dev Container. 

# Fine-tuning 

<!--- cSpell:disable --->
```shell
scode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /mnt/data
Not using distributed mode
[10:32:32.623323] job dir: /workspaces/mae
[10:32:32.623400] Namespace(batch_size=16,
epochs=50,
accum_iter=1,
model='vit_base_patch16',
input_size=224,
drop_path=0.1,
clip_grad=None,
weight_decay=0.05,
lr=None,
blr=0.001,
layer_decay=0.75,
min_lr=1e-06,
warmup_epochs=5,
color_jitter=None,
aa='rand-m9-mstd0.5-inc1',
smoothing=0.1,
reprob=0.25,
remode='pixel',
recount=1,
resplit=False,
mixup=0,
cutmix=0,
cutmix_minmax=None,
mixup_prob=1.0,
mixup_switch_prob=0.5,
mixup_mode='batch',
finetune='',
global_pool=True,
data_path='/mnt/data',
nb_classes=1000,
output_dir='./output_dir',
log_dir='./output_dir',
device='cuda',
seed=0,
resume='checkpoints/mae_finetuned_vit_base.pth',
start_epoch=0,
eval=True,
dist_eval=False,
num_workers=10,
pin_mem=True,
world_size=1,
local_rank=-1,
dist_on_itp=False,
dist_url='env://',
distributed=False)
[10:32:35.529315] Dataset ImageFolder
    Number of datapoints: 1281167
    Root location: /mnt/data/train
    StandardTransform
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
               RandomHorizontalFlip(p=0.5)
               <timm.data.auto_augment.RandAugment object at 0x7f1aa230bf40>
               ToTensor()
               Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
               <timm.data.random_erasing.RandomErasing object at 0x7f1aa235a230>
           )
[10:32:35.659536] Dataset ImageFolder
    Number of datapoints: 50000
    Root location: /mnt/data/val
    StandardTransform
Transform: Compose(
               Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)
               CenterCrop(size=(224, 224))
               ToTensor()
               Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
           )
[10:32:35.659710] Sampler_train = <torch.utils.data.distributed.DistributedSampler object at 0x7f1aa2359cc0>
[10:32:37.122401] Model = VisionTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (blocks): ModuleList(
    (0): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
    (1-11): 11 x Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (pre_logits): Identity()
  (head): Linear(in_features=768, out_features=1000, bias=True)
  (fc_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
)
[10:32:37.122475] number of params (M): 86.57
[10:32:37.122493] base lr: 1.00e-03
[10:32:37.122503] actual lr: 6.25e-05
[10:32:37.122512] accumulate grad iterations: 1
[10:32:37.122521] effective batch size: 16
[10:32:37.124830] criterion = LabelSmoothingCrossEntropy()
[10:32:37.299692] Resume checkpoint checkpoints/mae_finetuned_vit_base.pth
[10:32:38.093088] Test:  [   0/3125]  eta: 0:39:51  loss: 0.5996 (0.5996)  acc1: 93.7500 (93.7500)  acc5: 93.7500 (93.7500)  time: 0.7653  data: 0.3154  max mem: 593
[10:32:38.175881] Test:  [  10/3125]  eta: 0:04:00  loss: 0.3770 (0.4417)  acc1: 93.7500 (92.0455)  acc5: 100.0000 (97.7273)  time: 0.0770  data: 0.0288  max mem: 593
[10:32:38.264821] Test:  [  20/3125]  eta: 0:02:18  loss: 0.3465 (0.4567)  acc1: 93.7500 (91.6667)  acc5: 100.0000 (97.9167)  time: 0.0085  data: 0.0001  max mem: 593
[10:32:38.354977] Test:  [  30/3125]  eta: 0:01:42  loss: 0.3863 (0.4657)  acc1: 87.5000 (90.5242)  acc5: 100.0000 (97.9839)  time: 0.0089  data: 0.0001  max mem: 593
[10:32:38.444000] Test:  [  40/3125]  eta: 0:01:23  loss: 0.1875 (0.3969)  acc1: 93.7500 (92.3780)  acc5: 100.0000 (98.4756)  time: 0.0089  data: 0.0001  max mem: 593
[10:32:38.534095] Test:  [  50/3125]  eta: 0:01:12  loss: 0.1508 (0.3756)  acc1: 100.0000 (93.1373)  acc5: 100.0000 (98.6520)  time: 0.0089  data: 0.0001  max mem: 593
...
[10:33:09.590530] Test:  [3080/3125]  eta: 0:00:00  loss: 0.3019 (0.7322)  acc1: 93.7500 (83.7147)  acc5: 100.0000 (96.5210)  time: 0.0112  data: 0.0019  max mem: 593
[10:33:09.675610] Test:  [3090/3125]  eta: 0:00:00  loss: 0.1472 (0.7313)  acc1: 100.0000 (83.7128)  acc5: 100.0000 (96.5302)  time: 0.0109  data: 0.0019  max mem: 593
[10:33:09.768789] Test:  [3100/3125]  eta: 0:00:00  loss: 0.1758 (0.7296)  acc1: 100.0000 (83.7613)  acc5: 100.0000 (96.5414)  time: 0.0088  data: 0.0001  max mem: 593
[10:33:09.891028] Test:  [3110/3125]  eta: 0:00:00  loss: 0.1780 (0.7287)  acc1: 100.0000 (83.7874)  acc5: 100.0000 (96.5405)  time: 0.0107  data: 0.0015  max mem: 593
[10:33:09.983262] Test:  [3120/3125]  eta: 0:00:00  loss: 0.3616 (0.7284)  acc1: 93.7500 (83.7772)  acc5: 100.0000 (96.5476)  time: 0.0107  data: 0.0017  max mem: 593
[10:33:10.019760] Test:  [3124/3125]  eta: 0:00:00  loss: 0.4602 (0.7298)  acc1: 87.5000 (83.7420)  acc5: 100.0000 (96.5380)  time: 0.0088  data: 0.0003  max mem: 593
[10:33:10.066181] Test: Total time: 0:00:32 (0.0105 s / it)
[10:33:10.066259] * Acc@1 83.742 Acc@5 96.538 loss 0.730
[10:33:10.066543] Accuracy of the network on the 50000 test images: 83.7%

real    0m39.679s
user    5m2.657s
sys     0m44.833s
vscode ➜ /workspaces/mae (test_1) $ 
```
<!--- cSpell:enable --->

Results in line with the ViT-B original results (Acc@1 83.664 Acc@5 96.530 loss 0.731) (see [FINETUNE.md](./FINETUNE.md)). We now increase the batch size from `16` to `32`: 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 32 --data_path /mnt/data
```
<!--- cSpell:enable --->

we get:

<!--- cSpell:disable --->
```shell
[08:48:18.466517] Test:  [1562/1563]  eta: 0:00:00  loss: 0.3426 (0.7301)  acc1: 93.7500 (83.7400)  acc5: 100.0000 (96.5360)  time: 0.0307  data: 0.0086  max mem: 687
[08:48:18.517925] Test: Total time: 0:00:31 (0.0202 s / it)
[08:48:18.518002] * Acc@1 83.740 Acc@5 96.536 loss 0.730
[08:48:18.518289] Accuracy of the network on the 50000 test images: 83.7%

real    0m38.950s
user    4m49.123s
sys     0m49.274s
```
<!--- cSpell:enable --->

Used 1351MiB of 46068MiB GPU memory. 

We now increase the batch size from `32` to `64`: 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 64 --data_path /mnt/data
```
<!--- cSpell:enable --->

We get:

<!--- cSpell:disable --->
```shell
[08:55:14.947099] Test: Total time: 0:00:32 (0.0413 s / it)
[08:55:14.947231] * Acc@1 83.740 Acc@5 96.538 loss 0.731
[08:55:14.947445] Accuracy of the network on the 50000 test images: 83.7%

real    0m39.349s
user    4m43.379s
sys     0m52.450s
```
<!--- cSpell:enable --->


Used 2159MiB of 46068MiB GPU memory. 

We now increase the batch size from `32` to `64`: 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 128 --data_path /mnt/data
```
<!--- cSpell:enable --->

We get:

<!--- cSpell:disable --->
```shell
[09:00:25.394188] Test: Total time: 0:00:33 (0.0868 s / it)
[09:00:25.394260] * Acc@1 83.740 Acc@5 96.538 loss 0.731
[09:00:25.394572] Accuracy of the network on the 50000 test images: 83.7%

real    0m41.071s
user    4m35.950s
sys     0m53.903s
```
<!--- cSpell:enable --->


The **effective batch size** is 32 (batch_size per gpu) * 4 (nodes) * 8 (gpus per node) = 1024. Training time is ~7h11m in 32 V100 GPUs. We now experiment with the effective batch size on a single GPU. 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 1024 --data_path /mnt/data
```
<!--- cSpell:enable --->

We get: 

<!--- cSpell:disable --->
```shell
[10:52:41.944991] Test: Total time: 0:00:45 (0.9267 s / it)
[10:52:41.945139] * Acc@1 83.740 Acc@5 96.538 loss 0.729
[10:52:41.945375] Accuracy of the network on the 50000 test images: 83.7%

real    0m54.010s
user    4m30.717s
sys     1m3.246s
```
<!--- cSpell:enable --->


Used 2159MiB of 46068MiB GPU memory. 

The runs above are very consistent. This is to be expected because we are fine tuning on the same data as the training data. We now experiment on the large and huge models on a single GPU to see how much memory is required. Here are the sizes of the pickled models:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae/checkpoints (test_1) $ ls -lh
total 3.9G
-rw-r--r-- 1 vscode vscode 331M Dec 29  2021 mae_finetuned_vit_base.pth
-rw-r--r-- 1 vscode vscode 2.4G Dec 29  2021 mae_finetuned_vit_huge.pth
-rw-r--r-- 1 vscode vscode 1.2G Dec 29  2021 mae_finetuned_vit_large.pth
```
<!--- cSpell:enable --->

### Large model

Download the large model: 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ cd checkpoints
vscode ➜ /workspaces/mae/checkpoints (test_1) $ wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth
--2024-04-17 10:55:24--  https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth
Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 18.154.41.8, 18.154.41.12, 18.154.41.96, ...
Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|18.154.41.8|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1217415191 (1.1G) [binary/octet-stream]
Saving to: ‘mae_finetuned_vit_large.pth’

mae_finetuned_vit_large.pth       100%[==========================================================>]   1.13G  4.86MB/s    in 2m 29s  

2024-04-17 10:57:54 (7.78 MB/s) - ‘mae_finetuned_vit_large.pth’ saved [1217415191/1217415191]

vscode ➜ /workspaces/mae/checkpoints (test_1) $ 
```
<!--- cSpell:enable --->

Fine tune the large model:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_large.pth --model vit_large_patch16 --batch_size 1024 --data_path /mnt/data
```
<!--- cSpell:enable --->

The **effective batch size** is 32 (batch_size per gpu) * 4 (nodes) * 8 (gpus per node) = 1024. Training time is ~8h52m in 32 V100 GPUs. TODO: increase batch size to 32 * 32 * 8 = 8192.

We get:

<!--- cSpell:disable --->
```shell
[11:14:17.064829] Test: Total time: 0:01:51 (2.2821 s / it)
[11:14:17.064972] * Acc@1 85.962 Acc@5 97.560 loss 0.645
[11:14:17.065211] Accuracy of the network on the 50000 test images: 86.0%

real    2m4.937s
user    4m50.669s
sys     2m6.140s
```
<!--- cSpell:enable --->

Used 12999MiB of 46068MiB GPU memory.

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_large.pth --model vit_large_patch16 --batch_size 8192 --data_path /mnt/data
```
<!--- cSpell:enable --->

Fails:

<!--- cSpell:disable --->
```shell
[11:53:57.208794] Resume checkpoint checkpoints/mae_finetuned_vit_large.pth
Killed

real    1m28.367s
user    0m29.820s
sys     0m25.562s
```
<!--- cSpell:enable --->

Batch size 1024 used 12999MiB of 46068MiB GPU memory. (2m4.937s)
Batch size 2048 used 24851MiB of 46068MiB GPU memory. (2m33.231s)
Batch size 4096 used 42069MiB of 46068MiB GPU memory. (2m36.484s)
Batch size 5120 used ? MiB of 46068MiB GPU memory. (Out of memory)


### Huge model

Download the large model: 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ cd checkpoints
vscode ➜ /workspaces/mae/checkpoints (test_1) $ wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth
--2024-04-17 10:59:24--  https://dt/
Resolving dt (dt)... failed: Name or service not known.
wget: unable to resolve host address ‘dt’
--2024-04-17 10:59:24--  https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth
Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 18.154.41.12, 18.154.41.57, 18.154.41.96, ...
Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|18.154.41.12|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2528327351 (2.4G) [binary/octet-stream]
Saving to: ‘mae_finetuned_vit_huge.pth’

mae_finetuned_vit_huge.pth        100%[==========================================================>]   2.35G  29.1MB/s    in 1m 40s  

2024-04-17 11:01:05 (24.2 MB/s) - ‘mae_finetuned_vit_huge.pth’ saved [2528327351/2528327351]

FINISHED --2024-04-17 11:01:05--
Total wall clock time: 1m 41s
Downloaded: 1 files, 2.4G in 1m 40s (24.2 MB/s)
```
<!--- cSpell:enable --->

Fine tune the huge model:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_huge.pth --model vit_huge_patch14 --batch_size 1024 --data_path /mnt/data
```
<!--- cSpell:enable --->

The **effective batch size** is 32 (batch_size per gpu) * 4 (nodes) * 8 (gpus per node) = 1024. Training time is ~13h9m in 64 V100 GPU. TODO: increase batch size to 32 * 64 * 8 = 16384.

We get:

<!--- cSpell:disable --->
```shell
[11:32:49.066282] Test: Total time: 0:04:30 (5.5208 s / it)
[11:32:49.066401] * Acc@1 86.884 Acc@5 98.076 loss 0.584
[11:32:49.066594] Accuracy of the network on the 50000 test images: 86.9%

real    4m50.013s
user    5m24.819s
sys     4m31.485s
```
<!--- cSpell:enable --->


Used 21261MiB of 46068MiB GPU memory.

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ time python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_huge.pth --model vit_huge_patch14 --batch_size 8192 --data_path /mnt/data
```
<!--- cSpell:enable --->

Fine tuning fails:

<!--- cSpell:disable --->
```shell
[11:50:05.374859] criterion = LabelSmoothingCrossEntropy()
[11:50:07.956181] Resume checkpoint checkpoints/mae_finetuned_vit_huge.pth
Killed

real    1m51.307s
user    0m49.190s
sys     0m30.752s
```
<!--- cSpell:enable --->


Batch size 1024 used 21261MiB of 46068MiB GPU memory. (4m50.013s)
Batch size 2048 used 39427MiB of 46068MiB GPU memory. (4m58.468s)
Batch size 4096 used ? MiB of 46068MiB GPU memory. (Out of memory)
Batch size 3072 used ? MiB of 46068MiB GPU memory. (Out of memory)

# Pre-Training

<!--- cSpell:disable --->
```shell
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}

python submitit_pretrain.py \
    --job_dir ./job \
    --nodes 1 \
    --ngpus 1 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /mnt/data

```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
Traceback (most recent call last):
  File "/workspaces/mae/submitit_pretrain.py", line 15, in <module>
    import main_pretrain as trainer
  File "/workspaces/mae/main_pretrain.py", line 27, in <module>
    assert timm.__version__ == "0.3.2"  # version check
AssertionError
```
<!--- cSpell:enable --->

`main_pretrain.py`

<!--- cSpell:disable --->
```python
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
```
<!--- cSpell:enable --->



<!--- cSpell:disable --->
```shell
Traceback (most recent call last):
  File "/workspaces/mae/submitit_pretrain.py", line 131, in <module>
    main()
  File "/workspaces/mae/submitit_pretrain.py", line 120, in main
    args.dist_url = get_init_file().as_uri()
  File "/workspaces/mae/submitit_pretrain.py", line 44, in get_init_file
    os.makedirs(str(get_shared_folder()), exist_ok=True)
  File "/workspaces/mae/submitit_pretrain.py", line 39, in get_shared_folder
    raise RuntimeError("No shared folder available")
RuntimeError: No shared folder available
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ echo $USER
vscode
vscode ➜ /workspaces/mae (test_1) $ mkdir checkpoint
vscode ➜ /workspaces/mae (test_1) $ mkdir checkpoint/vscode
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```python
def get_shared_folder() -> Path:
    user = os.getenv("USER")
    # HF: make this local 
    checkpoint = "./checkpoint/"
    if Path().is_dir():
        p = Path(f"{checkpoint}/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```shell
Traceback (most recent call last):
  File "/workspaces/mae/submitit_pretrain.py", line 133, in <module>
    main()
  File "/workspaces/mae/submitit_pretrain.py", line 122, in main
    args.dist_url = get_init_file().as_uri()
  File "/usr/lib/python3.10/pathlib.py", line 651, in as_uri
    raise ValueError("relative path can't be expressed as a file URI")
ValueError: relative path can't be expressed as a file URI
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```python
def get_shared_folder() -> Path:
    user = os.getenv("USER")
    # HF: make this local 
    checkpoint = "/workspaces/mae/checkpoint/"
    if Path().is_dir():
        p = Path(f"{checkpoint}/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ python submitit_pretrain.py     --job_dir ./job     --nodes 1     --ngpus 1     --use_volta32     --batch_size 64     --model mae_vit_large_patch16     --norm_pix_loss     --mask_ratio 0.75     --epochs 800     --warmup_epochs 40     --blr 1.5e-4 --weight_decay 0.05     --data_path /mnt/data
54226
```
<!--- cSpell:enable --->

The number output is the submit job ID. 

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ ps ax | grep -i python | grep -i mae
  54226 pts/1    S      0:00 /usr/local/python/current/bin/python -m submitit.local._local /workspaces/mae/job
  54227 pts/1    Rl     3:31 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54313 pts/1    Rl     0:40 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54327 pts/1    Sl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54341 pts/1    Sl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54342 pts/1    Sl     0:40 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54356 pts/1    Sl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54383 pts/1    Sl     0:40 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54397 pts/1    Sl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54398 pts/1    Sl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54399 pts/1    Rl     0:41 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
  54439 pts/1    Rl     0:40 /usr/local/python/current/bin/python -u -m submitit.core._submit /workspaces/mae/job
```
<!--- cSpell:enable --->

To remove these processes we need to remove the `submitit.local._local` process plus one other `submitit.core._submit`. 

Look at the logs in the `./job` path. From `$PID_O_log.out`, we have (PID is output from command above):

<!--- cSpell:disable --->
```shell
[13:17:26.603022] Sampler_train = <torch.utils.data.distributed.DistributedSampler object at 0x7f7dc4d02410>
submitit ERROR (2024-03-20 13:17:29,070) - Submitted job triggered an exception
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
File "/workspaces/mae/util/pos_embed.py", line 56, in get_1d_sincos_pos_embed_from_grid
    omega = np.arange(embed_dim // 2, dtype=np.float)
  File "/home/vscode/.local/lib/python3.10/site-packages/numpy/__init__.py", line 324, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'cfloat'?
```
<!--- cSpell:enable --->

https://github.com/jdb78/pytorch-forecasting/pull/1257
https://github.com/pytorch/pytorch/issues/91329


<!--- cSpell:disable --->
```shell
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # HF: omega = np.arange(embed_dim // 2, dtype=np.float)
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
```
<!--- cSpell:enable --->

After the changes, training starts. Each iteration was taking about 0.1748 seconds. 

TODO, remove `--use_volta32`?


<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
```
<!--- cSpell:enable --->




