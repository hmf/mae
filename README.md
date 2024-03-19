## Masked Autoencoders: A PyTorch Implementation

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

Next we tried to get the ImageNet dataset from HuggingFace. For a full account of what was tried, see [these notes](./imagenet1k.md).

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


