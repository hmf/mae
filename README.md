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

We need to sign-up and register at: http://image-net.org/download. The answer is sent via e-mail within 5 days of the request. 

You have submitted a request at Fri Feb 16 03:29:35 2024. We are reviewing your request. When we approve your request, we will notify you by email. You should expect to hear from us in 5 work days.

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



<!--- cSpell:disable --->
imagenet_full_size-061417.tar.gz

python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}

python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ./data

```shell
vscode ➜ /workspaces/mae (test_1) $ python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ./data

```

<!--- cSpell:enable --->


1. [ModuleNotFoundError: No module named 'torch._six'](https://github.com/microsoft/DeepSpeed/issues/2845)
   1. Convert from ```from torch._six import inf```
   1. to `from torch import inf`
1. `FileNotFoundError: [Errno 2] No such file or directory: './data/train'`
1. `FileNotFoundError: Couldn't find any class folder in ./data/train`



1. [ImageNet sample images (minimal, not split)](https://github.com/EliSchwartz/imagenet-sample-images)
1. [HuggingFace ImageNet 1k](https://huggingface.co/datasets/imagenet-1k)
   1. [Files and versions](https://huggingface.co/datasets/imagenet-1k/tree/main)
      1. Entre data directory 
      1. Download data
         1. [test_images.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/test_images.tar.gz?download=true)
         1. [val_images.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/val_images.tar.gz?download=true)
         1. [train_images_0.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true)
         1. [train_images_1.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true)
         1. [train_images_2.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz?download=true)
         1. [train_images_3.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_3.tar.gz?download=true)
         1. [train_images_4.tar.gz](https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_4.tar.gz?download=true)


<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/train
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/test
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/val
```
<!--- cSpell:enable --->

Make sure you can launch the dev container (Docker). 

Go to local source with data:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cd /mnt/data02/data/src
ubuntu@cese-produtech3r:/mnt/data02/data/src$ 
ubuntu@cese-produtech3r:/mnt/data02/data/src$ mkdir imagenet-1k
```
<!--- cSpell:enable --->

Copy source data from the local to the remote node (these files are large so may take a while, may be vest todownload dirctly via curl or wget):

<!--- cSpell:disable --->
```shell
usr@node:~$ cd /mnt/ssd2/usr/datasets/computer_vision/imagenet-1k/
usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ 
usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp *.py ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
classes.py                                                                                                                                        100%   45KB 761.6KB/s   00:00    
imagenet-1k.py                                                                                                                                    100% 4721   130.4KB/s   00:00    
h
mf@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp gitattributes ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
gitattributes                                                                                                                                     100% 1566    52.0KB/s   00:00    
usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp README.md ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
README.md                                                                                                                                         100%   83KB   1.1MB/s   00:00    
usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp test*.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
test_images.tar.gz                                                                                                                                100%   13GB   9.0MB/s   23:55    

usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp val*.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
val_images.tar.gz                                                                                                                                 100% 6358MB   9.4MB/s   11:18    

usr@node:/mnt/ssd2/usr/datasets/computer_vision/imagenet-1k$ scp train_images_0.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
train_images_0.tar.gz                                                                                                                             100%   27GB   9.3MB/s   49:34    


```
<!--- cSpell:enable --->

<!-- https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/ -->


scp file.txt remote_username@10.10.0.2:/remote/directory
scp *.py ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp gitattributes ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp README.md ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp test*.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k

scp val*.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k

scp train_images_0.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp train_images_1.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp train_images_2.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp train_images_3.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k
scp train_images_4.tar.gz ubuntu@10.61.14.231:/mnt/data02/data/src/imagenet-1k


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ wget https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true
--2024-02-23 09:57:26--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true
Resolving huggingface.co (huggingface.co)... 54.192.95.26, 54.192.95.79, 54.192.95.70, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.26|:443... connected.
HTTP request sent, awaiting response... 401 Unauthorized

Username/Password Authentication Failed.
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
https://askubuntu.com/questions/29079/how-do-i-provide-a-username-and-password-to-wget
wget --user user --password pass url
wget --user user --ask-password url
wget http://user:password@host/path

wget --user usr --ask-password https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true
```
<!--- cSpell:enable --->

Space in front of command not to save in history

<!--- cSpell:disable --->
```shell
 wget --user usr --password "Sw9?yd=6qmCq$$ https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true
```
<!--- cSpell:enable --->

Space in front of command not to save in history

<!--- cSpell:disable --->
```shell
 wget --user=usr --password="Sw9?yd=6qmCq$$ https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true

Need single quots due to double quotes in password
 wget --user=usr --password='PASS' https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz?download=true
```
<!--- cSpell:enable --->

 https://serverfault.com/questions/150282/escape-a-in-the-password-parameter-of-wget


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true
--2024-02-23 11:24:13--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true
Resolving huggingface.co (huggingface.co)... 54.192.95.21, 54.192.95.79, 54.192.95.26, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.21|:443... connected.
HTTP request sent, awaiting response... 401 Unauthorized
Unknown authentication scheme.

Username/Password Authentication Failed.
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
https://discuss.huggingface.co/t/private-data-and-wget/35115/2
wget --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/GeneralAwareness/Various/resolve/main/file.zip
https://huggingface.co/settings/tokens

Space in front of command not to save in history

 wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true

 wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz?download=true

 wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_3.tar.gz?download=true

 wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_4.tar.gz?download=true 
```
<!--- cSpell:enable --->



 <!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true
--2024-02-23 11:36:14--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true
Resolving huggingface.co (huggingface.co)... 54.192.95.70, 54.192.95.21, 54.192.95.79, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.70|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/216cd7b2f345ab50cec3bea6090aa10d5ebf351bca3900627be4645e87e873fd?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_1.tar.gz%3B+filename%3D%22train_images_1.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708947375&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0NzM3NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzIxNmNkN2IyZjM0NWFiNTBjZWMzYmVhNjA5MGFhMTBkNWViZjM1MWJjYTM5MDA2MjdiZTQ2NDVlODdlODczZmQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Au0tLdPAfyMQWwoZaZa21u4DxnHfZy9DGuYaBhjXGhuNY0OeM0uWMSjjxnmAihMRw3PagX%7Epzyb%7Ey916mOGKrQFIc3oXk1f0WYhOmLWEru6yowyB9kH8d7tE1Ra-nnZ-eVE9kMvT1d3juX%7EsxHmXQ%7EhV6Oo8pJ99DUKjSIvtSIIlln1thXE38HSaPKWj6CeW5x0XSiR58A7Z0X-DW2mWwIAEwLDfybymEcN%7ExQ0l9IA5cW8FfcJhp%7E1aEQfwyjjFHyeNFsCXbZTjtzCSD2LZUA0QvwE4NLVSjzNoUgZ-Mdm0%7EhJc%7Eh8ei7XmHHAbq-qBbEJiRnPD7xCHQPC7dndjCA__&Key-Pair-Id=KVTP0A1DKRTAX [following]
--2024-02-23 11:36:15--  https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/216cd7b2f345ab50cec3bea6090aa10d5ebf351bca3900627be4645e87e873fd?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_1.tar.gz%3B+filename%3D%22train_images_1.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708947375&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0NzM3NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzIxNmNkN2IyZjM0NWFiNTBjZWMzYmVhNjA5MGFhMTBkNWViZjM1MWJjYTM5MDA2MjdiZTQ2NDVlODdlODczZmQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Au0tLdPAfyMQWwoZaZa21u4DxnHfZy9DGuYaBhjXGhuNY0OeM0uWMSjjxnmAihMRw3PagX%7Epzyb%7Ey916mOGKrQFIc3oXk1f0WYhOmLWEru6yowyB9kH8d7tE1Ra-nnZ-eVE9kMvT1d3juX%7EsxHmXQ%7EhV6Oo8pJ99DUKjSIvtSIIlln1thXE38HSaPKWj6CeW5x0XSiR58A7Z0X-DW2mWwIAEwLDfybymEcN%7ExQ0l9IA5cW8FfcJhp%7E1aEQfwyjjFHyeNFsCXbZTjtzCSD2LZUA0QvwE4NLVSjzNoUgZ-Mdm0%7EhJc%7Eh8ei7XmHHAbq-qBbEJiRnPD7xCHQPC7dndjCA__&Key-Pair-Id=KVTP0A1DKRTAX
Resolving cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)... 108.157.109.99, 108.157.109.59, 108.157.109.91, ...
Connecting to cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)|108.157.109.99|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 29261436971 (27G) [application/gzip]
Saving to: ‘train_images_1.tar.gz?download=true’

train_images_1.tar.gz?download=true      100%[================================================================================>]  27.25G  8.50MB/s    in 47m 10s 

2024-02-23 12:23:25 (9.86 MB/s) - ‘train_images_1.tar.gz?download=true’ saved [29261436971/29261436971]
```
<!--- cSpell:enable --->



 <!--- cSpell:disable --->
```shell
wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz


ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz
--2024-02-23 12:28:06--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz
Resolving huggingface.co (huggingface.co)... 54.192.95.70, 54.192.95.79, 54.192.95.21, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.70|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/0a7c68e057bd2c65f9ab4de3458b01c6538eb2fbcc0bce59e77836e82369622f?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_2.tar.gz%3B+filename%3D%22train_images_2.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708946867&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0Njg2N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzBhN2M2OGUwNTdiZDJjNjVmOWFiNGRlMzQ1OGIwMWM2NTM4ZWIyZmJjYzBiY2U1OWU3NzgzNmU4MjM2OTYyMmY%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Kji5s8eJ2o6AGUZJqPhlZWbth776k9DL4Li3C%7E9APxiwtsEtYZ%7EttohS40NOtEaVEc1XbJrLdZybytwdNE77FQ-OtKwvHiMzhWSKOLZhfu0QvuF8TXQtgOiweHpn2ywy0ATRfMl5DYUkGIvRFRfRc72mpjuUsy5SZcqr0djOpd01YT0Q69jOh7EfsWAi57b3QhHPRBdwOLNYyA0U5VsT3oPbwsMD06WtqwIU77ItHuGQNBLj1ULkz7uUhjmlYB2BlfOX1YKfLNsnsT43yg9jSKLGkG49tZdB1JUxgPD1JWU7BdYae2Ql0%7ELwDWMV2WxtYq0KRd7JkXxoiT009MqyVg__&Key-Pair-Id=KVTP0A1DKRTAX [following]
--2024-02-23 12:28:06--  https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/0a7c68e057bd2c65f9ab4de3458b01c6538eb2fbcc0bce59e77836e82369622f?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_2.tar.gz%3B+filename%3D%22train_images_2.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708946867&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0Njg2N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzBhN2M2OGUwNTdiZDJjNjVmOWFiNGRlMzQ1OGIwMWM2NTM4ZWIyZmJjYzBiY2U1OWU3NzgzNmU4MjM2OTYyMmY%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=Kji5s8eJ2o6AGUZJqPhlZWbth776k9DL4Li3C%7E9APxiwtsEtYZ%7EttohS40NOtEaVEc1XbJrLdZybytwdNE77FQ-OtKwvHiMzhWSKOLZhfu0QvuF8TXQtgOiweHpn2ywy0ATRfMl5DYUkGIvRFRfRc72mpjuUsy5SZcqr0djOpd01YT0Q69jOh7EfsWAi57b3QhHPRBdwOLNYyA0U5VsT3oPbwsMD06WtqwIU77ItHuGQNBLj1ULkz7uUhjmlYB2BlfOX1YKfLNsnsT43yg9jSKLGkG49tZdB1JUxgPD1JWU7BdYae2Ql0%7ELwDWMV2WxtYq0KRd7JkXxoiT009MqyVg__&Key-Pair-Id=KVTP0A1DKRTAX
Resolving cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)... 108.157.109.59, 108.157.109.99, 108.157.109.100, ...
Connecting to cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)|108.157.109.59|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 29036415239 (27G) [application/gzip]
Saving to: ‘train_images_2.tar.gz’

train_images_2.tar.gz                    100%[================================================================================>]  27.04G  43.8MB/s    in 10m 28s 

2024-02-23 12:38:35 (44.1 MB/s) - ‘train_images_2.tar.gz’ saved [29036415239/29036415239]
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_3.tar.gz
--2024-02-23 12:41:56--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_3.tar.gz
Resolving huggingface.co (huggingface.co)... 54.192.95.79, 54.192.95.70, 54.192.95.26, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.79|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/0cc2290c0f2d2be6a060f7edaef881e76558e5c3fda8ab30c0a3a78021ac5619?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_3.tar.gz%3B+filename%3D%22train_images_3.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708951316&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk1MTMxNn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzBjYzIyOTBjMGYyZDJiZTZhMDYwZjdlZGFlZjg4MWU3NjU1OGU1YzNmZGE4YWIzMGMwYTNhNzgwMjFhYzU2MTk%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=xapeGFm3s2D4bRZ4iasH91N8qLcRXQw4y1YOsJMM0t3q54VfNDziPPcTpboQ0VTKO%7E8%7E6iUH6Q%7EzAZEY7VLuNiT24pcbyPvQ9Ug0Zj2KJ3WcEbDg5JM4%7EMGhGD-SI8wxz7f%7EETxw9DFkGr5Biuz0l0v3HJzW0za4AgQXa4TAWMglWK32bO21qNvWK%7E63SwB2WuBPe25eQ8HS50RnO6xzscx2rbHA-%7EuxYzUR7AFTwm-pGn7icILjgqHQ4-HLTNKiUXBE3ydRLfhebU0E1YB7TWcmWWn7dOKTRdp9i37hzlchhMPTP0flHDW6Ncvw-nAvxj6oRk4KarmOKJa5cX5-NA__&Key-Pair-Id=KVTP0A1DKRTAX [following]
--2024-02-23 12:41:56--  https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/0cc2290c0f2d2be6a060f7edaef881e76558e5c3fda8ab30c0a3a78021ac5619?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_3.tar.gz%3B+filename%3D%22train_images_3.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708951316&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk1MTMxNn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5LzBjYzIyOTBjMGYyZDJiZTZhMDYwZjdlZGFlZjg4MWU3NjU1OGU1YzNmZGE4YWIzMGMwYTNhNzgwMjFhYzU2MTk%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=xapeGFm3s2D4bRZ4iasH91N8qLcRXQw4y1YOsJMM0t3q54VfNDziPPcTpboQ0VTKO%7E8%7E6iUH6Q%7EzAZEY7VLuNiT24pcbyPvQ9Ug0Zj2KJ3WcEbDg5JM4%7EMGhGD-SI8wxz7f%7EETxw9DFkGr5Biuz0l0v3HJzW0za4AgQXa4TAWMglWK32bO21qNvWK%7E63SwB2WuBPe25eQ8HS50RnO6xzscx2rbHA-%7EuxYzUR7AFTwm-pGn7icILjgqHQ4-HLTNKiUXBE3ydRLfhebU0E1YB7TWcmWWn7dOKTRdp9i37hzlchhMPTP0flHDW6Ncvw-nAvxj6oRk4KarmOKJa5cX5-NA__&Key-Pair-Id=KVTP0A1DKRTAX
Resolving cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)... 108.157.109.100, 108.157.109.59, 108.157.109.99, ...
Connecting to cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)|108.157.109.100|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 29227044756 (27G) [application/gzip]
Saving to: ‘train_images_3.tar.gz’

train_images_3.tar.gz                    100%[================================================================================>]  27.22G  34.1MB/s    in 12m 3s  

2024-02-23 12:54:00 (38.5 MB/s) - ‘train_images_3.tar.gz’ saved [29227044756/29227044756]
```
<!--- cSpell:enable --->



<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$   wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_4.tar.gz
--2024-02-23 12:55:13--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_4.tar.gz
Resolving huggingface.co (huggingface.co)... 54.192.95.70, 54.192.95.26, 54.192.95.21, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.70|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/bf6ab4b53f4b66adbff204ed7f4e36c9c704be6852b407c5147934f2a45c4595?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_4.tar.gz%3B+filename%3D%22train_images_4.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708949015&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0OTAxNX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5L2JmNmFiNGI1M2Y0YjY2YWRiZmYyMDRlZDdmNGUzNmM5YzcwNGJlNjg1MmI0MDdjNTE0NzkzNGYyYTQ1YzQ1OTU%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=cg3QRxZQzcLYXeCKRgSea83t2tTlJRp1FqyWEQR-KCoMjZxcNWKaU%7Eo77EVp6JfARfujfn4C5cjAzYdsiMzKfIrLZ7ZwTJ1MDek19HKb2xFH9BN7N6iZVWX%7EXM7INsNNHoYOvSrUn296%7E8egJZrY%7Ecd9MtzIy2ldIqW-3MVKb99FjfiFL2i9uXu5y0x4kCNB0twgcFK28QcwSeXtLN%7EAdKQqdLiZvi85s8jM0u2ws7fI9M0A7i2XddFlZ3JhGo4VsC4QRQ-NHvJGU9SSjBauBRTI2EjkNmT12msyvhbCT9L%7EG7LoW5jQhSYav3feIIWIuKWBPLQFskfpT9OnfKb9Aw__&Key-Pair-Id=KVTP0A1DKRTAX [following]
--2024-02-23 12:55:13--  https://cdn-lfs.huggingface.co/repos/7b/90/7b90a2edf952802c9c7e2de6b12c802cce10009f1476c3029595e3fc9bbd1fe9/bf6ab4b53f4b66adbff204ed7f4e36c9c704be6852b407c5147934f2a45c4595?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27train_images_4.tar.gz%3B+filename%3D%22train_images_4.tar.gz%22%3B&response-content-type=application%2Fgzip&Expires=1708949015&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk0OTAxNX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy83Yi85MC83YjkwYTJlZGY5NTI4MDJjOWM3ZTJkZTZiMTJjODAyY2NlMTAwMDlmMTQ3NmMzMDI5NTk1ZTNmYzliYmQxZmU5L2JmNmFiNGI1M2Y0YjY2YWRiZmYyMDRlZDdmNGUzNmM5YzcwNGJlNjg1MmI0MDdjNTE0NzkzNGYyYTQ1YzQ1OTU%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=cg3QRxZQzcLYXeCKRgSea83t2tTlJRp1FqyWEQR-KCoMjZxcNWKaU%7Eo77EVp6JfARfujfn4C5cjAzYdsiMzKfIrLZ7ZwTJ1MDek19HKb2xFH9BN7N6iZVWX%7EXM7INsNNHoYOvSrUn296%7E8egJZrY%7Ecd9MtzIy2ldIqW-3MVKb99FjfiFL2i9uXu5y0x4kCNB0twgcFK28QcwSeXtLN%7EAdKQqdLiZvi85s8jM0u2ws7fI9M0A7i2XddFlZ3JhGo4VsC4QRQ-NHvJGU9SSjBauBRTI2EjkNmT12msyvhbCT9L%7EG7LoW5jQhSYav3feIIWIuKWBPLQFskfpT9OnfKb9Aw__&Key-Pair-Id=KVTP0A1DKRTAX
Resolving cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)... 108.157.109.99, 108.157.109.100, 108.157.109.59, ...
Connecting to cdn-lfs.huggingface.co (cdn-lfs.huggingface.co)|108.157.109.99|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 29147095755 (27G) [application/gzip]
Saving to: ‘train_images_4.tar.gz’

train_images_4.tar.gz                    100%[================================================================================>]  27.14G  35.1MB/s    in 15m 51s 

2024-02-23 13:11:04 (29.2 MB/s) - ‘train_images_4.tar.gz’ saved [29147095755/29147095755]
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ ls -lh
total 155G
-rw-rw-r-- 1 ubuntu root  84K Feb 23 09:46 README.md
-rw-rw-r-- 1 ubuntu root  46K Feb 23 09:45 classes.py
-rw-rw-r-- 1 ubuntu root 1.6K Feb 23 09:46 gitattributes
-rw-rw-r-- 1 ubuntu root 4.7K Feb 23 09:45 imagenet-1k.py
-rw-rw-r-- 1 ubuntu root  13G Feb 23 10:12 test_images.tar.gz
-rw-rw-r-- 1 ubuntu root  28G Feb 23 12:11 train_images_0.tar.gz
-rw-rw-r-- 1 ubuntu root  28G May 24  2022 train_images_1.tar.gz
-rw-rw-r-- 1 ubuntu root  28G May 24  2022 train_images_2.tar.gz
-rw-rw-r-- 1 ubuntu root  28G May 24  2022 train_images_3.tar.gz
-rw-rw-r-- 1 ubuntu root  28G May 24  2022 train_images_4.tar.gz
-rw-rw-r-- 1 ubuntu root 6.3G Feb 23 12:37 val_images.tar.gz
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cd /mnt/data02/data/cache/
ubuntu@cese-produtech3r:/mnt/data02/data/cache$ mkdir imagenet-1k
ubuntu@cese-produtech3r:/mnt/data02/data/cache$ cd imagenet-1k/
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k$ mkdir train
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k$ mkdir test
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k$ mkdir val
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k$ 
```
<!--- cSpell:enable --->

Copy and extract test data (will take some time, avoid using the `-v` (verbose) flag):

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ cp test_images.tar.gz /mnt/data02/data/cache/imagenet-1k/test/

ubuntu@cese-produtech3r:~$ cd /mnt/data02/data/cache/imagenet-1k/test/
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/test$ 
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/test$ tar -xvzf test_images.tar.gz 
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/test$ rm test_images.tar.gz 
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/test$ ls -l | wc -l
100000
```
<!--- cSpell:enable --->

Copy and extract test data:

time wget -q -O /dev/null https://unix.stackexchange.com/


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp val_images.tar.gz /mnt/data02/data/cache/imagenet-1k/val/

real	5m38.139s
user	0m0.065s
sys	0m10.050s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/val$ 
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/val$ time tar -xzf val_images.tar.gz 

real	46m36.322s
user	1m20.056s
sys	0m50.401s
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/val$ rm val_images.tar.gz
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/val$ ls -l | wc -l
50001
```
<!--- cSpell:enable --->

Copy and extract test data:

<!--- cSpell:disable --->
```shell
buntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp train_images_0.tar.gz /mnt/data02/data/cache/imagenet-1k/train/

real	17m48.831s
user	0m0.280s
sys	0m38.470s

ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp train_images_1.tar.gz /mnt/data02/data/cache/imagenet-1k/train/

real	17m27.775s
user	0m0.246s
sys	0m37.565s

ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp train_images_2.tar.gz /mnt/data02/data/cache/imagenet-1k/train/

real	14m48.684s
user	0m0.192s
sys	0m37.380s

ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp train_images_3.tar.gz /mnt/data02/data/cache/imagenet-1k/train/

real	14m54.299s
user	0m0.216s
sys	0m37.659s

ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ time cp train_images_4.tar.gz /mnt/data02/data/cache/imagenet-1k/train/

real	10m48.578s
user	0m0.245s
sys	0m38.058s

ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$ cd /mnt/data02/data/cache/imagenet-1k/train
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ 


ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time tar -xzf train_images_0.tar.gz

real	74m1.013s
user	5m23.077s
sys	3m24.777s


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

