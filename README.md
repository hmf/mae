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

We need to update th code. Code originally written for PyTorch v1.x. Some references for the update:

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

Next we tried to get the ImageNet dataset from HuggingFace. Here are the links (to download them, you need to be registered in HuggingFace):

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


These are the set of folders that are expected by the code:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/train
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/test
vscode ➜ /workspaces/mae (test_1) $ mkdir ./data/val
```
<!--- cSpell:enable --->

The dataset if very large so it is necessary to use the share. First make sure you can access the share via a standard SSH login. Create the destination folder to store the data.

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cd /mnt/data02/data/src
ubuntu@cese-produtech3r:/mnt/data02/data/src$ 
ubuntu@cese-produtech3r:/mnt/data02/data/src$ mkdir imagenet-1k
```
<!--- cSpell:enable --->

Make sure you can launch the dev container (Docker). The container must also bind to the share.  

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ ls /mnt/data02/data/cache/imagenet-1k/
test  train  val
```
<!--- cSpell:enable --->


Go to local source data path. Copy the source data from the local to the remote node. These files are large so it may take a while. Alternatively download (via curl or wget) the data directly from HuggingFace from the remote shared folder. Here is an example of copying our local files to the remote shared node:

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

Here are the examples of downloading the training data directly from HuggingFace. Note that we require to login to downlaod thr data. The following command will fail:

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

<!-- https://askubuntu.com/questions/29079/how-do-i-provide-a-username-and-password-to-wget -->
<!--  https://serverfault.com/questions/150282/escape-a-in-the-password-parameter-of-wget -->

Here is a command that provides the user and password. Note the space in front of command so as not to save in history. This prevents others from recalling your sensitive data. This command also fails because HuggingFace uses a token for controlling the access to the data:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz
--2024-02-23 11:24:13--  https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz?download=true
Resolving huggingface.co (huggingface.co)... 54.192.95.21, 54.192.95.79, 54.192.95.26, ...
Connecting to huggingface.co (huggingface.co)|54.192.95.21|:443... connected.
HTTP request sent, awaiting response... 401 Unauthorized
Unknown authentication scheme.

Username/Password Authentication Failed.
```
<!--- cSpell:enable --->

<!-- https://discuss.huggingface.co/t/private-data-and-wget/35115/2 -->
The next commands show the data download using the token (replace `HF_TOKEN` with your token).

File `train_images_1.tar.gz` (`train_images_0.tar.gz` missing):

 <!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/src/imagenet-1k$  wget --user=usr --password='PASS' --header="Authorization: Bearer HF_TOKEN" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz
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

2024-02-23 12:23:25 (9.86 MB/s) - ‘train_images_1.tar.gz’ saved [29261436971/29261436971]
```
<!--- cSpell:enable --->

File: `train_images_2.tar.gz`

 <!--- cSpell:disable --->
```shell
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

File: `train_images_3.tar.gz`

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

File: `train_images_4.tar.gz`

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

These are the files copied to the shared folder:

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

Create the path were the data will be paced for training and evaluation:

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

Copy and extract **test data** (will take some time, avoid using the `-v` (verbose) flag). Here we copy the test data and extract it in the final path used by the code. The archive is removed because it is not required anymore:

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

Copy and extract **validation data**. The archive is removed because it is not required anymore. We use the `time` command to check the time (taking too long):


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

Copy and extract **train data**:

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

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time tar -xzf train_images_1.tar.gz

real	48m38.565s
user	5m14.130s
sys	3m16.233s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time tar -xzf train_images_2.tar.gz

real	55m34.729s
user	5m28.846s
sys	3m13.015s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time tar -xzf train_images_3.tar.gz

real	51m28.530s
user	5m18.405s
sys	3m18.663s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time tar -xzf train_images_4.tar.gz

real	57m31.923s
user	5m26.846s
sys	3m26.966s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ time ls -l | wc -l
1281173

real	2m9.870s
user	0m2.591s
sys	0m13.099s

ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ df -H
Filesystem                          Size  Used Avail Use% Mounted on
tmpfs                                12G  1.4M   12G   1% /run
/dev/vda1                           104G   67G   38G  65% /
tmpfs                                60G     0   60G   0% /dev/shm
tmpfs                               5.3M     0  5.3M   0% /run/lock
/dev/vda15                          110M  6.4M  104M   6% /boot/efi
10.55.0.23:/mnt/pool03/cese/data02  1.1T  589G  512G  54% /mnt/data02
tmpfs                                12G  8.2k   12G   1% /run/user/1002

```
<!--- cSpell:enable --->

Remove the archives of the training data that are not required anymore:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ ls *.gz
train_images_0.tar.gz  train_images_1.tar.gz  train_images_2.tar.gz  train_images_3.tar.gz  train_images_4.tar.gz
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k/train$ rm -v train_images_*.tar.gz
removed 'train_images_0.tar.gz'
removed 'train_images_1.tar.gz'
removed 'train_images_2.tar.gz'
removed 'train_images_3.tar.gz'
removed 'train_images_4.tar.gz'
```
<!--- cSpell:enable --->

Current space left:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ df -H
Filesystem                          Size  Used Avail Use% Mounted on
tmpfs                                12G  1.4M   12G   1% /run
/dev/vda1                           104G   68G   37G  65% /
tmpfs                                60G     0   60G   0% /dev/shm
tmpfs                               5.3M     0  5.3M   0% /run/lock
/dev/vda15                          110M  6.4M  104M   6% /boot/efi
10.55.0.23:/mnt/pool03/cese/data02  1.1T  443G  657G  41% /mnt/data02
tmpfs                                12G  8.2k   12G   1% /run/user/1002
```
<!--- cSpell:enable --->


Here is the command used to used to train the model using the previously downloaded checkpoint and assuming the data is in the share `./data` path:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/mae (test_1) $ python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /mnt/data02/data/cache/imagenet-1k/

```
<!--- cSpell:enable --->

Process stalls but is executing (see below). After 15 minutes, the GPU is till not being used. The process was forcefully terminated.

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ ps ax | grep -i finetune
1869259 pts/1    Dl+    0:03 python main_finetune.py --eval --resume checkpoints/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /mnt/data02/data/cache/imagenet-1k/
```
<!--- cSpell:enable --->

Can we copy this to the local VM disk and continue?. The following commands also stalled:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ du /mnt/data02/data/cache/imagenet-1k/
ubuntu@cese-produtech3r:~$ du -sh /mnt/data02/data/cache/imagenet-1k/
ubuntu@cese-produtech3r:~$ time du -sh /mnt/data02/data/cache/imagenet-1k/
164G	/mnt/data02/data/cache/imagenet-1k/

real	12m34.151s
user	0m0.978s
sys	1m24.493s
```
<!--- cSpell:enable --->

Seems not, we need 164GB and only have a total of 104G (37G available). This is a show stopper. 

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ df -H
Filesystem                          Size  Used Avail Use% Mounted on
tmpfs                                12G  1.4M   12G   1% /run
/dev/vda1                           104G   68G   37G  65% /
tmpfs                                60G     0   60G   0% /dev/shm
tmpfs                               5.3M     0  5.3M   0% /run/lock
/dev/vda15                          110M  6.4M  104M   6% /boot/efi
10.55.0.23:/mnt/pool03/cese/data02  1.1T  443G  657G  41% /mnt/data02
tmpfs                                12G  8.2k   12G   1% /run/user/1002
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data02/data/cache/imagenet-1k$ time du -h
6.6G	./val
144G	./train
14G	./test
164G	.

real	8m47.133s
user	0m0.723s
sys	1m19.383s
```
<!--- cSpell:enable --->


Need to copy data to new partition. 

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ lsblk
NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS
loop0     7:0    0 63.9M  1 loop /snap/core20/2105
loop1     7:1    0 63.9M  1 loop /snap/core20/2182
loop2     7:2    0   87M  1 loop /snap/lxd/26881
loop3     7:3    0   87M  1 loop /snap/lxd/27037
loop4     7:4    0 40.4M  1 loop /snap/snapd/20671
vda     252:0    0  100G  0 disk 
├─vda1  252:1    0 99.9G  0 part /
├─vda14 252:14   0    4M  0 part 
└─vda15 252:15   0  106M  0 part /boot/efi
vdb     252:16   0  170G  0 disk 
```
<!--- cSpell:enable --->

<!-- https://phoenixnap.com/kb/linux-format-disk -->

The partition/disk `vdb` has no file system:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ lsblk -f
NAME FSTYPE FSVER LABEL UUID                                 FSAVAIL FSUSE% MOUNTPOINTS
loop0
     squash 4.0                                                    0   100% /snap/core20/2105
loop1
     squash 4.0                                                    0   100% /snap/core20/2182
loop2
     squash 4.0                                                    0   100% /snap/lxd/26881
loop3
     squash 4.0                                                    0   100% /snap/lxd/27037
loop4
     squash 4.0                                                    0   100% /snap/snapd/20671
vda                                                                         
├─vda1
│    ext4   1.0   cloudimg-rootfs
│                       f44c6339-a6fd-4158-9e95-b421ec3173c9   33.9G    65% /
├─vda14
│                                                                           
└─vda15
     vfat   FAT32 UEFI  AD67-58A5                              98.3M     6% /boot/efi
vdb                                                                         
```
<!--- cSpell:enable --->

Format with `ext4`: 

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ sudo mkfs -t ext4 /dev/vdb
mke2fs 1.46.5 (30-Dec-2021)
Discarding device blocks: done                            
Creating filesystem with 44564480 4k blocks and 11141120 inodes
Filesystem UUID: 78ae64d2-664d-4bf0-a681-1f7ca8258365
Superblock backups stored on blocks: 
	32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208, 
	4096000, 7962624, 11239424, 20480000, 23887872

Allocating group tables: done                            
Writing inode tables: done                            
Creating journal (262144 blocks): done
Writing superblocks and filesystem accounting information: done     

```
<!--- cSpell:enable --->

We now have `vdb` with an `ext4` filesystem: 

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ lsblk -f
NAME    FSTYPE   FSVER LABEL           UUID                                 FSAVAIL FSUSE% MOUNTPOINTS
loop0   squashfs 4.0                                                              0   100% /snap/core20/2105
loop1   squashfs 4.0                                                              0   100% /snap/core20/2182
loop2   squashfs 4.0                                                              0   100% /snap/lxd/26881
loop3   squashfs 4.0                                                              0   100% /snap/lxd/27037
loop4   squashfs 4.0                                                              0   100% /snap/snapd/20671
vda                                                                                        
├─vda1  ext4     1.0   cloudimg-rootfs f44c6339-a6fd-4158-9e95-b421ec3173c9   33.9G    65% /
├─vda14                                                                                    
└─vda15 vfat     FAT32 UEFI            AD67-58A5                              98.3M     6% /boot/efi
vdb     ext4     1.0                   78ae64d2-664d-4bf0-a681-1f7ca8258365                
```
<!--- cSpell:enable --->


Add the partition and its respective mount point. We have the following set-up:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cat /etc/fstab
LABEL=cloudimg-rootfs	/	 ext4	discard,errors=remount-ro	0 1
LABEL=UEFI	/boot/efi	vfat	umask=0077	0 1
10.55.0.23:/mnt/pool03/cese/data02  /mnt/data02 nfs nfsvers=4,defaults,noatime,nodiratime 0 0
```
<!--- cSpell:enable --->

<!-- https://formatswap.com/blog/linux-tutorials/how-to-mount-a-hard-drive-at-boot-in-ubuntu-22-04/ -->

sudo mkdir /mnt/data

/dev/MOUNT_POINT /media/MOUNT_DIR ext4 defaults 0 0

/dev/vdb /mnt/data ext4 defaults 0 0

sudo mount -a

Create the mount point:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ sudo mkdir /mnt/data
ubuntu@cese-produtech3r:~$ ls /mnt/
data  data02
```
<!--- cSpell:enable --->

Add the mount point to the OS configuration (backup before changes):

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ sudo cp /etc/fstab /etc/fstab.1
ubuntu@cese-produtech3r:~$ sudo nano /etc/fstab
```
<!--- cSpell:enable --->

I used tabs to separate the fields. Here is the content of the configuration file `fstab`:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cat /etc/fstab
LABEL=cloudimg-rootfs	/	 ext4	discard,errors=remount-ro	0 1
LABEL=UEFI	/boot/efi	vfat	umask=0077	0 1
/dev/vdb	/mnt/data	ext4	defaults	0 0
10.55.0.23:/mnt/pool03/cese/data02  /mnt/data02 nfs nfsvers=4,defaults,noatime,nodiratime 0 0
```
<!--- cSpell:enable --->

Force mount and check that its available:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ sudo mount -a
ubuntu@cese-produtech3r:~$ ls -l /mnt/
total 18
drwxr-xr-x 3 root root 4096 Mar  1 13:59 data
drwxrwxrwx 5 root root    6 Feb 28 18:49 data02
```
<!--- cSpell:enable --->

Create the top folder to hold the ImageNet-1K data:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cd /mnt/data
ubuntu@cese-produtech3r:/mnt/data$ 

ubuntu@cese-produtech3r:/mnt/data$ cd ..
ubuntu@cese-produtech3r:/mnt$ ls -l
total 18
drwxr-xr-x 3 root root 4096 Mar  1 13:59 data
drwxrwxrwx 5 root root    6 Feb 28 18:49 data02
ubuntu@cese-produtech3r:/mnt$ whoami 
ubuntu
ubuntu@cese-produtech3r:/mnt$ sudo chown ubuntu:ubuntu /mnt/data
ubuntu@cese-produtech3r:/mnt$ ls -l
total 18
drwxr-xr-x 3 ubuntu ubuntu 4096 Mar  1 13:59 data
drwxrwxrwx 5 root   root      6 Feb 28 18:49 data02
```
<!--- cSpell:enable --->

Create the separate folders for training with the ImageNet-1K data:

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt$ cd data
ubuntu@cese-produtech3r:/mnt/data$ mkdir train
ubuntu@cese-produtech3r:/mnt/data$ mkdir test
ubuntu@cese-produtech3r:/mnt/data$ mkdir val
ubuntu@cese-produtech3r:/mnt/data$ ls -l
total 28
drwx------ 2 root   root   16384 Mar  1 13:59 lost+found
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 test
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 train
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 val
```
<!--- cSpell:enable --->

<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:~$ cd /mnt/data
ubuntu@cese-produtech3r:/mnt/data$ ls
lost+found  test  train  val
ubuntu@cese-produtech3r:/mnt/data$ ls -l
total 28
drwx------ 2 root   root   16384 Mar  1 13:59 lost+found
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 test
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 train
drwxrwxr-x 2 ubuntu ubuntu  4096 Mar  1 14:27 val
ubuntu@cese-produtech3r:/mnt/data$ cd train/
```
<!--- cSpell:enable --->

Each download took about 14 minutes:

<!--- cSpell:disable --->
```shell
  wget --user=usr --password='PASS' --header="Authorization: Bearer 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" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_0.tar.gz

  wget --user=usr --password='PASS' --header="Authorization: Bearer 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" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_1.tar.gz

  wget --user=usr --password='PASS' --header="Authorization: Bearer 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" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_2.tar.gz

  wget --user=usr --password='PASS' --header="Authorization: Bearer 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" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_3.tar.gz

  wget --user=usr --password='PASS' --header="Authorization: Bearer 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" https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train_images_4.tar.gz
```
<!--- cSpell:enable --->



<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data/train$ ls -lh
total 136G
-rw-rw-r-- 1 ubuntu ubuntu 28G May 24  2022 train_images_0.tar.gz
-rw-rw-r-- 1 ubuntu ubuntu 28G May 24  2022 train_images_1.tar.gz
-rw-rw-r-- 1 ubuntu ubuntu 28G May 24  2022 train_images_2.tar.gz
-rw-rw-r-- 1 ubuntu ubuntu 28G May 24  2022 train_images_3.tar.gz
-rw-rw-r-- 1 ubuntu ubuntu 28G May 24  2022 train_images_4.tar.gz
```
<!--- cSpell:enable --->


time tar -xzf train_images_0.tar.gz

ubuntu@cese-produtech3r:/mnt/data/train$ df -H
Filesystem                          Size  Used Avail Use% Mounted on
tmpfs                                12G  1.4M   12G   1% /run
/dev/vda1                           104G   68G   37G  65% /
tmpfs                                60G     0   60G   0% /dev/shm
tmpfs                               5.3M     0  5.3M   0% /run/lock
/dev/vda15                          110M  6.4M  104M   6% /boot/efi
10.55.0.23:/mnt/pool03/cese/data02  1.1T  443G  657G  41% /mnt/data02
/dev/vdb                            179G  170G     0 100% /mnt/data
tmpfs                                12G  8.2k   12G   1% /run/user/1002
ubuntu@cese-produtech3r:/mnt/data/train$ ls -l | wc -l
214766
ubuntu@cese-produtech3r:/mnt/data/train$ rm *.JPEG
-bash: /usr/bin/rm: Argument list too long


<!--- cSpell:disable --->
```shell
ubuntu@cese-produtech3r:/mnt/data/train$ time tar -xzf train_images_0.tar.gz

real	3m16.906s
user	2m28.103s
sys	0m50.372s

ubuntu@cese-produtech3r:/mnt/data/train$ time rm train_images_0.tar.gz 

real	0m1.317s
user	0m0.001s
sys	0m1.305s

ubuntu@cese-produtech3r:/mnt/data/train$ time tar -xzf train_images_1.tar.gz

real	2m46.793s
user	2m23.159s
sys	0m45.031s

ubuntu@cese-produtech3r:/mnt/data/train$ time rm train_images_1.tar.gz 

real	0m0.662s
user	0m0.001s
sys	0m0.650s

ubuntu@cese-produtech3r:/mnt/data/train$ time tar -xzf train_images_2.tar.gz

real	2m31.879s
user	2m17.275s
sys	0m45.706s

ubuntu@cese-produtech3r:/mnt/data/train$ time rm train_images_2.tar.gz 

real	0m3.082s
user	0m0.000s
sys	0m3.065s
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

