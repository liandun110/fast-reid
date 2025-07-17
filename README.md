## Added by zhbli

### 安装

注意，不能在 arm64 架构的机器（如 NVIDIA orin）上运行。因为在此架构的机器上，QT的安装有问题。

### 代码功能

`python datasets/geo_align.py`: 计算监控图中像素点与地图中像素点的对应关系

`bash det_crop_reid.bash`: 对一段视频，依次执行：行人检测、行人子图裁剪、reid特征提取。

`python demo/UI.py`: 跨境追踪主界面

`python demo/UI_footprint.py`: UI.py的升级版，带有地图。当选中监控画面的行人后，会在地图上显示该行人的位置。

### 如何在 headless 的远程服务器上运行 QT 的 UI 界面

#### 本地准备
(1) 安装必要工具
```bash
sudo apt update
sudo apt install -y xauth openssh-client
```
(2) 配置 SSH 快捷连接
编辑～/.ssh/config 文件：nano ~/.ssh/config

添加以下内容（替换端口和用户名）：

```
Host featurize
HostName featurize.cn
User username
Port 61389
ForwardX11 yes
ForwardX11Trusted yes
```

#### 服务器端配置
(1) 安装 X11 和 Qt 依赖
```bash
sudo apt update
sudo apt install -y xauth libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 \
libxcb-xfixes0 libxcb-cursor0
```

在服务器测试连接：xclock
如果本地弹出时钟窗口，说明 X11 转发成功。

#### 运行 PyQt5 程序

(1) 设置环境变量

```bash
export DISPLAY=localhost:10.0 # 通常 SSH -Y 会自动设置 export
```

(2) 启动程序

```bash
python demo/UI.py
```

### 数据集组织方式

```bash
seq_path
  mg1  # ffmpeg -i 人脸追踪*.mp4 -f image2 -vcodec mjpeg -qscale:v 2 images/%06d.jpg
  det  # python datasets/person_detect.py --seq_path=seq_path
  person_crops  # python datasets/crop_person.py --seq_path=seq_path
  reid_features  # python demo/demo.py --seq-path=seq_path
```

### 界面功能

1. 指定seq_path后，自动获取上述各项信息。

2. 鼠标点击画面左侧某个行人（query person），自动显示其行人框。（这种显示是必要的。因为如果点击后没显示出行人框，说明目标检测器根本没有找到这个人，后续所有操作都是无意义的。也为后续工作人员始终关注这个人提供便利。）

3. 将query person子图（根据detection_id从person_crops中获取）展示在UI下方，根据detection_id从reid_features文件夹中获取其特征，并与右侧视频所有特征计算相似度，同样在UI下方展示若干得分最高的candidate persons及相关信息，如帧号、得分.

4. 点击某个candidate person，右侧视频自动跳转到对应帧，并在相应行人上绘制边框。

---

<img src=".github/FastReID-Logo.png" width="300" >

[![Gitter](https://badges.gitter.im/fast-reid/community.svg)](https://gitter.im/fast-reid/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Gitter: [fast-reid/community](https://gitter.im/fast-reid/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

Wechat: 

<img src=".github/wechat_group.png" width="150" >


FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a ground-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## What's New

- [Sep 2021] [DG-ReID](https://github.com/xiaomingzhid/sskd) is updated, you can check the [paper](https://arxiv.org/pdf/2108.05045.pdf).
- [June 2021] [Contiguous parameters](https://github.com/PhilJd/contiguous_pytorch_params) is supported, now it can
  accelerate ~20%.
- [May 2021] Vision Transformer backbone supported, see `configs/Market1501/bagtricks_vit.yml`.
- [Apr 2021] Partial FC supported in [FastFace](projects/FastFace)!
- [Jan 2021] TRT network definition APIs in [FastRT](projects/FastRT) has been released! 
Thanks for [Darren](https://github.com/TCHeish)'s contribution.
- [Jan 2021] NAIC20(reid track) [1-st solution](projects/NAIC20) based on fastreid has been released！
- [Jan 2021] FastReID V1.0 has been released！🎉
  Support many tasks beyond reid, such image retrieval and face recognition. See [release notes](https://github.com/JDAI-CV/fast-reid/releases/tag/v1.0.0).
- [Oct 2020] Added the [Hyper-Parameter Optimization](projects/FastTune) based on fastreid. See `projects/FastTune`.
- [Sep 2020] Added the [person attribute recognition](projects/FastAttr) based on fastreid. See `projects/FastAttr`.
- [Sep 2020] Automatic Mixed Precision training is supported with `apex`. Set `cfg.SOLVER.FP16_ENABLED=True` to switch it on.
- [Aug 2020] [Model Distillation](projects/FastDistill) is supported, thanks for [guan'an wang](https://github.com/wangguanan)'s contribution.
- [Aug 2020] ONNX/TensorRT converter is supported.
- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](projects) on top of it. We'll open source more research projects in this way.
- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).

We write a [fastreid intro](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2020/05/29/fastreid.html) 
and [fastreid v1.0](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2021/04/28/fastreid-v1.html) about this toolbox.

## Changelog

Please refer to [changelog.md](CHANGELOG.md) for details and release history.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](GETTING_STARTED.md).

Learn more at out [documentation](https://fast-reid.readthedocs.io/). And see [projects/](projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](LICENSE).

## Citing FastReID

If you use FastReID in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
