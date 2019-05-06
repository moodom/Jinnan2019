# Jinnan2019
这是天池津南数字制造算法挑战赛——赛场二：物流货流限制品监测的代码分享，取得了27/2157的成绩。
比赛链接：https://tianchi.aliyun.com/competition/entrance/231703/introduction?spm=5176.12281957.1004.5.38b04c2aN7TkHY
该比赛的初赛是对5类限制品的检测任务，复赛是对限制品的语义分割的任务。
### 安装环境
Python3.6
PyTorch1.0
该代码主体部分来自mmdetection，配置环境请参照 https://github.com/open-mmlab/mmdetection 
### 网络结构
由于测试集有大约2/3的图片没有包含任何限制品，所以算法的策略是首先用二分类判断出哪些图片包含限制品，在选出具有
限制的图片中进行检测或分割任务。语义分割的网络选择的是cascade_mask_rcnn的结构，并采用多尺度训练，OHEM, soft_nms。
二分类网络的代码以及数据划分在utils里，语义分割网络的配置文件在configs里。
### 学习率调整
本次比赛采用了余弦退火学习率，可以使网络更好更快的收敛，大概可以提高一个点。当然据说大佬都是手调SGD，不过无脑训练确实有奇效。
### 数据增强
本次比赛采用了最简单线上旋转和翻折的数据增强。utils里还有生成假数据的代码，不过最后并没有使用。
### 网络融合
由于cascade_mask_rcnn没有多尺度测试的代码，比赛限制使用两个模型，我最后采用了一种简单的策略，即将测试图片以不同长宽比输入网络，得到不同结果后进行一个求平均的操作 。
### 其他
utils里有语义分割可视化的代码以及测试miou的代码
##总结
这是我第一次打比赛，历时两个月感觉学到很多，感谢这段时间帮助我的师兄KaiJin以及donglee,Hssen。
比赛的过程中暴露出一些问题，就是自己读的论文还是太少了，对整个算法的思考比较少，希望之后能沉下心好好研究一下。
