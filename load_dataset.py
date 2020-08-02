# coding = utf-8
# @File    : load_dataset.py


import torch
import torch.utils.data as data
import numpy as np
import cv2
import glob
import pandas as pd

'''
# 字母标记名称(文件夹名字）
defect_label_order = ["CME", "CSOM", "EACB", "IC", "NE", "OE", "SOM", "TMC"]
# 与字母标记一一对应(类别名字）
defect_code = {
    "cholestestoma of middle ear": "CME",
    "chromic suppurative otits media": "CSOM",
    "external auditory cana bleeding": "EACB",
    "impacted cerumen": "IC",
    "normal eardrum": "NE",
    "otomycosis external": "OE",
    "secretory otitis media": "SOM",
    "tympanic membrane calcification": "TMC"
}
# 与数字标记一一对应(类别与数字标签一一对应）
defect_label0 = {
    "CME": "0",
    "CSOM": "1",
    "EACB": "2",
    "IC": "3",
    "NE": "4",
    "OE": "5",
    "SOM": "6",
    "TMC": "7"
}
'''

defect_label = {
    "厨余垃圾菜叶菜根": "0",
    "厨余垃圾茶叶渣": "1",
    "厨余垃圾大骨头": "2",
    "厨余垃圾蛋壳": "3",
    "厨余垃圾剩饭剩菜": "4",
    "厨余垃圾水果果皮": "5",
    "厨余垃圾水果果肉": "6",
    "厨余垃圾鱼骨": "7",
    "可回收物包": "8",
    "可回收物玻璃杯": "9",
    "可回收物插头电线": "10",
    "可回收物充电宝": "11",
    "可回收物锅": "12",
    "可回收物化妆品瓶": "13",
    "可回收物金属食品罐": "14",
    "可回收物酒瓶": "15",
    "可回收物旧衣服": "16",
    "可回收物快递纸袋": "17",
    "可回收物毛绒玩具": "18",
    "可回收物皮鞋": "19",
    "可回收物食用油桶": "20",
    "可回收物塑料玩具": "21",
    "可回收物塑料碗盆": "22",
    "可回收物塑料衣架": "23",
    "可回收物调料瓶": "24",
    "可回收物洗发水瓶": "25",
    "可回收物易拉罐": "26",
    "可回收物饮料瓶": "27",
    "可回收物砧板": "28",
    "可回收物枕头": "29",
    "可回收物纸板箱": "30",
    "其他垃圾破碎花盆及碟碗": "31",
    "其他垃圾污损塑料": "32",
    "其他垃圾牙签": "33",
    "其他垃圾烟蒂": "34",
    "其他垃圾一次性快餐盒": "35",
    "其他垃圾竹筷": "36",
    "有害垃圾干电池": "37",
    "有害垃圾过期药物": "38",
    "有害垃圾软膏": "39"
}


# 用字典存储类别名字和数字标记
# label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))

# 获取图片路径
def get_image_pd(img_root):  # img-root = '/home/lzz/Ear'
    # 利用glob指令获取图片列表（/*的个数根据文件构成确定）获取完整路径
    img_list = glob.glob(img_root + "/*/*.jpg")
    # print(img_list)
    # 利用DataFrame指令构建图片列表的字典，即图片列表的序号与其路径一一对应
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # 获取文件夹名称，也可以认为是标签名称
    image_pd["label_name"] = image_pd["ImageName"].apply(
        lambda x: x.split("\\")[-2])
    # print(image_pd)
    # 将标签名称转化为数字标记
    image_pd["label"] = image_pd["label_name"].apply(lambda x: defect_label[x])
    # image_pd["label"] = image_pd["lable_name"].apply(lambda x: defect_label[x])
    # print(image_pd["label"].value_counts())
    # print(image_pd)
    return image_pd

# 数据集
class dataset(data.Dataset):
    # 参数预定义
    def __init__(self, anno_pd, transforms=None, debug=False, test=False):
        # 图像路径
        self.paths = anno_pd['ImageName'].tolist()
        # 图像数字标签
        self.labels = anno_pd['label'].tolist()
        # 数字增强
        self.transforms = transforms
        # 程序调试
        self.debug = debug
        # 判定是否训练或测试
        self.test = test
    # 返回图片个数
    def __len__(self):
        return len(self.paths)
    # 获取每个图片
    def __getitem__(self, item):
        # 图像路径
        img_path = self.paths[item]
        # 读取
        img = cv_imread(img_path)
        # 格式转换
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 是否进行数据增强
        if self.transforms is not None:
            img = self.transforms(img)
        # 图像对应标签
        label = self.labels[item]
        # tensor和对应标签
        return torch.from_numpy(img).float(), int(label)

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    return cv_img

# 整理图片
def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label
