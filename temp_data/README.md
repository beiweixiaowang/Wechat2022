## 中间文件
### 文本中间文件生成方法  
 将title, ocr的前60个字和后60个字和asr的前60个字和后60个字拼接起来，形成新的文件，存成pickle形式


### 中间视频特征生成方法
。。。

### 文件结构
 ```bash
.
├── temp_text
│   ├── train.pickle
│   ├── test_a.pickle
│   └── unlabeled.pickle
├── README.md
├── verify_data.py
└── temp_zip_feats
    |__...
```