import random
import zipfile
from io import BytesIO
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from utils.category_id_map import category_id_to_lv2id


def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # 加载text文件
        self.anns = pd.read_pickle(ann_path)[:100]
        # print(self.anns.columns)
        # 定义分词器
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns.iloc[idx]['id']

        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        frame_feats = torch.from_numpy(raw_feats)
        # print(frame_feats.size())
        num_frames, feat_dim = raw_feats.shape
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.bert_seq_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = torch.tensor(encoded_inputs['input_ids']).flatten()
        mask = torch.tensor(encoded_inputs['attention_mask']).flatten()
        return input_ids, mask

    def tokenize_text_title_topK(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=200, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)
        # frame_token_type_ids = torch.ones(frame_input.shape[:-1], dtype=torch.long)

        # Step 2, load text tokens
        # title_top20_input, title_top20_mask = self.tokenize_text_title_topK(self.anns.iloc[idx]['title_top20'])
        text_input, text_mask = self.tokenize_text(self.anns.iloc[idx]['all_text'])


        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            # frame_token_type_ids=frame_token_type_ids,
            text_input=text_input,
            text_mask=text_mask,
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns.iloc[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
