import torch
from torch.utils.data import SequentialSampler, DataLoader

from config.new_config import parse_args
from utils.new_data_helper import MultiModalDataset
from utils.category_id_map import lv2id_to_category_id
from model.Leah_model import MultiModal


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    print(dataset)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load nezha_model
    print("开始加载模型")
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    print("加载模型完毕")

    # 3. inference
    print("开始推理")
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())
    print("推理完毕")

    # 4. dump results
    print("写入文件")
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns['id'].tolist()):
            print(ann)
            video_id = ann
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
