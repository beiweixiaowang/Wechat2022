import warnings

warnings.simplefilter("ignore")
import logging
import os
import time
import torch

from config.new_config import parse_args
from utils.new_data_helper import create_dataloaders
from model.Leah_model_4 import MultiModal
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from ark_nlp.factory.utils.ema import EMA
from ark_nlp.factory.utils.attack import FGM, PGD
from tqdm import tqdm


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build nezha_model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    # 增加ema和PGD
    # ema = EMA(model.parameters(), decay=0.995)
    for epoch in range(args.max_epochs):
        with tqdm(total=len(train_dataloader)) as t:
            for batch in train_dataloader:
                t.set_description('Epoch %i' % epoch)
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                optimizer.step()
                # 训练过程中，更新完参数后，同步update shadow weights
                # ema.update(model.parameters())
                optimizer.zero_grad()
                scheduler.step()
                t.set_postfix(loss=float(loss.cpu()), acc=float(accuracy.cpu()))
                time.sleep(0.1)
                t.update(1)
                step += 1
            # 4. validation
            # eval前，进行ema的权重替换
            # ema.store(model.parameters())
            # ema.copy_to(model.parameters())
            loss, results = validate(model, val_dataloader)
            # eval之后，恢复原来模型的参数
            # ema.restore(model.parameters())

            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                best_score = mean_f1
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
