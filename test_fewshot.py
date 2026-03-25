import argparse
import json
import os
import os.path as osp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import FewShotPairDataset
from metrics import euclidean_metric
from model import PGM
from sampler import CategoriesSampler

def seed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default='./img/')
    parser.add_argument("--brpi-root", default='./brpi/')
    parser.add_argument("--checkpoint", default='./model_mstar17.pth')
    parser.add_argument("--split", default="val")
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default='./results/eval_fewshot.json')
    return parser.parse_args()


def main():
    args = parse_args()
    seed_random(args.seed)
    device = torch.device("cuda:0")
    dataset = FewShotPairDataset(image_root=args.image_root, split=args.split, brpi_root=args.brpi_root)
    sampler = CategoriesSampler(labels=dataset.labels, n_batch=args.episodes, n_cls=args.way, n_per=args.shot + args.query)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    model = PGM().to(device)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.eval()

    acc_list = []
    with torch.no_grad():
        for data, brpi, _ in loader:
            data = data.to(device)
            brpi = brpi.to(device)

            p = args.shot * args.way
            support_x, query_x = data[:p], data[p:]
            support_p, query_p = brpi[:p], brpi[p:]

            proto = model(support_x, support_p).view(args.shot, args.way, -1).mean(dim=0)
            query_feat = model(query_x, query_p)
            logits = euclidean_metric(proto, query_feat)
            labels = torch.arange(args.way, device=device).repeat(args.query)
            acc_list.append(count_acc(logits, labels))

    acc_array = np.asarray(acc_list, dtype=np.float32)
    result = {
        "episodes": int(args.episodes),
        "val_acc": float(acc_array.mean()),
    }
    print(result)

    # if args.out_json:
    #     out_dir = osp.dirname(args.out_json)
    #     if out_dir:
    #         os.makedirs(out_dir, exist_ok=True)
    #     with open(args.out_json, "w", encoding="utf-8") as f:
    #         json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
