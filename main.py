"""Main script for ADDA."""
import torch
from torchvision import datasets, transforms, models

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed

from datasets import get_office_home, get_office_31


if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_office_31(dataset = 'office-31-amazon', train=True)
    src_data_loader_eval = get_office_31(dataset = 'office-31-amazon', train=False)
    tgt_data_loader = get_office_31(dataset = 'office-31-webcam', train=True)
    tgt_data_loader_eval = get_office_31(dataset = 'office-31-webcam', train=False)

    progenitor = models.googlenet(pretrained=True, aux_logits=False)
    progenitor.fc = torch.nn.Linear(1024, 31)
    progenitor = progenitor.to(torch.device('cuda:0'))

    src_encoder = torch.nn.Sequential(*(list(progenitor.children())[10:-1]))
    src_classifier = torch.nn.Linear(1024, 31).to(torch.device('cuda:0'))
    tgt_encoder = torch.nn.Sequential(*(list(progenitor.children())[10:-1]))
    tgt_classifier = torch.nn.Linear(1024, 31).to(torch.device('cuda:0'))

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
