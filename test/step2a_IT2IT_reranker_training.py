import json
from argparse import ArgumentParser

import os
from pathlib import Path
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR



from dataset import IT2IT_section_RerankerDataset, qformer_collate_fn

from data_utils import base_path, squarepad_transform, targetpad_transform, process_images_in_parallel
import wandb


device = "cuda"


def IT2IT_reranker_ft(
    num_epochs: int,
    reranker_model_name: str,
    learning_rate: float,
    batch_size: int,
    neg_num: int,
    transform: str,
    save_training: bool,
    checkpoint_path: str = None,
    reranker_checkpoint_path: str = None,
    **kwargs,
):
    training_path: Path = Path(
        base_path / f"reranker_model/{reranker_model_name}/{kwargs['dataset_name']}/lr{learning_rate}_bs{batch_size}_neg_num{neg_num}"
    )
    training_path.mkdir(exist_ok=True, parents=True)

    with open(training_path / "training_hyperparameters.json", "w+") as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
        file.close()

    reranker_model, _ , txt_processors = load_model_and_preprocess(
        name=reranker_model_name, model_type="pretrain", is_eval=False, device="cuda"
    )

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print("Square pad preprocess pipeline is used")
    elif transform == "targetpad":
        target_ratio = kwargs["target_ratio"]
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f"Target pad with {target_ratio = } preprocess pipeline is used")
    else:
        raise ValueError(
            "Preprocess transform should be in ['clip', 'squarepad', 'targetpad']"
        )

    train_dataset = IT2IT_section_RerankerDataset(
        knowledge_base_file=kwargs["knowledge_base_file"],
        train_file=kwargs["train_file"],
        negative_db_file=kwargs["negative_db_file"],
        wiki_img_csv_dir = kwargs["wiki_img_csv_dir"],
        wiki_img_path_prefix = kwargs["wiki_img_path_prefix"],
        preprocess=preprocess,
        neg_parrallel_process = process_images_in_parallel,
        use_hard_negative=True,
        neg_num=neg_num,
    )
    print("Training datset length: ", len(train_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        collate_fn=qformer_collate_fn,
        num_workers=kwargs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print("Training dataloader length: ", len(train_dataloader))

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, reranker_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    total_params = sum(p.numel() for p in reranker_model.parameters())
    trainable_params = sum(p.numel() for p in reranker_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}") # 1.17B
    print(f"Trainable parameters: {trainable_params}") # 186M

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start= 0.3,
        div_factor=100.0,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
    )

    scaler = torch.cuda.amp.GradScaler()

    if checkpoint_path and reranker_checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        reranker_model.load_state_dict(reranker_checkpoint_path)
        print(f"reranker model loaded from {reranker_checkpoint_path}")
        print(f"Checkpoint loaded, resuming training from epoch {start_epoch}")
    
    if kwargs["wandb"] == 1:
        wandb.watch(reranker_model, log="all")
    
    training_log_frame = pd.DataFrame()
   
    print("Training loop started")
    for epoch in range(num_epochs):
        train_bar = tqdm(train_dataloader, ncols=150)
        save_step = int(len(train_dataloader) * kwargs["save_frequency"])
        reranker_model.train()
        train_sample_cnt = 0
        for idx, (
            reference_images,
            questions,
            positive_images,
            positive_sections,
            negative_images,
            negative_sections
        ) in enumerate(train_bar):
            reference_img_num = reference_images.size(0)
            train_sample_cnt += reference_img_num
            step = len(train_bar) * epoch + idx + 1
            optimizer.zero_grad()
            reference_images = reference_images.to(device, non_blocking=True)
            positive_images = positive_images.to(device, non_blocking=True)
            

            positive_sections = [
                txt_processors["eval"](positive_section)
                for positive_section in positive_sections
            ]
            negative_sections = [
                [txt_processors["eval"](section) for section in negative_section_list]
                for negative_section_list in negative_sections
            ] 
            questions = [txt_processors["eval"](question) for question in questions]

            with torch.cuda.amp.autocast():

                loss, pos_sim, temp_value = reranker_model(
                    {
                        "reference_images": reference_images,
                        "questions": questions,
                        "positive_images": positive_images,
                        "positive_captions": positive_sections,
                        "negative_images": negative_images,
                        "negative_captions": negative_sections,             
                    }
                )
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_f = round(loss.to("cpu").detach().item(), 3)
            
            bar_content = (
                f"loss: {loss_f}, "
            )
            if kwargs["wandb"] == 1:
                wandb.log({"train_loss": loss_f, "pos_sim": pos_sim, 'temp_value': temp_value})
            train_bar.set_description(desc=f"[{epoch}/{num_epochs}][step {step}] loss: {loss_f}")

            if save_training and (step % save_step == 0) :
                checkpoint = {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(checkpoint, str(training_path / "checkpoint.pth"))
                torch.save(
                    reranker_model.state_dict(),
                    str(
                        training_path / f"model_{step}.pth"
                    ),
                )
                loss_log_dict = {"epoch": epoch, "step": step, "loss": loss_f}
            
                training_log_frame = pd.concat(
                    [training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])]
                )
                training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)
        print(f'Epoch {epoch} finished, total valid training samples: {train_sample_cnt}')
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--num-epochs", default=1, type=int, help="number training epochs"
    )
    parser.add_argument(
        "--reranker-model-name",
        default="qformer_IT2IT_reranker",
        type=str,
        help="lavis modeling registry name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name, should be in ['InfoSeek', 'E-VQA']",
    )
    parser.add_argument(
        "--learning-rate", default=1e-5, type=float, help="Learning rate"
    )
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--neg-num", default=15, type=int, help="negative IT number")
    parser.add_argument(
        "--target-ratio", default=1.25, type=float, help="TargetPad target ratio"
    )
    parser.add_argument(
        "--transform",
        default="targetpad",
        type=str,
        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ",
    )
    parser.add_argument(
        "--save-training",
        dest="save_training",
        action="store_true",
        help="Whether save the training model",
    )
    parser.add_argument(
        "--save_frequency",
        default=0.25,
        type=float,
        help="Save frequency expressed in ration of steps in an epoch",
    )
    parser.add_argument(
        "--checkpoint-path", default=None, type=str, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--reranker-checkpoint-path",
        default=None,
        type=str,
        help="Path to the reranker checkpoint",
    )
    parser.add_argument("--train_file", type=str, help="Path to the training file")
    parser.add_argument(
        "--knowledge_base_file", type=str, help="Path to the knowledge base file"
    )
    parser.add_argument(
        "--negative_db_file", type=str, help="Path to the negative db file"
    )
    parser.add_argument(
        "--wiki_img_csv_dir", default = '../../datasets/wiki_img/full/output/', type=str, help="Path to the wiki image csv directory"
    )
    parser.add_argument(
        "--wiki_img_path_prefix", default = '../../datasets/wiki_img/', type=str, help="Path to the wiki image csv directory"
    )
    parser.add_argument(
        "--wandb", default = 0, type=int, help="whether to use wandb, set your wandb api key in the start of the code"
    )
    

    args = parser.parse_args()
    if args.wandb == 1:
        wandb.init(project="it2it_qformer_section",
            name= f"{args.dataset_name}_lr{args.learning_rate}_bs{args.batch_size}_neg{args.neg_num}_epoch{args.num_epochs}")
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "reranker_model_name": args.reranker_model_name,
        'dataset_name': args.dataset_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "neg_num": args.neg_num,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "checkpoint_path": args.checkpoint_path,
        "reranker_checkpoint_path": args.reranker_checkpoint_path,
        "save_frequency": args.save_frequency,
        "train_file": args.train_file,
        "knowledge_base_file": args.knowledge_base_file,
        "negative_db_file": args.negative_db_file,
        "wiki_img_csv_dir": args.wiki_img_csv_dir,
        "wiki_img_path_prefix": args.wiki_img_path_prefix,
        "wandb": args.wandb,
    }
    print("training_hyper_params: ", training_hyper_params)
    
    IT2IT_reranker_ft(**training_hyper_params)
