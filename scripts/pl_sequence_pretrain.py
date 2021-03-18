import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

from sg2im.utils import timeit, bool_flag, LossManager
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.data.vg import SequenceTransformerVgSceneGraphDataset

import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast, BertTokenizer, EncoderDecoderModel, EncoderDecoderConfig, AutoModel
)

from pytorch_lightning.plugins import DDPPlugin

from pl_sequence_train import *


class PretrainedGAN(GAN):
    def training_step(self, batch, batch_idx):
        # sample noise
        # z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        # z = z.type_as(imgs)

        generator_batch = {
            "input_ids": batch["sent_input/input_ids"],
            "attention_mask": batch["sent_input/attention_mask"],
            "decoder_input_ids": batch["code_output/input_ids"],
            "decoder_attention_mask": batch["code_output/attention_mask"],
            "labels": batch["code_output/input_ids"].clone()
        }

        # exlude the loss for padding tokens
        generator_batch["labels"][generator_batch["labels"] == self.tokenizer.pad_token_id] = -100

        # train generator
        outputs = self.generator.forward(**generator_batch)

        image_loss = outputs["loss"]

        predictions = F.gumbel_softmax(outputs["logits"], tau=self.tau, hard=True, dim=-1)

        predictions_embedding = self.generator.apply_word_embeddings(predictions)

        generator_batch = {
            "inputs_embeds": predictions_embedding,
            "attention_mask": batch["code_output/attention_mask"],
            "decoder_input_ids": batch["sent_output/input_ids"],
            "decoder_attention_mask": batch["sent_output/attention_mask"],
            "labels": batch["sent_output/input_ids"].clone()
        }

        generator_batch["labels"][generator_batch["labels"] == self.tokenizer.pad_token_id] = -100

        graph_loss = self.generator.forward_loss(**generator_batch)

        self.log('image_loss', image_loss, prog_bar=True)
        self.log('graph_loss', graph_loss, prog_bar=True)

        return {"loss": image_loss + graph_loss}

    def configure_optimizers(self):
        lr = self.args.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        return [opt_g], []


def main(args):
    backbone = "bert-base-uncased-itokens"
    tokenizer = BertTokenizerFast.from_pretrained(backbone)

    if args.test:
        model = PretrainedGAN.load_from_checkpoint(
            args.load_checkpoint,
            args=args, 
            tokenizer=tokenizer, 
            backbone=backbone
        )
        model.cuda()
        model.eval()

        model.inference(args.scene_graphs_json)
        
        return
    
    # train
    if args.gpus > 1:
        dm = VGDataModule(args, tokenizer, 2)
    else:
        dm = VGDataModule(args, tokenizer)

    if args.load_checkpoint != "":
        model = PretrainedGAN.load_from_checkpoint(
            args.load_checkpoint, 
            args=args, 
            tokenizer=tokenizer, 
            backbone=backbone
        )
    else:
        model = PretrainedGAN(args, tokenizer, backbone)

    training_args = {
        "gpus": args.gpus,
        "fast_dev_run": False,
        "max_steps": args.num_iterations,
        "precision": 32,
        "gradient_clip_val": 1,
    }

    if args.gpus > 1:
        additional_args = {
            "accelerator": "ddp",
            "plugins": [DDPPlugin(find_unused_parameters=True)]
            # "plugins": [my_ddp]
        }

        training_args.update(additional_args)

    trainer = pl.Trainer(**training_args)
    trainer.fit(model, dm)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)