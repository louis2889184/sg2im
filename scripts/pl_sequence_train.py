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
    BertTokenizerFast, 
    BertTokenizer, 
    EncoderDecoderModel, 
    EncoderDecoderConfig, 
    AutoModel,
    BertForSequenceClassification,
)

from pytorch_lightning.plugins import DDPPlugin


VG_DIR = os.path.expanduser('datasets/vg')
COCO_DIR = os.path.expanduser('datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--load_checkpoint', default="")

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--gpus', default=1, type=int)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)
parser.add_argument('--max_lengths_for_image', default=1024, type=int)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight 
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)


class VGDataModule(pl.LightningDataModule):

    def __init__(self, args, tokenizer, num_workers=8):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = args.batch_size

    def setup(self, stage=None):
        args = self.args
        with open(args.vocab_json, 'r') as f:
            vocab = json.load(f)
        dset_kwargs = {
            'vocab': vocab,
            'h5_path': args.train_h5,
            'image_dir': args.vg_image_dir,
            'image_size': args.image_size,
            'max_samples': args.num_train_samples,
            'max_objects': args.max_objects_per_image,
            'use_orphaned_objects': args.vg_use_orphaned_objects,
            'include_relationships': args.include_relationships,
            'max_lengths_for_image': args.max_lengths_for_image
        }
        train_dset = SequenceTransformerVgSceneGraphDataset(
            **dset_kwargs, tokenizer=self.tokenizer
        )
        # iter_per_epoch = len(train_dset) // args.batch_size
        # print('There are %d iterations per epoch' % iter_per_epoch)

        dset_kwargs['h5_path'] = args.val_h5
        del dset_kwargs['max_samples']

        val_dset = SequenceTransformerVgSceneGraphDataset(
            **dset_kwargs, tokenizer=self.tokenizer
        )
        self.train_dset = train_dset
        self.val_dset = val_dset

    def train_dataloader(self):
        return DataLoader(
            self.train_dset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=self.num_workers)


class Discriminator(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = BertForSequenceClassification.from_pretrained(backbone)
        
    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)

        return outputs["loss"]

    def apply_word_embeddings(self, inputs):
        """
        Because Gumbel softmax outputs cannot directly feed to huggingface model,
        we have to compute the `input_embed` manually.
        """
        word_embeddings = self.backbone.bert.embeddings.word_embeddings

        return torch.matmul(inputs, word_embeddings.weight)


class Generator(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = EncoderDecoderModel.from_encoder_decoder_pretrained(
            backbone, backbone, tie_encoder_decoder=True
        )

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def forward_logits(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)["logits"]

    def forward_loss(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)["loss"]

    def apply_word_embeddings(self, inputs):
        """
        Because Gumbel softmax outputs cannot directly feed to huggingface model,
        we have to compute the `input_embed` manually.
        """
        word_embeddings = self.backbone.encoder.embeddings.word_embeddings

        return torch.matmul(inputs, word_embeddings.weight)


class GAN(pl.LightningModule):

    def __init__(
        self,
        args,
        tokenizer,
        backbone=None,
    ):
        super().__init__()

        self.args = args

        self.validation_z = torch.randn(8, 100)
        self.tokenizer = tokenizer
        self.discriminator = Discriminator(backbone)
        self.generator = Generator(backbone)

        self.graph_special_token = "[graph]"
        self.image_special_token = "[image]"

        self.tau = 1

        self.image_token_id_list, self.text_token_id_list = self.retrieve_bad_image_text_tokens_ids()

    def retrieve_bad_image_text_tokens_ids(self):
        special_tokens_list = ["[CLS]", "[SEP]"]
        image_tokens_list = [f"[itoken{i}]" for i in range(512)]
        extra_image_tokens_list = [f"[itoken{i}]" for i in range(512, 32 * 32)]
        
        vocab = self.tokenizer.get_vocab()

        special_tokens_id_list = [vocab[token] for token in special_tokens_list]
        image_token_id_list = [vocab[token] for token in image_tokens_list]
        extra_image_tokens_id_list = [vocab[token] for token in extra_image_tokens_list]
        text_token_id_list = [v for k, v in vocab.items()]

        text_token_id_list = \
            list(set(text_token_id_list) - set(image_token_id_list) - set(extra_image_tokens_id_list))

        return image_token_id_list + extra_image_tokens_id_list, text_token_id_list + extra_image_tokens_id_list

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        if optimizer_idx == 0:
            logits = self.generator.forward_logits(**generator_batch)

            predictions = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop

            predictions_embedding = self.generator.apply_word_embeddings(predictions)

            fake_batch = {
                "inputs_embeds": predictions_embedding,
                "attention_mask": batch["code_output/attention_mask"],
                "decoder_input_ids": batch["sent_output/input_ids"],
                "decoder_attention_mask": batch["sent_output/attention_mask"],
                "labels": batch["sent_output/input_ids"].clone()
            }

            fake_batch["labels"][fake_batch["labels"] == self.tokenizer.pad_token_id] = -100

            ac_loss = self.generator.forward_loss(**fake_batch)

            predictions_embedding = self.discriminator.apply_word_embeddings(predictions)

            fake_dis_batch = {
                "inputs_embeds": predictions_embedding,
                "attention_mask": batch["code_output/attention_mask"],
                "labels": torch.ones(predictions_embedding.shape[0]).type_as(predictions_embedding).long()
            }

            g_d_loss = self.discriminator(**fake_dis_batch)

            g_loss = g_d_loss + ac_loss
            # g_loss = ac_loss

            self.log('g_ac_loss', ac_loss, prog_bar=True)
            self.log('g_d_loss', g_d_loss, prog_bar=True)

            # return {"loss": g_loss}
        # train discriminator (inverse generator)
        # if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            logits = self.generator.forward_logits(**generator_batch)

            predictions = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

            # don't compute the gradients of the generator
            predictions = predictions.detach()

            predictions_embedding = self.generator.apply_word_embeddings(predictions)

            fake_batch = {
                "inputs_embeds": predictions_embedding,
                "attention_mask": batch["code_output/attention_mask"],
                "decoder_input_ids": batch["sent_output/input_ids"],
                "decoder_attention_mask": batch["sent_output/attention_mask"],
                "labels": batch["sent_output/input_ids"].clone()
            }

            fake_batch["labels"][fake_batch["labels"] == self.tokenizer.pad_token_id] = -100

            fake_ac_loss = self.generator.forward_loss(**fake_batch)

            # For real data
            real_batch = {
                "input_ids": batch["code_output/input_ids"],
                "attention_mask": batch["code_output/attention_mask"],
                "decoder_input_ids": batch["sent_output/input_ids"],
                "decoder_attention_mask": batch["sent_output/attention_mask"],
                "labels": batch["sent_output/input_ids"].clone()
            }

            real_batch["labels"][real_batch["labels"] == self.tokenizer.pad_token_id] = -100

            real_ac_loss = self.generator.forward_loss(**real_batch)

            ac_loss = (real_ac_loss + fake_ac_loss) / 2

            self.log('ac_loss', ac_loss, prog_bar=True)
            # return {"loss": ac_loss}
            return g_loss + ac_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            logits = self.generator.forward_logits(**generator_batch)

            # don't compute the gradients of the generator
            predictions = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

            predictions_embedding = self.discriminator.apply_word_embeddings(predictions)

            fake_dis_batch = {
                "inputs_embeds": predictions_embedding,
                "attention_mask": batch["code_output/attention_mask"],
                "labels": torch.zeros(predictions.shape[0]).type_as(predictions).long()
            }

            fake_loss = self.discriminator(**fake_dis_batch)

            # fake = torch.zeros(fake_preds.shape)
            # fake = fake.type_as(fake_preds)

            # fake_loss = self.adversarial_loss(fake_preds, fake)

            real_dis_batch = {
                "input_ids": batch["code_output/input_ids"],
                "attention_mask": batch["code_output/attention_mask"],
                "labels": torch.ones(predictions.shape[0]).type_as(predictions).long()
            }

            real_loss = self.discriminator(**real_dis_batch)

            # real = torch.ones(real_preds.shape)
            # real = real.type_as(real_preds)

            # real_loss = self.adversarial_loss(real_preds, real)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            self.log('d_loss', d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.args.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )
        return [opt_g, opt_d], []

    # def on_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        pass

    def inference(self, scene_graphs_json):
        scene_graphs = self.read_scene_graphs(scene_graphs_json)

        image_tokens_generation = self.generator.backbone.generate(
            scene_graphs["input_ids"], 
            max_length=66, 
            # num_beams=5, 
            # no_repeat_ngram_size=2, 
            # early_stopping=True,
            do_sample=True,
            top_p=0.92, 
            top_k=0,
            decoder_start_token_id=self.generator.backbone.config.decoder.pad_token_id,
            bad_words_ids=[[ids] for ids in self.text_token_id_list],
        )

        print(image_tokens_generation)

        output = []

        for data in image_tokens_generation:
            output.append(self.tokenizer.decode(data, skip_special_tokens=True))
            print(output[-1])

        reconstructed_graph = self.generator.backbone.generate(
            image_tokens_generation, 
            max_length=64, 
            # num_beams=5, 
            # no_repeat_ngram_size=2, 
            # early_stopping=True,
            do_sample=True,
            top_p=0.92, 
            top_k=0,
            decoder_start_token_id=self.generator.backbone.config.decoder.pad_token_id,
            bad_words_ids=[[ids]for ids in self.image_token_id_list],
        )

        for data in reconstructed_graph:
            print(self.tokenizer.decode(data, skip_special_tokens=True))

        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        itokens_output_file = os.path.join(self.args.output_dir, "itokens_output.json")

        with open(itokens_output_file, "w") as f:
            json.dump(output, f, indent=2)

    def read_scene_graphs(self, scene_graphs_json):
        with open(scene_graphs_json, 'r') as f:
            scene_graphs = json.load(f)

        if isinstance(scene_graphs, dict):
            # We just got a single scene graph, so promote it to a list
            scene_graphs = [scene_graphs]

        objs, triples, obj_to_img = [], [], []
        obj_offset = 0
        sents_list = []
        for i, sg in enumerate(scene_graphs):
            # Insert dummy __image__ object and __in_image__ relationships
            sents = []
            for s, p, o in sg['relationships']:
                sent = f"{sg['objects'][s]} {p} {sg['objects'][o]}."
                sents.append(sent)

            sent = " ".join(sents)
            sent = f"{self.graph_special_token} {sent} {self.image_special_token}"

            sents_list.append(sent)

            print(sent)
        
        sent_tensor = self.tokenizer(
            sents_list, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=64, 
            truncation=True,
            add_special_tokens=False
        )

        device = next(self.parameters()).device
        sent_tensor = {k: v.to(device) for k, v in sent_tensor.items()}

        return sent_tensor


def main(args):
    backbone = "bert-base-uncased-itokens"
    tokenizer = BertTokenizerFast.from_pretrained(backbone)

    # encoder_decoder_config = EncoderDecoderConfig.from_pretrained("bert-base-uncased-itokens")
    # model = EncoderDecoderModel.from_pretrained(
    #     "bert-base-uncased-itokens", config=encoder_decoder_config
    # )

    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    #     "bert-base-uncased-itokens", "bert-base-uncased-itokens", tie_encoder_decoder=True
    # )

    # generator = Generator(model)

    # discriminator = Discriminator(
    #     AutoModel.from_pretrained("bert-base-uncased-itokens")
    # )

    if args.test:
        model = GAN.load_from_checkpoint(
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
        model = GAN.load_from_checkpoint(
            args.load_checkpoint, 
            args=args, 
            tokenizer=tokenizer, 
            backbone=backbone
        )
    else:
        model = GAN(args, tokenizer, backbone)

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