import torch
import torch.nn as nn
from transformers import (
  EncoderDecoderModel, AutoConfig, AutoTokenizer, EncoderDecoderConfig, BertForMaskedLM,
  AutoModelForCausalLM
)


def add_tokens(tokenizer):
  tokenizer.add_tokens("[image]")
  tokenizer.add_tokens("[graph]")
  tokenizer.add_tokens("[text]")

  for i in range(32 * 32):
    tokenizer.add_tokens(f"[itoken{i}]")


def load_weights(model, pretrained_model):
    pretrained_model_params = pretrained_model.state_dict()

    for name, param in model.named_parameters():
      if pretrained_model_params.get(name, None) is not None:
        pretrained_weight = pretrained_model_params[name]
        if "word_embeddings" in name:
          param.data[:pretrained_weight.shape[0], :].copy_(pretrained_weight.data)
        elif "position_embeddings" in name:
          size = pretrained_weight.shape[0]
          start = 0
          while start < param.shape[0]:
            end = min(start + size, param.shape[0])
            interval = end - start
            param.data[start:end, :].copy_(pretrained_weight.data[:interval])
            start = end
        elif "decoder.cls.predictions.bias" in name or "cls.predictions.bias" in name:
          param.data[:pretrained_weight.shape[0]].copy_(pretrained_weight.data)
        else:
          param.data.copy_(pretrained_weight.data)
      else:
        print(name)


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # pretrained_model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
    pretrained_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    add_tokens(tokenizer)
    config.max_position_embeddings = 1024 + 2
    config.vocab_size = len(tokenizer.get_vocab())

    # config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
    # model = EncoderDecoderModel(config=config)
    model = BertForMaskedLM(config)
    load_weights(model, pretrained_model)

    model.save_pretrained('bert-base-uncased-itokens')
    tokenizer.save_pretrained('bert-base-uncased-itokens')
    