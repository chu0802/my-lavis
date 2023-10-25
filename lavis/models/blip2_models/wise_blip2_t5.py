import torch
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_t5 import Blip2T5
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy
from dataclasses import dataclass


def _merge(theta0, theta1, alpha=0.5):
    if isinstance(theta0, nn.Parameter):
        return nn.parameter.Parameter(alpha * theta0 + (1 - alpha) * theta1)

    return {
        key: alpha * theta0[key] + (1 - alpha) * theta1[key] for key in theta0.keys()
    }


def _extract_model_state_dict(model):
    if isinstance(model, nn.Module):
        return model.state_dict()
    # if model has been a state dict, don't do anything
    return model


def merge_model(m0, m1, alpha):
    param0 = {k: v.clone() for k, v in _extract_model_state_dict(m0).items()}
    param1 = {k: v.clone() for k, v in _extract_model_state_dict(m1).items()}
    return _merge(param0, param1, alpha)


@dataclass
class QformerParams:
    query_tokens: nn.Parameter
    qformer_state_dict: dict
    t5_proj_state_dict: dict


@registry.register_model("wise_blip2_t5")
class WiseBlip2T5(Blip2T5):
    @classmethod
    def from_config(cls, cfg):
        load_finetuned = cfg.get("load_finetuned", False)

        pre_trained_cfg = copy.deepcopy(cfg)
        pre_trained_cfg.load_finetuned = False
        pre_trained_cfg.finetuned = None

        pre_trained_model = Blip2T5.from_config(pre_trained_cfg)

        pre_trained_parameters = QformerParams(
            query_tokens=pre_trained_model.query_tokens,
            qformer_state_dict=pre_trained_model.Qformer.state_dict(),
            t5_proj_state_dict=pre_trained_model.t5_proj.state_dict(),
        )

        del pre_trained_model

        fine_tuned_model = Blip2T5.from_config(cfg)

        alpha = cfg.get("alpha", 0.5)

        fine_tuned_model.query_tokens = _merge(
            fine_tuned_model.query_tokens, pre_trained_parameters.query_tokens, alpha
        )

        fine_tuned_model.Qformer.load_state_dict(
            merge_model(
                fine_tuned_model.Qformer,
                pre_trained_parameters.qformer_state_dict,
                alpha,
            )
        )

        fine_tuned_model.t5_proj.load_state_dict(
            merge_model(
                fine_tuned_model.t5_proj,
                pre_trained_parameters.t5_proj_state_dict,
                alpha,
            )
        )

        return fine_tuned_model

        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # img_size = cfg.get("image_size")
        # num_query_token = cfg.get("num_query_token")
        # t5_model = cfg.get("t5_model")

        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)
        # freeze_qformer = cfg.get("freeze_qformer", False)

        # prompt = cfg.get("prompt", "")
        # max_txt_len = cfg.get("max_txt_len", 32)

        # apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # model = cls(
        #     vit_model=vit_model,
        #     img_size=img_size,
        #     drop_path_rate=drop_path_rate,
        #     use_grad_checkpoint=use_grad_checkpoint,
        #     vit_precision=vit_precision,
        #     freeze_vit=freeze_vit,
        #     freeze_qformer=freeze_qformer,
        #     num_query_token=num_query_token,
        #     t5_model=t5_model,
        #     prompt=prompt,
        #     max_txt_len=max_txt_len,
        #     apply_lemmatizer=apply_lemmatizer,
        # )
        # model.load_checkpoint_from_config(cfg)

        # return model
