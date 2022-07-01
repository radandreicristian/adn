from adn.attentions.consts import FULL, FAVORPLUS, EFFICIENT, LSH, LINEAR, GROUP, LINFORMER
from adn.attentions.efficient import EfficientSelfAttention
from adn.attentions.favorplus import FavorPlusAttention
from adn.attentions.full import MultiHeadAttention
from adn.attentions.linear import LinearAttention
from adn.attentions.lsh import LshSelfAttention
from adn.attentions.group import GroupAttention
from adn.attentions.linformer import LinformerAttention
import torch.nn as nn


class AttentionFactory:

    @classmethod
    def build_attention(cls, attention_type, **kwargs) -> nn.Module:
        if attention_type == FULL:
            return MultiHeadAttention(**kwargs)
        elif attention_type == FAVORPLUS:
            return FavorPlusAttention(**kwargs)
        elif attention_type == EFFICIENT:
            return EfficientSelfAttention(**kwargs)
        elif attention_type == LSH:
            return LshSelfAttention(**kwargs)
        elif attention_type == LINEAR:
            return LinearAttention(**kwargs)
        elif attention_type == GROUP:
            return GroupAttention(**kwargs)
        elif attention_type == LINFORMER:
            return LinformerAttention(**kwargs)
        else:
            raise ValueError("Invalid attention type.")
