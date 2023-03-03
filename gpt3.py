import collections
import math
import os.path
from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions


class GPT3Config(PretrainedConfig):
    model_type = 'gpt3'

    def __init__(
            self,
            vocab_size=25600,
            hidden_size=768,
            ffn_hidden_size=None,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,
            type_vocab_size=2,
            layer_norm_epsilon=1e-12,
            bias_gelu_fusion=True,
            sequence_parallel=False,
            apply_query_key_layer_scaling=True,
            kv_channels=None,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            bias_dropout_fusion=True,
            apply_residual_connection_post_layer_norm=False,
            hidden_dropout=0.1,
            initializer_range=0.02,
            **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = 4 * hidden_size \
            if ffn_hidden_size is None else ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.bias_gelu_fusion = bias_gelu_fusion
        self.sequence_parallel = sequence_parallel

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        if kv_channels is None:
            assert hidden_size % num_attention_heads == 0
            self.kv_channels = hidden_size // num_attention_heads
        self.masked_softmax_fusion = masked_softmax_fusion
        self.attention_dropout = attention_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.apply_residual_connection_post_layer_norm = \
            apply_residual_connection_post_layer_norm
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range


def convert_damo_to_ours(model_path, save_path):
    from transformers import BertTokenizer

    filename = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = GPT3Config.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    mappings = {
        "language_model.word_embeddings.weight": "transformer.word_embeddings.weight",
        "language_model.position_embeddings.weight": "transformer.position_embeddings.weight",
    }
    state_dict = torch.load(filename, map_location='cpu')
    new_state_dict = collections.OrderedDict()

    for k, v in state_dict.items():

        if k.startswith("model."):
            k = k[6:]
        if k in mappings:
            new_k = mappings[k]
        elif "language_model.transformer.layers." in k:
            new_k = k.replace("language_model.transformer.layers.", "transformer.h.")
        elif "language_model.transformer.final_layernorm" in k:
            new_k = k.replace("language_model.transformer.final_layernorm", "transformer.final_layernorm")
        else:
            new_k = None
            # print(k)
        if new_k is not None:
            new_state_dict[new_k] = v
    new_state_dict["lm_head.weight"] = new_state_dict["transformer.word_embeddings.weight"]

    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(new_state_dict, os.path.join(save_path, "pytorch_model.bin"))


def _make_causal_mask(
        input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _prepare_attn_mask(
        attention_mask: torch.Tensor, input_shape: Tuple[int, int],
) -> torch.BoolTensor:
    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    if src_length > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, device=device, past_key_values_length=0
        )

    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


class GPT3SelfAttention(nn.Module):

    def __init__(self, config: GPT3Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # Per attention head
        self.hidden_size_per_attention_head = \
            self.hidden_size // self.num_attention_heads

        self.query_key_value = nn.Linear(self.hidden_size,
                                         3 * self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(
            config.attention_probs_dropout_prob)

        # Output.
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _split_heads(self,
                     tensor, num_partitions,
                     contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
        return tensor_list

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor],
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False, ):

        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv, 3)

        query_layer = self._transpose_for_scores(query_layer)
        key_layer = self._transpose_for_scores(key_layer)
        value_layer = self._transpose_for_scores(value_layer)

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)

        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)

        # Attention probabilities. [b, np, s, s]
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output_tensor = self.dense(context_layer)
        output_tensor = self.output_dropout(output_tensor)

        outputs = (output_tensor,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class GPT3MLP(nn.Module):
    def __init__(self, config: GPT3Config):
        super().__init__()

        hidden_size = config.hidden_size
        # Project to 4h.
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.activation_func = F.gelu
        # Project back to h.
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT3Block(nn.Module):

    def __init__(self, config: GPT3Config):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)

        # Self attention.
        self.attention = GPT3SelfAttention(config)

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)

        # MLP
        self.mlp = GPT3MLP(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                output_attentions: bool = False):
        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attn_outputs = self.attention(
            layernorm_output, attention_mask, output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output
        outputs = (output,) + outputs[1:]
        return outputs


class GPT3PreTrainedModel(PreTrainedModel, ABC):
    config_class = GPT3Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT3Block"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, GPT3Model):
            module.gradient_checkpointing = value


class GPT3Model(GPT3PreTrainedModel):
    def __init__(self, config: GPT3Config):
        super().__init__(config)
        # Embeddings.
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer.
        self.h = nn.ModuleList([GPT3Block(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        causal_mask = _prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
        )
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    causal_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    causal_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        # Add last hidden state
        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPT3ForCausalLM(GPT3PreTrainedModel, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: GPT3Config):
        super().__init__(config)
        self.transformer = GPT3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> dict:

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
