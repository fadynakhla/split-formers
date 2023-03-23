from typing import Any, Optional, Tuple, Union
import torch
import transformers
from transformers import modeling_outputs


class ModelParallelT5Config(transformers.T5Config):
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 0.000001,
        initializer_factor=1,
        feed_forward_proj: str = "relu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        num_devices: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            vocab_size,
            d_model,
            d_kv,
            d_ff,
            num_layers,
            num_decoder_layers,
            num_heads,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            dropout_rate,
            layer_norm_epsilon,
            initializer_factor,
            feed_forward_proj,
            is_encoder_decoder,
            use_cache,
            pad_token_id,
            eos_token_id,
            **kwargs,
        )
        self.num_devices = num_devices


class ModelParallelT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config: ModelParallelT5Config) -> None:
        super().__init__(config)
        self.config = config
        self.model_parallel = self.config.num_devices and self.config.num_devices > 1
        if self.model_parallel:
            self.devices = [
                torch.device(f"cuda:{i}") for i in range(self.config.num_devices)
            ]
            self.num_layers_per_device = len(self.encoder.block) // len(self.devices)
            for i, device in enumerate(self.devices):
                start_layer = i * self.num_layers_per_device
                end_layer = (i + 1) * self.num_layers_per_device
                self.encoder.block[start_layer:end_layer].to(device)
                self.decoder.block[start_layer:end_layer].to(device)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], modeling_outputs.Seq2SeqLMOutput]:
        if not self.model_parallel:
            return super().forward(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                head_mask,
                decoder_head_mask,
                cross_attn_head_mask,
                encoder_outputs,
                past_key_values,
                inputs_embeds,
                decoder_inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = input_ids
            # Convert encoder inputs in embeddings if needed
            for i, _ in enumerate(self.devices):
                start_layer = i * self.num_layers_per_device
                end_layer = (i + 1) * self.num_layers_per_device
                to_device = self.devices[(i + 1) % len(self.devices)]
                # Move the necessary tensors to the correct device
                encoder_outputs = self.encoder.block[start_layer:end_layer](
                    input_ids=encoder_outputs,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0].to(to_device)
                attention_mask = attention_mask.to(to_device)

        elif return_dict and not isinstance(
            encoder_outputs, modeling_outputs.BaseModelOutput
        ):
            encoder_outputs = modeling_outputs.BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        for i, _ in enumerate(self.devices):
            start_layer = i * self.num_layers_per_device
            end_layer = (i + 1) * self.num_layers_per_device
            decoder_outputs = decoder_input_ids
            decoder_outputs = self.decoder.block[start_layer:end_layer](
                input_ids=decoder_outputs,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_value=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0].to(self.devices[(i + 1) % len(self.devices)])
