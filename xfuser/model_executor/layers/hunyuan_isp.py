import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoTransformerBlock,
    HunyuanVideoSingleTransformerBlock,
)
import functools
from typing import Optional, Tuple
import torch.distributed as dist
import torch.nn.functional as F


COMMUNICATION_STREAM = torch.cuda.Stream()

def all_gather(x, gather_dim, process_group=None, no_cat: bool = False, stream=None):
    if stream is not None:
        with torch.cuda.stream(stream):
            tensor_list = [torch.empty_like(x) for _ in range(dist.get_world_size(process_group))]
            dist.all_gather(tensor_list=tensor_list, tensor=x, group=process_group)
    else:
        tensor_list = [torch.empty_like(x) for _ in range(dist.get_world_size(process_group))]
        dist.all_gather(tensor_list=tensor_list, tensor=x, group=process_group)

    if no_cat:
        return tensor_list
    else:
        output = torch.cat(tensor_list, dim=gather_dim)
        return output



def modify_transformer_block(block: HunyuanVideoTransformerBlock):
    @functools.wraps(block.__class__.forward)
    def new_forward(
        self, 
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ):
        # hidden states are of shape [B, S, HD] e.g. [1, 5200, 3072]
        # encoder hidden states are of shape [B, S_text, HD] e.g. [1, 256, 3072]
        ws = dist.get_world_size()
        rank = dist.get_rank()
        global_hidden_states_len = hidden_states.shape[1]
        global_encoder_hidden_states_len = encoder_hidden_states.shape[1]
        sub_seq_len_per_device = (global_hidden_states_len + global_encoder_hidden_states_len) // ws
        assert sub_seq_len_per_device * ws == global_hidden_states_len + global_encoder_hidden_states_len, \
            f"The combined sequence length {global_hidden_states_len + global_encoder_hidden_states_len} is not divisible by the world size {ws}"

        # =====================
        # 1. Input normalization
        # =====================
        # norm_hidden_states: [B, S, HD]
        # gate_msa: [B, HD]
        # shift_mlp: [B, HD]
        # scale_mlp: [B, HD]
        # gate_mlp: [B, HD]
        # norm_encoder_hidden_states: [B, S_text, HD]
        # c_gate_msa: [B, HD]
        # c_shift_mlp: [B, HD]
        # c_scale_mlp: [B, HD]
        # c_gate_mlp: [B, HD]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # ===================== 
        # 2. Joint attention
        # =====================
        # original code
        # attn_output, context_attn_output = self.attn(
        #     hidden_states=norm_hidden_states,
        #     encoder_hidden_states=norm_encoder_hidden_states,
        #     attention_mask=attention_mask,
        #     image_rotary_emb=freqs_cis,
        # )

        if self.attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # get the current partition of hidden states
        start = rank * sub_seq_len_per_device
        length = min(sub_seq_len_per_device, global_hidden_states_len - start)
        sub_hidden_states = hidden_states.narrow(dim=1, start=start, length=length).contiguous()

        # 2.1. QKV projections
        query = self.attn.to_q(sub_hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)

        # q, k, and v are of shape [B, S, H, D] -> [B, H, S, D]
        query = query.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)

        # 2.2 QK normalization
        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)


        # 2.3. Rotational positional embeddings applied to latent stream
        image_rotary_emb = freqs_cis
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if self.attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                # for query, we only apply the rotary embeddings to the current partition
                chunked_image_rotary_emb = [
                    val.narrow(dim=0, start=start, length=length).contiguous()
                    for val in image_rotary_emb
                ]
                query = apply_rotary_emb(query, chunked_image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 2.4. Encoder condition QKV projection and normalization
        if self.attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = self.attn.add_q_proj(encoder_hidden_states)
            encoder_key = self.attn.add_k_proj(encoder_hidden_states)
            encoder_value = self.attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)

            if self.attn.norm_added_q is not None:
                encoder_query = self.attn.norm_added_q(encoder_query)
            if self.attn.norm_added_k is not None:
                encoder_key = self.attn.norm_added_k(encoder_key)

            # NOTE: we only do the computation on the last rank since the encoder hidden states are at the end of the concatenated sequence
            if rank == ws - 1:
                query = torch.cat([query, encoder_query], dim=2)

            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        
        output_list = []
        chunk_length = query.shape[2] // ws

        for chunk_idx, sub_q_chunk in enumerate(query.chunk(ws, dim=2)):
            # 2.5. Attention
            # calculate the start and end index with respect to the global sequence
            # slice out the corresponding attention mask
            sub_q_chunk_start = rank * sub_seq_len_per_device + chunk_idx * chunk_length
            sub_q_chunk_end = sub_q_chunk_start + chunk_length
            sub_attention_mask = attention_mask.narrow(dim=1, start=sub_q_chunk_start, length=chunk_length).contiguous()

            # perform attention
            attn_output_chunk = F.scaled_dot_product_attention(
                sub_q_chunk, key, value, attn_mask=sub_attention_mask, dropout_p=0.0, is_causal=False
            )
            attn_output_chunk = attn_output_chunk.transpose(1, 2).flatten(2, 3)
            attn_output_chunk = attn_output_chunk.to(query.dtype)

            # 2.6. Output projection
            if encoder_hidden_states is not None:
                if sub_q_chunk_end < global_hidden_states_len:
                    # this chunk is completely hidden states
                    attn_output_chunk = attn_output_chunk
                    encoder_hidden_states_chunk = None
                    sub_hidden_states_chunk_start = sub_q_chunk_start
                    sub_hidden_states_chunk_end = sub_q_chunk_end
                    sub_encoder_hidden_states_chunk_start = -1
                    sub_encoder_hidden_states_chunk_end = -1

                elif sub_q_chunk_start >= global_hidden_states_len:
                    # this chunk is completely encoder hidden states
                    attn_output_chunk = None
                    encoder_hidden_states_chunk = attn_output_chunk
                    sub_hidden_states_chunk_start = -1
                    sub_hidden_states_chunk_end = -1
                    sub_encoder_hidden_states_chunk_start = sub_q_chunk_start - global_hidden_states_len
                    sub_encoder_hidden_states_chunk_end = sub_q_chunk_end - global_hidden_states_len
                else:
                    encoder_hidden_states_split_len = sub_q_chunk_end - global_hidden_states_len
                    # this chunk is a mixture of hidden states and encoder hidden states
                    attn_output_chunk = attn_output_chunk[:, :-encoder_hidden_states_split_len]
                    encoder_hidden_states_chunk = attn_output_chunk[:, -encoder_hidden_states_split_len:]
                    sub_hidden_states_chunk_start = sub_q_chunk_start
                    sub_hidden_states_chunk_end = sub_q_chunk_end - encoder_hidden_states_split_len
                    sub_encoder_hidden_states_chunk_start = 0
                    sub_encoder_hidden_states_chunk_end = encoder_hidden_states_split_len

                if getattr(self.attn, "to_out", None) is not None and attn_output_chunk is not None:
                    attn_output_chunk = self.attn.to_out[0](attn_output_chunk)
                    attn_output_chunk = self.attn.to_out[1](attn_output_chunk)

                if getattr(self.attn, "to_add_out", None) is not None and encoder_hidden_states_chunk is not None:
                    encoder_hidden_states_chunk = self.attn.to_add_out(encoder_hidden_states_chunk)

            # =====================
            # 3. Modulation and residual connection and FFN
            # =====================
            chunk_list = []
            if attn_output_chunk is not None:
                # get the current partition of the hidden states
                sub_hidden_states_chunk = hidden_states.narrow(dim=1, start=sub_hidden_states_chunk_start, length=sub_hidden_states_chunk_end - sub_hidden_states_chunk_start)
                sub_hidden_states_chunk = sub_hidden_states_chunk + attn_output_chunk * gate_msa.unsqueeze(1)
                sub_norm_hidden_states_chunk = self.norm2(sub_hidden_states_chunk)
                sub_norm_hidden_states_chunk = sub_norm_hidden_states_chunk * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                sub_ff_output = self.ff(sub_norm_hidden_states_chunk)
                sub_hidden_states_chunk = sub_hidden_states_chunk + gate_mlp.unsqueeze(1) * sub_ff_output
                chunk_list.append(sub_hidden_states_chunk)
            
            if encoder_hidden_states_chunk is not None:
                sub_encoder_hidden_states_chunk = encoder_hidden_states.narrow(dim=1, start=sub_encoder_hidden_states_chunk_start, length=sub_encoder_hidden_states_chunk_end - sub_encoder_hidden_states_chunk_start)
                sub_encoder_hidden_states = sub_encoder_hidden_states_chunk + encoder_hidden_states_chunk * c_gate_msa.unsqueeze(1)
                sub_norm_encoder_hidden_states_chunk = self.norm2_context(sub_encoder_hidden_states)
                sub_norm_encoder_hidden_states_chunk = sub_norm_encoder_hidden_states_chunk * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                sub_context_ff_output = self.ff_context(sub_norm_encoder_hidden_states_chunk)
                sub_encoder_hidden_states_chunk = sub_encoder_hidden_states_chunk + c_gate_mlp.unsqueeze(1) * sub_context_ff_output
                chunk_list.append(sub_encoder_hidden_states_chunk)

            # ISP: run all gather here
            assert len(chunk_list) != 0
            if len(chunk_list) == 2:
                merged_chunk = torch.cat(chunk_list, dim=1).contiguous()
            else:
                merged_chunk = chunk_list[0].contiguous()
            COMMUNICATION_STREAM.wait_stream(torch.cuda.current_stream())
            output_list.extend(all_gather(merged_chunk, gather_dim=1, no_cat=True, stream=COMMUNICATION_STREAM))

        
        torch.cuda.current_stream().wait_stream(COMMUNICATION_STREAM)

        # order the output list
        # the output list is in list [1, 2, 3, 4, 1, 2, 3, 4, ...], order it to be [1, 1, 2, 2, 3, 3, 4, 4, ...]
        sorted_output_list = []
        for i in range(ws):
            for j in range(i, len(output_list), ws):
                sorted_output_list.append(output_list[j])
        hidden_states = torch.cat(sorted_output_list, dim=1)

        # split
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-global_encoder_hidden_states_len, :],
            hidden_states[:, -global_encoder_hidden_states_len:, :],
        )

        return hidden_states, encoder_hidden_states

    new_forward = new_forward.__get__(block)
    block.forward = new_forward




def modify_single_transformer_block(block: HunyuanVideoSingleTransformerBlock):
    @functools.wraps(block.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # ========================
        # some global variables
        ws = dist.get_world_size()
        rank = dist.get_rank()
        global_hidden_states_len = hidden_states.shape[1]
        global_encoder_hidden_states_len = encoder_hidden_states.shape[1]
        sub_seq_len_per_device = (global_hidden_states_len + global_encoder_hidden_states_len) // ws
        global_sequence_start = rank * sub_seq_len_per_device
        global_sequence_end = global_sequence_start + sub_seq_len_per_device
        assert sub_seq_len_per_device * ws == global_hidden_states_len + global_encoder_hidden_states_len, \
            f"The combined sequence length {global_hidden_states_len + global_encoder_hidden_states_len} is not divisible by the world size {ws}"

        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = hidden_states.chunk(ws, dim=1)[rank].contiguous()

        # ========================
        # 1. Input normalization
        # ========================
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        sub_norm_hidden_states = norm_hidden_states.chunk(ws, dim=1)[rank].contiguous()
        sub_mlp_hidden_states = self.act_mlp(self.proj_mlp(sub_norm_hidden_states))

        # ========================
        # 2. Attention
        # ========================
        # norm_hidden_states, norm_encoder_hidden_states = (
        #     norm_hidden_states[:, :-text_seq_length, :],
        #     norm_hidden_states[:, -text_seq_length:, :],
        # )
        # attn_output, context_attn_output = self.attn(
        #     hidden_states=norm_hidden_states,
        #     encoder_hidden_states=norm_encoder_hidden_states,
        #     attention_mask=attention_mask,
        #     image_rotary_emb=image_rotary_emb,
        # )

        # simply the code by removing some if-else branches
        # so that we make assertion here to make sure the inputs are consistent
        assert self.attn.add_q_proj is None and encoder_hidden_states is not None
        

        # 1. QKV projections
        query = self.attn.to_q(sub_norm_hidden_states)
        key = self.attn.to_k(norm_hidden_states)
        value = self.attn.to_v(norm_hidden_states)

        query = query.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            # because query is a partition of the whole sequence, we need to apply the rotary
            # embedding on different segments
            if global_sequence_end < global_hidden_states_len:
                # case 1: query contains only hidden states
                sub_image_rotary_emb = [
                    val.narrow(dim=0, start=global_sequence_start, length=global_sequence_end - global_sequence_start).contiguous()
                    for val in image_rotary_emb
                ]
                query = apply_rotary_emb(query, sub_image_rotary_emb)
            elif global_sequence_start >= global_hidden_states_len:
                # case 2: query contains only encoder hidden states
                query = query
            else:
                # case 3: query contains both hidden states and encoder hidden states
                sub_image_rotary_emb = [
                    val.narrow(
                        dim=0, 
                        start=global_sequence_start,
                        length=global_hidden_states_len - global_sequence_start
                    ).contiguous()
                    for val in image_rotary_emb
                ]
                encoder_segment_len = global_sequence_end - global_hidden_states_len
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_segment_len], sub_image_rotary_emb),
                        query[:, :, -encoder_segment_len :],
                    ],
                    dim=2,
                )

            # we can keep key the same
            key = torch.cat(
                [
                    apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                    key[:, :, -encoder_hidden_states.shape[1] :],
                ],
                dim=2,
            )

        # 5. Attention
        output_list = []
        for chunk_idx, (sub_query_chunk, sub_mlp_chunk, sub_residual) in enumerate(zip(query.chunk(ws, dim=2), sub_mlp_hidden_states.chunk(ws, dim=1), residual.chunk(ws, dim=1))):
            # compute attention
            global_chunk_start = global_sequence_start + chunk_idx * sub_seq_len_per_device // ws
            chunk_size = sub_query_chunk.shape[2]
            sub_attention_mask = attention_mask.narrow(dim=1, start=global_chunk_start, length=chunk_size).contiguous()
            sub_attn_output = F.scaled_dot_product_attention(
                sub_query_chunk, key, value, attn_mask=sub_attention_mask, dropout_p=0.0, is_causal=False
            )
            sub_attn_output = sub_attn_output.transpose(1, 2).flatten(2, 3)
            sub_attn_output = sub_attn_output.to(query.dtype)

            # 6. Output projection
            to_be_merged = []
            if encoder_hidden_states is not None:
                if global_sequence_end < global_hidden_states_len:
                    # case 1: attn_output contains only hidden states
                    sub_attn_output, sub_encoder_hidden_states = sub_attn_output, None
                elif global_sequence_start >= global_hidden_states_len:
                    # case 2: attn_output contains only encoder hidden states
                    sub_attn_output, sub_encoder_hidden_states = None, sub_attn_output
                else:
                    # case 3: attn_output contains both hidden states and encoder hidden states
                    encoder_segment_len = global_sequence_end - global_hidden_states_len
                    sub_attn_output, sub_encoder_hidden_states = (
                        sub_attn_output[:, :-encoder_segment_len],
                        sub_attn_output[:, -encoder_segment_len :],
                    )

                if getattr(self.attn, "to_out", None) is not None and sub_attn_output is not None:
                    sub_attn_output = self.attn.to_out[0](sub_attn_output)
                    sub_attn_output = self.attn.to_out[1](sub_attn_output)
                    
                if getattr(self.attn, "to_add_out", None) is not None and sub_encoder_hidden_states is not None:
                    sub_encoder_hidden_states = self.attn.to_add_out(sub_encoder_hidden_states)

                if sub_attn_output is not None:
                    to_be_merged.append(sub_attn_output)
                if sub_encoder_hidden_states is not None:
                    to_be_merged.append(sub_encoder_hidden_states)
            sub_attn_output = torch.cat(to_be_merged, dim=1)

            # ========================
            # 3. Modulation and residual connection
            # ========================
            sub_hidden_states = torch.cat([sub_attn_output, sub_mlp_chunk], dim=2)
            sub_hidden_states = gate.unsqueeze(1) * self.proj_out(sub_hidden_states)
            sub_hidden_states = sub_hidden_states + sub_residual

            # ISP: run all gather here
            COMMUNICATION_STREAM.wait_stream(torch.cuda.current_stream())
            output_list.extend(all_gather(sub_hidden_states, gather_dim=1, no_cat=True, stream=COMMUNICATION_STREAM))
        
        torch.cuda.current_stream().wait_stream(COMMUNICATION_STREAM)

        # order the output list
        # the output list is in list [1, 2, 3, 4, 1, 2, 3, 4, ...], order it to be [1, 1, 2, 2, 3, 3, 4, 4, ...]
        sorted_output_list = []
        for i in range(ws):
            for j in range(i, len(output_list), ws):
                sorted_output_list.append(output_list[j])
        hidden_states = torch.cat(sorted_output_list, dim=1)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states

    new_forward = new_forward.__get__(block)
    block.forward = new_forward