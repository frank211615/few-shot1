import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignAttention(nn.Module):
    """
    Class-Discriminative Adaptive Feature Alignment and Filtering.

    adaptive=False:
        Original FAFM behavior.

    adaptive=True:
        1. Estimate query patch saliency.
        2. Estimate class-discriminative patch margin.
        3. Estimate query difficulty by class entropy.
        4. Generate a soft, class-conditioned adaptive gate.
    """

    def __init__(
        self,
        hidden_size,
        inner_size=None,
        num_patch=25,
        feats_size=5,
        drop_prob=0.,
        shot=1,
        keep_rate=0.5,
        adaptive=False,
        gate_temperature=0.5,
        margin_weight=1.0,
        saliency_weight=0.25,
        hardness_weight=0.5,
    ):
        super(AlignAttention, self).__init__()

        self.hidden_size = hidden_size
        self.inner_size = (
            inner_size if inner_size is not None else hidden_size // 8
        )
        self.num_patch = num_patch
        self.keep_rate = keep_rate

        # New adaptive filtering switch.
        self.adaptive = adaptive

        # Hyperparameters of adaptive gate.
        self.gate_temperature = gate_temperature
        self.margin_weight = margin_weight
        self.saliency_weight = saliency_weight
        self.hardness_weight = hardness_weight

        self.num_heads = 1
        inner_dim = self.inner_size * self.num_heads

        self.to_qkv = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                inner_dim * 3,
                bias=False
            )
        )

    @staticmethod
    def _normalize_weights(weights, eps=1e-8):
        """
        Normalize weights along the patch dimension.
        The last dimension is assumed to be the spatial patch dimension.
        """
        return weights / (weights.sum(dim=-1, keepdim=True) + eps)

    def _original_filter(self, att_probs):
        """
        Original FAFM hard quantile filtering.
        """
        threshold_value = torch.quantile(
            att_probs,
            q=(1. - self.keep_rate),
            dim=-1,
            keepdim=True,
            interpolation='higher'
        )
        return torch.relu(att_probs - threshold_value)

    def _adaptive_gate(
        self,
        reconstructed_features_b,
        value_b,
        query_attn,
    ):
        """
        Class-discriminative adaptive gate.

        reconstructed_features_b:
            [Nq, Nway, 1, P, D]

        value_b:
            [Nq, 1, 1, P, D]

        query_attn:
            [Nq, 1, 1, 1, P]

        returns:
            gate:
            [Nq, Nway, 1, P]

            hardness:
            [Nq, 1]
        """

        n_query = reconstructed_features_b.size(0)
        n_way = reconstructed_features_b.size(1)

        # ---------------------------------------------------------
        # 1. Patch-level similarity between query and class reconstruction
        # ---------------------------------------------------------
        query_patch = value_b.expand_as(reconstructed_features_b)

        patch_similarity = F.cosine_similarity(
            reconstructed_features_b,
            query_patch,
            dim=-1
        )
        # [Nq, Nway, 1, P]

        # ---------------------------------------------------------
        # 2. Class-discriminative margin
        # ---------------------------------------------------------
        if n_way > 1:
            sum_similarity = patch_similarity.sum(
                dim=1,
                keepdim=True
            )

            mean_other_similarity = (
                sum_similarity - patch_similarity
            ) / float(n_way - 1)

            margin = (
                patch_similarity
                - mean_other_similarity
            )
        else:
            # 1-way is not a meaningful classification setting,
            # but keep the code numerically safe.
            margin = torch.zeros_like(patch_similarity)

        # ---------------------------------------------------------
        # 3. Query saliency
        # ---------------------------------------------------------
        saliency = query_attn.squeeze(1).squeeze(1).squeeze(1)
        # [Nq, P]

        saliency = self._normalize_weights(saliency)

        # Center log-saliency so that it does not change
        # the overall scale too aggressively.
        saliency_log = torch.log(saliency + 1e-8)
        saliency_log = (
            saliency_log
            - saliency_log.mean(dim=-1, keepdim=True)
        )

        saliency_log = saliency_log.unsqueeze(1).unsqueeze(1)
        # [Nq, 1, 1, P]

        # ---------------------------------------------------------
        # 4. Class-level confidence and task/query difficulty
        # ---------------------------------------------------------
        saliency_for_class = saliency.unsqueeze(1).unsqueeze(1)
        # [Nq, 1, 1, P]

        class_score = (
            patch_similarity
            * saliency_for_class
        ).sum(dim=-1)
        # [Nq, Nway, 1]

        class_score = class_score.squeeze(-1)
        # [Nq, Nway]

        class_prob = F.softmax(class_score, dim=1)

        entropy = -(
            class_prob
            * torch.log(class_prob + 1e-8)
        ).sum(dim=1)

        if n_way > 1:
            hardness = entropy / math.log(float(n_way))
        else:
            hardness = torch.zeros(
                n_query,
                device=reconstructed_features_b.device,
                dtype=reconstructed_features_b.dtype
            )

        # [Nq, 1]
        hardness = hardness.unsqueeze(-1)

        # ---------------------------------------------------------
        # 5. Combine class discrimination + saliency
        # ---------------------------------------------------------
        gate_logits = (
            self.margin_weight * margin
            + self.saliency_weight * saliency_log
        )

        # Base threshold is the average score for each query/class.
        base_threshold = gate_logits.mean(
            dim=-1,
            keepdim=True
        )

        # Harder queries -> lower threshold -> retain more patches.
        adaptive_threshold = (
            base_threshold
            - self.hardness_weight
            * hardness.unsqueeze(-1)
        )

        gate = torch.sigmoid(
            (
                gate_logits
                - adaptive_threshold
            )
            / max(self.gate_temperature, 1e-4)
        )

        return gate, hardness

    def compute_distances(
        self,
        query_a,
        key_a,
        value_a,
        query_b,
        key_b,
        value_b,
        features_a,
        features_b
    ):

        # =========================================================
        # 1. Feature reconstruction
        # =========================================================
        value_a = value_a.unsqueeze(0)
        value_b = value_b.unsqueeze(1)

        n_way = value_a.size(1)

        # Query-to-support reconstruction.
        att_scores = torch.matmul(
            query_b.unsqueeze(1),
            key_a.unsqueeze(0).transpose(-1, -2).contiguous()
        )

        support_attn = F.softmax(
            att_scores / math.sqrt(self.inner_size),
            dim=-1
        )

        reconstructed_features_b = torch.matmul(
            support_attn,
            value_a
        )

        # =========================================================
        # 2. Query self-attention / saliency
        # =========================================================
        global_b = key_b.mean(-2)

        att_scores_query = torch.matmul(
            global_b.unsqueeze(-2).unsqueeze(-2),
            value_b.transpose(-2, -1)
        )

        query_attn = F.softmax(
            att_scores_query / math.sqrt(self.inner_size),
            dim=-1
        )

        # =========================================================
        # 3. Adaptive or original filtering
        # =========================================================
        if not self.adaptive:

            # -----------------------------------------------------
            # Original FAFM
            # -----------------------------------------------------
            filtered_query_weights = self._original_filter(
                query_attn
            )

            bar_q = torch.matmul(
                filtered_query_weights,
                value_b
            )

            # Second original hard filtering.
            att_scores_recon = torch.matmul(
                bar_q,
                reconstructed_features_b.transpose(-2, -1)
            )

            recon_attn = F.softmax(
                att_scores_recon / math.sqrt(self.inner_size),
                dim=-1
            )

            filtered_recon_weights = self._original_filter(
                recon_attn
            )

            agg_recon_b = torch.matmul(
                filtered_recon_weights,
                reconstructed_features_b
            )

        else:

            # -----------------------------------------------------
            # New class-discriminative adaptive gate
            # -----------------------------------------------------
            gate, hardness = self._adaptive_gate(
                reconstructed_features_b,
                value_b,
                query_attn
            )

            # Query attention is expanded to every candidate class.
            query_attn_class = query_attn.expand(
                -1,
                n_way,
                -1,
                -1,
                -1
            )

            # Apply class-conditioned soft gate.
            gated_query_weights = (
                query_attn_class
                * gate.unsqueeze(-2)
            )

            gated_query_weights = self._normalize_weights(
                gated_query_weights
            )

            # Class-specific filtered query representation.
            bar_q = torch.matmul(
                gated_query_weights,
                value_b
            )

            # -----------------------------------------------------
            # Second-stage reconstruction attention
            # -----------------------------------------------------
            att_scores_recon = torch.matmul(
                bar_q,
                reconstructed_features_b.transpose(-2, -1)
            )

            recon_attn = F.softmax(
                att_scores_recon / math.sqrt(self.inner_size),
                dim=-1
            )

            # Use the same discriminative gate to suppress
            # irrelevant reconstructed patches.
            gated_recon_weights = (
                recon_attn
                * gate.unsqueeze(-2)
            )

            gated_recon_weights = self._normalize_weights(
                gated_recon_weights
            )

            agg_recon_b = torch.matmul(
                gated_recon_weights,
                reconstructed_features_b
            )

        # =========================================================
        # 4. Filtered similarity
        # =========================================================
        scores_filter = torch.matmul(
            agg_recon_b,
            bar_q.transpose(-2, -1)
        )

        # Safe squeeze: preserve [Nq, Nway].
        scores_filter = scores_filter[..., 0, 0]

        # =========================================================
        # 5. Original alignment similarity
        # Keep this branch unchanged so that the comparison
        # isolates the filtering improvement.
        # =========================================================
        reconstructed_features_b_mean = (
            reconstructed_features_b.mean(dim=-2)
        )

        value_b_mean = value_b.mean(dim=-2)

        scores_align = torch.matmul(
            reconstructed_features_b_mean,
            value_b_mean.transpose(-2, -1)
        )

        scores_align = scores_align[..., 0, 0]

        if self.adaptive:
            return scores_filter, scores_align, hardness

        return scores_filter, scores_align

    def forward(self, features_a, features_b):

        # =========================================================
        # Support projection
        # =========================================================
        features_a = features_a.view(
            features_a.size(0),
            features_a.size(1),
            -1
        ).permute(0, 2, 1).contiguous()

        b_a, l_a, d_a = features_a.shape

        qkv_a = self.to_qkv(features_a)

        qkv_a = qkv_a.view(
            b_a,
            l_a,
            3,
            self.num_heads,
            -1
        ).permute(
            2, 0, 3, 1, 4
        ).contiguous()

        query_a, key_a, value_a = qkv_a.chunk(3)

        query_a = query_a.squeeze(0)
        key_a = key_a.squeeze(0)
        value_a = value_a.squeeze(0)

        # =========================================================
        # Query projection
        # =========================================================
        features_b = features_b.view(
            features_b.size(0),
            features_b.size(1),
            -1
        ).permute(0, 2, 1).contiguous()

        b_b, l_b, d_b = features_b.shape

        qkv_b = self.to_qkv(features_b)

        qkv_b = qkv_b.view(
            b_b,
            l_b,
            3,
            self.num_heads,
            -1
        ).permute(
            2, 0, 3, 1, 4
        ).contiguous()

        query_b, key_b, value_b = qkv_b.chunk(3)

        query_b = query_b.squeeze(0)
        key_b = key_b.squeeze(0)
        value_b = value_b.squeeze(0)

        distances = self.compute_distances(
            query_a,
            key_a,
            value_a,
            query_b,
            key_b,
            value_b,
            features_a,
            features_b
        )

        # Keep the original two-output interface.
        if self.adaptive:
            scores_filter, scores_align, _ = distances
            return scores_filter, scores_align

        return distances