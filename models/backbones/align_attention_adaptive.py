import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignAttention(nn.Module):
    """
    Query-aware feature alignment + class-discriminative adaptive filtering.

    This module keeps the same high-level interface as the original FAFN
    AlignAttention:

        scores_filter, scores_align = module(support_features, query_features)

    Expected inputs
    ---------------
    support_features:
        [N_way, C, shot, H, W]
        or [N_way, C, shot, L]
        or [N_way, shot, L, C]

    query_features:
        [N_query, C, H, W]
        or [N_query, C, L]
        or [N_query, L, C]

    Outputs
    -------
    scores_filter:
        [N_query, N_way]
    scores_align:
        [N_query, N_way]

    The `adaptive=False` branch is provided for ablation and falls back to
    percentile-based hard filtering, while `adaptive=True` uses a soft,
    class-discriminative gate with query-hardness-aware thresholds.
    """

    def __init__(
        self,
        hidden_size,
        inner_size,
        feats_size=None,
        num_patch=25,
        shot=1,
        keep_rate=0.5,
        adaptive=False,
        gate_temperature=0.5,
        margin_weight=1.0,
        saliency_weight=0.25,
        hardness_weight=0.5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.inner_size = inner_size

        # Compatibility with the existing Ours.py constructor.
        # The current implementation infers the runtime spatial/token size
        # directly from the input tensor, so feats_size is kept for API
        # compatibility and does not alter the computation.
        self.feats_size = feats_size

        self.num_patch = num_patch
        self.shot = shot
        self.keep_rate = float(keep_rate)
        self.adaptive = adaptive

        self.gate_temperature = max(float(gate_temperature), 1e-4)
        self.margin_weight = float(margin_weight)
        self.saliency_weight = float(saliency_weight)
        self.hardness_weight = float(hardness_weight)

        # Shared Q/K/V projection, following the original attention-style
        # implementation. Values are projected back to hidden_size before
        # feature similarity is measured.
        self.to_qkv = nn.Linear(hidden_size, inner_size * 3, bias=False)
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _query_to_tokens(self, query):
        """Convert query features to [Nq, L, C]."""
        if query.dim() == 5:
            # [N, C, H, W] is not possible with 5 dims; here we expect a
            # leading singleton shot-like dimension only as a safety path.
            if query.size(1) == 1:
                query = query.squeeze(1)
            elif query.size(2) == 1:
                query = query.squeeze(2)
            else:
                raise ValueError(
                    "Unsupported 5-D query shape: {}. Expected [N,C,H,W] "
                    "or [N,C,L] / [N,L,C].".format(tuple(query.shape))
                )

        if query.dim() != 3 and query.dim() != 4:
            raise ValueError(
                "query_features must be 3-D or 4-D, got shape {}".format(
                    tuple(query.shape)
                )
            )

        if query.dim() == 4:
            # [N, C, H, W]
            n, c, h, w = query.shape
            return query.flatten(2).transpose(1, 2).contiguous()

        # 3-D input: decide between [N,C,L] and [N,L,C].
        n, a, b = query.shape
        if a == self.hidden_size and b != self.hidden_size:
            return query.transpose(1, 2).contiguous()
        if b == self.hidden_size:
            return query.contiguous()

        # Fall back to the conventional [N,C,L] interpretation.
        return query.transpose(1, 2).contiguous()

    def _support_to_tokens(self, support):
        """Convert support to [Nway, shot, L, C]."""
        if support.dim() == 5:
            # Standard: [Nway, C, shot, H, W]
            n_way, c, shot, h, w = support.shape
            if c == self.hidden_size:
                return support.permute(0, 2, 3, 4, 1).reshape(
                    n_way, shot, h * w, c
                ).contiguous()

            # Safety path for [Nway, shot, C, H, W].
            if shot == self.hidden_size:
                return support.permute(0, 1, 3, 4, 2).reshape(
                    n_way, c, h * w, shot
                ).contiguous()

            raise ValueError(
                "Unsupported 5-D support shape: {}. Expected [Nway,C,shot,H,W].".format(
                    tuple(support.shape)
                )
            )

        if support.dim() != 4:
            raise ValueError(
                "support_features must be 4-D or 5-D, got shape {}".format(
                    tuple(support.shape)
                )
            )

        # [Nway, C, shot, L]
        n_way, a, b, c = support.shape
        if a == self.hidden_size:
            return support.permute(0, 2, 3, 1).contiguous()

        # [Nway, shot, L, C]
        if c == self.hidden_size:
            return support.contiguous()

        # [Nway, shot, C, L]
        if b == self.hidden_size:
            return support.permute(0, 1, 3, 2).contiguous()

        raise ValueError(
            "Cannot infer support channel dimension from shape {} with hidden_size={}.".format(
                tuple(support.shape), self.hidden_size
            )
        )

    @staticmethod
    def _normalize_feature(x, dim=-1, eps=1e-6):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

    @staticmethod
    def _safe_quantile(x, q):
        """Quantile wrapper compatible with common PyTorch 1.x/2.x versions."""
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        return torch.quantile(x, q, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def _cross_align(self, support_tokens, query_tokens):
        """
        Cross-attend every query patch to each class's support patches.

        support_tokens: [Nway, shot, Ls, C]
        query_tokens:   [Nq, Lq, C]

        returns
            aligned: [Nq, Nway, Lq, C]
        """
        n_query, lq, _ = query_tokens.shape
        n_way, shot, ls, _ = support_tokens.shape

        support_flat = support_tokens.reshape(n_way, shot * ls, self.hidden_size)

        qkv_q = self.to_qkv(query_tokens)
        q_query, _, _ = torch.chunk(qkv_q, 3, dim=-1)

        qkv_s = self.to_qkv(support_flat)
        _, k_support, v_support = torch.chunk(qkv_s, 3, dim=-1)

        scale = 1.0 / math.sqrt(float(self.inner_size))

        # q_query: [Nq,Lq,D]
        # k_support: [Nway,Ls_total,D]
        # -> attention: [Nq,Nway,Lq,Ls_total]
        attn = torch.einsum("nld,msd->nmls", q_query, k_support) * scale
        attn = F.softmax(attn, dim=-1)

        aligned_inner = torch.einsum("nmls,msd->nmld", attn, v_support)
        aligned = self.out_proj(aligned_inner)

        return aligned

    # ------------------------------------------------------------------
    # Original FAFN-style percentile filtering
    # ------------------------------------------------------------------
    def _original_filter(self, aligned, query_tokens):
        """
        Percentile-based hard filtering.

        The query-global representation generates a local attention map. A
        fixed percentile threshold is then used to suppress low-response
        patches. This path is intentionally deterministic and is useful for
        baseline/ablation experiments.
        """
        aligned_n = self._normalize_feature(aligned, dim=-1)
        query_n = self._normalize_feature(query_tokens, dim=-1)

        # Query global descriptor: [Nq, C]
        query_global = F.normalize(query_n.mean(dim=1), dim=-1)

        # Local relevance: [Nq, Nway, L]
        saliency = torch.einsum("nc,nmlc->nml", query_global, aligned_n)

        # Convert to [0,1]-like positive response for filtering.
        saliency = F.relu(saliency)

        # keep_rate=0.5 -> threshold at 50th percentile.
        threshold_q = 1.0 - self.keep_rate
        threshold = self._safe_quantile(saliency, threshold_q)

        mask = F.relu(saliency - threshold)

        # Avoid an all-zero representation for degenerate cases.
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        filtered = (aligned_n * mask.unsqueeze(-1)).sum(dim=-2) / mask_sum

        scores_filter = torch.einsum("nc,nmc->nm", query_global, filtered)

        # Alignment score: mean patch-level cosine similarity.
        scores_align = torch.einsum("nlc,nmlc->nm", query_n, aligned_n) / float(aligned_n.size(-2))

        return scores_filter, scores_align, mask

    # ------------------------------------------------------------------
    # Proposed class-discriminative adaptive filtering
    # ------------------------------------------------------------------
    def _adaptive_gate(self, aligned, query_tokens):
        """
        Build a soft class-conditional patch gate.

        Core idea:
          1. Compute patch-level similarity between query and each class's
             aligned representation.
          2. Convert that into a class-vs-competitor margin.
          3. Add a lightweight query saliency cue.
          4. Estimate query hardness from class entropy.
          5. Move the threshold according to hardness.
          6. Use a sigmoid gate rather than a hard binary mask.

        Shapes are deliberately kept explicit to avoid accidental broadcasting
        across N_way versus N_query dimensions.
        """
        n_query, n_way, l, c = aligned.shape
        assert query_tokens.shape[0] == n_query
        assert query_tokens.shape[1] == l
        assert query_tokens.shape[2] == c

        aligned_n = self._normalize_feature(aligned, dim=-1)
        query_n = self._normalize_feature(query_tokens, dim=-1)

        # --------------------------------------------------------------
        # 1) Patch similarity: [Nq, Nway, L]
        # --------------------------------------------------------------
        patch_similarity = torch.einsum(
            "nlc,nmlc->nml", query_n, aligned_n
        )

        # --------------------------------------------------------------
        # 2) Class-discriminative margin.
        # --------------------------------------------------------------
        if n_way > 1:
            class_sum = patch_similarity.sum(dim=1, keepdim=True)
            mean_other = (class_sum - patch_similarity) / float(n_way - 1)
        else:
            mean_other = torch.zeros_like(patch_similarity)

        margin = patch_similarity - mean_other

        # --------------------------------------------------------------
        # 3) Query-local saliency cue.
        # --------------------------------------------------------------
        query_global = F.normalize(query_n.mean(dim=1), dim=-1)
        saliency = torch.einsum("nc,nlc->nl", query_global, query_n)

        # Per-query standardization makes this cue stable across episodes.
        saliency_mean = saliency.mean(dim=-1, keepdim=True)
        saliency_std = saliency.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
            1e-6
        )
        saliency_z = (saliency - saliency_mean) / saliency_std
        saliency_z = saliency_z.unsqueeze(1).expand(-1, n_way, -1)

        # --------------------------------------------------------------
        # 4) Class logits before thresholding.
        # --------------------------------------------------------------
        gate_logits = (
            self.margin_weight * margin
            + self.saliency_weight * saliency_z
        )

        # --------------------------------------------------------------
        # 5) Query hardness via class entropy.
        # --------------------------------------------------------------
        # Mean patch similarity provides a compact class score.
        class_scores = patch_similarity.mean(dim=-1)
        class_prob = F.softmax(class_scores, dim=1)

        entropy = -(
            class_prob * class_prob.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)

        if n_way > 1:
            hardness = entropy / math.log(float(n_way))
        else:
            hardness = torch.zeros_like(entropy)

        hardness = hardness.clamp(0.0, 1.0)  # [Nq,1]

        # --------------------------------------------------------------
        # 6) Adaptive threshold.
        # --------------------------------------------------------------
        # The baseline threshold is the (1-keep_rate)-quantile. With the
        # default keep_rate=0.5 this is the median.
        threshold_q = 1.0 - self.keep_rate

        # Explicitly keep a singleton class dimension here:
        # [Nq,Nway,1]. This is the key fix for the previous broadcasting bug
        # when Nq=225 and Nway=15.
        base_threshold = self._safe_quantile(gate_logits, threshold_q)

        hardness_3d = hardness.view(n_query, 1, 1)

        # Harder queries -> lower threshold -> retain more evidence.
        adaptive_threshold = base_threshold * (
            1.0 - self.hardness_weight * hardness_3d
        )

        gate = torch.sigmoid(
            (gate_logits - adaptive_threshold) / self.gate_temperature
        )

        # Soft gate is strictly positive; no all-zero representation.
        gate = gate.clamp_min(1e-4)

        # --------------------------------------------------------------
        # 7) Filtered class representation.
        # --------------------------------------------------------------
        gate_weight = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        filtered = torch.einsum(
            "nml,nmlc->nmc", gate_weight, aligned_n
        )

        # --------------------------------------------------------------
        # 8) Filtered score + alignment score.
        # --------------------------------------------------------------
        query_global = F.normalize(query_n.mean(dim=1), dim=-1)
        filtered = F.normalize(filtered, dim=-1)

        scores_filter = torch.einsum(
            "nc,nmc->nm", query_global, filtered
        )

        scores_align = torch.einsum(
            "nlc,nmlc->nm", query_n, aligned_n
        ) / float(aligned_n.size(-2))

        return scores_filter, scores_align, gate

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, support_features, query_features):
        support_tokens = self._support_to_tokens(support_features)
        query_tokens = self._query_to_tokens(query_features)

        # Basic shape checks.
        if support_tokens.size(-1) != self.hidden_size:
            raise ValueError(
                "Support channel dimension {} does not match hidden_size {}.".format(
                    support_tokens.size(-1), self.hidden_size
                )
            )
        if query_tokens.size(-1) != self.hidden_size:
            raise ValueError(
                "Query channel dimension {} does not match hidden_size {}.".format(
                    query_tokens.size(-1), self.hidden_size
                )
            )

        # Keep original spatial-token count when possible, but allow callers
        # to pass other token counts during debugging/ablation.
        if self.num_patch is not None:
            if support_tokens.size(2) != query_tokens.size(1):
                raise ValueError(
                    "Support/query patch number mismatch: support has {}, query has {}."
                    .format(support_tokens.size(2), query_tokens.size(1))
                )

        aligned = self._cross_align(support_tokens, query_tokens)

        if self.adaptive:
            scores_filter, scores_align, _ = self._adaptive_gate(
                aligned, query_tokens
            )
        else:
            scores_filter, scores_align, _ = self._original_filter(
                aligned, query_tokens
            )

        # IMPORTANT:
        # Explicitly squeeze only singleton trailing dimensions. This avoids
        # the previous [...,0,0] style indexing problem and guarantees
        # [N_query, N_way] output for Ours.py.
        scores_filter = (
            scores_filter
            .squeeze(-1)
            .squeeze(-1)
            .squeeze(-1)
        )
        scores_align = (
            scores_align
            .squeeze(-1)
            .squeeze(-1)
        )

        if scores_filter.dim() != 2 or scores_align.dim() != 2:
            raise RuntimeError(
                "Unexpected output shapes: scores_filter={}, scores_align={}. "
                "Expected [N_query,N_way].".format(
                    tuple(scores_filter.shape), tuple(scores_align.shape)
                )
            )

        return scores_filter, scores_align
