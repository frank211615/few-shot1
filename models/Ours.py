import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Conv_4, ResNet
from .backbones.FSRM import FSRM
from .backbones.align_attention_adaptive import AlignAttention


class Ours(nn.Module):
    """
    FAFN / Adaptive-FAFN top-level model.

    adaptive=False:
        原始 FAFN Filtering

    adaptive=True:
        使用改进后的 Adaptive / Class-Discriminative Filtering
    """

    def __init__(
        self,
        way=None,
        shots=None,
        resnet=False,
        keep_rate=0.5,
        adaptive=True
    ):
        super(Ours, self).__init__()

        # =========================================================
        # Basic settings
        # =========================================================
        self.resolution = 5 * 5
        self.shots = shots
        self.way = way

        if shots is None:
            raise ValueError(
                "shots cannot be None. "
                "Expected something like shots=[5, 15]."
            )

        # =========================================================
        # Backbone
        # =========================================================
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(
                self.num_channel
            )

        self.dim = self.num_channel * self.resolution
        self.d = self.num_channel
        self.patch_sz = (5, 5)

        # =========================================================
        # Similarity scaling
        # =========================================================
        # Same as the original FAFN implementation.
        self.div = self.num_channel ** -0.5

        # Learnable temperature.
        self.scale = nn.Parameter(
            torch.FloatTensor([1.0]),
            requires_grad=True
        )

        # Learnable weights for:
        # scores_filter and scores_align
        #
        # Keep the original initialization for fair comparison.
        self.w1 = nn.Parameter(
            torch.FloatTensor([1.0 / 3.0]),
            requires_grad=True
        )

        self.w2 = nn.Parameter(
            torch.FloatTensor([1.0 / 3.0]),
            requires_grad=True
        )

        # =========================================================
        # FSRM
        # =========================================================
        self.fsrm = FSRM(
            sequence_length=self.resolution,
            embedding_dim=self.num_channel,
            num_layers=1,
            num_heads=1,
            mlp_dropout_rate=0.0,
            attention_dropout=0.0,
            positional_embedding="sine"
        )

        # =========================================================
        # FAFM / Adaptive FAFM
        # =========================================================
        self.FAFM = AlignAttention(
            hidden_size=self.num_channel,
            inner_size=self.num_channel,
            num_patch=self.resolution,
            feats_size=5,
            drop_prob=0.0,
            shot=self.shots[0],
            keep_rate=keep_rate,

            # ---------------------------------------------
            # Core switch:
            # False -> original FAFN
            # True  -> your adaptive method
            # ---------------------------------------------
            adaptive=adaptive,

            # ---------------------------------------------
            # Parameters of adaptive filtering
            # ---------------------------------------------
            gate_temperature=0.5,
            margin_weight=1.0,
            saliency_weight=0.25,
            hardness_weight=0.5
        )

        self.adaptive = adaptive
        self.keep_rate = keep_rate

    # =============================================================
    # Backbone + FSRM
    # =============================================================
    def get_feature_map(self, inp):
        """
        Input:
            inp: [B, 3, 84, 84]

        Output:
            feature_map: [B, C, 5, 5]
        """

        batch_size = inp.size(0)

        # Backbone
        feature_map = self.feature_extractor(inp)

        # FSRM
        feature_map = self.fsrm(feature_map)

        # [B, 25, C] -> [B, C, 25]
        feature_map = feature_map.transpose(1, 2)

        # [B, C, 25] -> [B, C, 5, 5]
        feature_map = feature_map.view(
            batch_size,
            self.num_channel,
            5,
            5
        )

        return feature_map

    # =============================================================
    # Support / Query organization
    # =============================================================
    def split_support_query(
        self,
        feature_vector,
        way,
        shot,
        query_shot
    ):
        """
        Convert flattened feature map into support/query tensors.

        feature_vector:
            [B, C, 25]

        Support:
            [way, C, shot, 5, 5]

        Query:
            [way * query_shot, C, 5, 5]
        """

        d = self.d

        # ---------------------------------------------------------
        # Support
        # ---------------------------------------------------------
        support = feature_vector[
            :way * shot
        ]

        support = support.view(
            way,
            shot,
            d,
            *self.patch_sz
        )

        # [way, shot, d, 5, 5]
        #
        # -> [way, d, shot, 5, 5]
        support = support.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        # ---------------------------------------------------------
        # Query
        # ---------------------------------------------------------
        query = feature_vector[
            way * shot:
        ]

        query = query.view(
            way * query_shot,
            d,
            *self.patch_sz
        )

        return support, query

    # =============================================================
    # FAFM similarity
    # =============================================================
    def get_neg_l2_dist(
        self,
        inp,
        way,
        shot,
        query_shot
    ):
        """
        Calculate class similarity.

        Note:
        The function name is inherited from the original code.
        FAFM actually returns inner-product similarity rather than
        negative L2 distance.
        """

        # ---------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------
        feature_map = self.get_feature_map(inp)

        # [B, C, 5, 5]
        # -> [B, C, 25]
        feature_vector = feature_map.view(
            -1,
            self.d,
            self.resolution
        )

        # ---------------------------------------------------------
        # Split support and query
        # ---------------------------------------------------------
        support, query = self.split_support_query(
            feature_vector=feature_vector,
            way=way,
            shot=shot,
            query_shot=query_shot
        )

        # ---------------------------------------------------------
        # FAFM
        # ---------------------------------------------------------
        scores_F, scores_A = self.FAFM(
            support,
            query
        )

        # Expected shapes:
        #
        # scores_F: [N_query, way]
        # scores_A: [N_query, way]
        #
        # where
        # N_query = way * query_shot

        # ---------------------------------------------------------
        # Numerical sanity check
        # ---------------------------------------------------------
        if scores_F.dim() != 2:
            raise RuntimeError(
                "scores_F should have shape [N_query, way], "
                f"but got {tuple(scores_F.shape)}"
            )

        if scores_A.dim() != 2:
            raise RuntimeError(
                "scores_A should have shape [N_query, way], "
                f"but got {tuple(scores_A.shape)}"
            )

        # ---------------------------------------------------------
        # Fuse the two similarity branches
        # ---------------------------------------------------------
        scores = (
            self.w1 * scores_F
            + self.w2 * scores_A
        )

        return scores

    # =============================================================
    # Meta-test
    # =============================================================
    def meta_test(
        self,
        inp,
        way,
        shot,
        query_shot
    ):
        """
        Meta-test inference.

        Returns:
            predicted class index for each query.
        """

        scores = self.get_neg_l2_dist(
            inp=inp,
            way=way,
            shot=shot,
            query_shot=query_shot
        )

        _, max_index = torch.max(
            scores,
            dim=1
        )

        return max_index

    # =============================================================
    # Training forward
    # =============================================================
    def forward(self, inp):
        """
        Training forward.

        Input:
            inp:
                support + query images

        Output:
            log probabilities [N_query, way]
        """

        if self.way is None:
            raise ValueError(
                "self.way is None. "
                "Please provide way when constructing Ours."
            )

        if self.shots is None or len(self.shots) < 2:
            raise ValueError(
                "self.shots must contain [shot, query_shot]."
            )

        shot = self.shots[0]
        query_shot = self.shots[1]

        # ---------------------------------------------------------
        # Similarity
        # ---------------------------------------------------------
        scores = self.get_neg_l2_dist(
            inp=inp,
            way=self.way,
            shot=shot,
            query_shot=query_shot
        )

        # ---------------------------------------------------------
        # Temperature scaling
        # ---------------------------------------------------------
        logits = (
            scores
            * self.div
            * self.scale
        )

        # ---------------------------------------------------------
        # Classification probability
        # ---------------------------------------------------------
        log_prediction = F.log_softmax(
            logits,
            dim=1
        )

        return log_prediction

    # =============================================================
    # Debug information
    # =============================================================
    def extra_repr(self):
        return (
            f"way={self.way}, "
            f"shots={self.shots}, "
            f"channels={self.num_channel}, "
            f"resolution={self.resolution}, "
            f"keep_rate={self.keep_rate}, "
            f"adaptive={self.adaptive}"
        )