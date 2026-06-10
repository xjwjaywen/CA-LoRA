"""Contrastive anchor regularization losses."""
import torch
import torch.nn.functional as F


def contrastive_anchor_loss(
    generated_features: torch.Tensor,
    anchor_features: torch.Tensor,
    target_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Anchor consistency loss via InfoNCE.

    Pulls generated features toward target while keeping them
    within the anchor distribution (prevents forgetting).

    generated_features: (B, D) features of generated images at current step
    anchor_features: (N_a, D) features of anchor images
    target_features: (N_t, D) features of real target images
    """
    gen = F.normalize(generated_features, dim=-1)
    anc = F.normalize(anchor_features, dim=-1)
    tgt = F.normalize(target_features, dim=-1)

    # Positive: similarity to target centroid
    target_centroid = F.normalize(tgt.mean(dim=0, keepdim=True), dim=-1)
    pos_sim = (gen @ target_centroid.T).squeeze(-1) / temperature

    # Negatives: very dissimilar anchors (those far from target distribution)
    anchor_target_sim = (anc @ target_centroid.T).squeeze(-1)
    _, far_indices = anchor_target_sim.sort()
    n_neg = min(len(anc), 64)
    neg_anchors = anc[far_indices[:n_neg]]
    neg_sim = (gen @ neg_anchors.T) / temperature

    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
    labels = torch.zeros(len(gen), dtype=torch.long, device=gen.device)

    return F.cross_entropy(logits, labels)


def diversity_loss(
    generated_features: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Push generated samples apart to maintain diversity.
    Uses pairwise cosine distance with a margin.
    """
    gen = F.normalize(generated_features, dim=-1)
    sim_matrix = gen @ gen.T
    mask = ~torch.eye(len(gen), dtype=torch.bool, device=gen.device)
    pairwise_sim = sim_matrix[mask]

    loss = F.relu(pairwise_sim - margin).mean()
    return loss


def compute_feature_losses(
    unet_features: torch.Tensor,
    anchor_features: torch.Tensor,
    target_features: torch.Tensor,
    lambda_anchor: float = 0.1,
    lambda_diverse: float = 0.05,
    temperature: float = 0.07,
) -> dict:
    """Compute all CA-LoRA regularization losses."""
    l_anchor = contrastive_anchor_loss(
        unet_features, anchor_features, target_features, temperature
    )
    l_diverse = diversity_loss(unet_features)

    total = lambda_anchor * l_anchor + lambda_diverse * l_diverse

    return {
        "loss_anchor": l_anchor,
        "loss_diverse": l_diverse,
        "loss_reg_total": total,
    }
