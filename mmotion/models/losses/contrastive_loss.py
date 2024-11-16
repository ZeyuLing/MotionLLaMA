import torch
import torch.nn.functional as F

from mmotion.registry import MODELS


@MODELS.register_module()
class InfoNCE:
    """
    This class implements the InfoNCE loss function.

    Attributes:
    - t: a temperature parameter for the softmax function in the loss calculation.

    Methods:
    - __call__: computes the InfoNCE loss given the motion and text features, and an optional distance matrix.
    """

    def __init__(self, t: float = 0.1, threshold_selfsim: float = 0.85):
        """
        Initializes the InfoNCE object with a given temperature parameter.

        Inputs:
        - t: a temperature parameter for the softmax function in the loss calculation.
        """
        self.t = t
        self.threshold_selfsim = threshold_selfsim

    def __call__(self, f, dist):
        """
        Computes the InfoNCE loss given the motion and text features, and an optional distance matrix.

        Inputs:
        - f: a tuple containing the motion and text features. Each feature is a 2D tensor of shape (N, d).
        - dist: an optional distance matrix. If provided, it is used to mask the logits.

        Outputs:
        - loss_m: the InfoNCE loss computed using the motion features.
        - loss_t: the InfoNCE loss computed using the text features.
        """
        t = self.t
        f_motion, f_text = f[0], f[1]

        N, d = f_motion.shape[0], f_motion.shape[1]

        # Normalize the motion and text features
        Emb_motion = F.normalize(f_motion, dim=1)
        Emb_text = F.normalize(f_text, dim=1)

        # Compute the logits as the dot product of the normalized features
        t = torch.tensor(t).to(f_motion.device)
        logits = torch.mm(Emb_motion, Emb_text.T) / t

        # If a distance matrix is provided, use it to mask the logits
        if dist is not None:
            text_logits = dist.detach()
            mask = torch.where(
                torch.logical_and(text_logits > self.threshold_selfsim, text_logits < 1.),
                torch.tensor(-torch.inf).to(text_logits),
                torch.tensor(torch.inf).to(text_logits),
            )
            mask.diagonal().fill_(float("inf"))
            logits = torch.min(mask, logits)

        N = f_motion.shape[0]

        # Compute the labels as the indices of the features
        labels = torch.arange(N).to(f_motion.device)
        # Compute the InfoNCE loss for the motion and text features
        loss_m = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        loss = (loss_m + loss_t) / 2
        return loss

    def __repr__(self):
        return "InfoNCE()"

class ClipLoss:
    def __init__(self, t: float = 0.15):
        """
        Initializes the InfoNCE object with a given temperature parameter.

        Inputs:
        - t: a temperature parameter for the softmax function in the loss calculation.
        """
        self.t = t

    def __call__(self, f, dist):
        t = self.t
        f_motion, f_text = f[0], f[1]
        # Getting Image and Text Features


        # Calculating the Loss
        logits = (f_text @ f_motion.T) / self.t
        motion_similarity = f_motion @ f_motion.T
        texts_similarity = f_text @ f_text.T
        targets = F.softmax(
            (motion_similarity + texts_similarity) / 2 * self.t, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        motion_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (motion_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
