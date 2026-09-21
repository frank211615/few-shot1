import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss

def alignment_consistency_loss(alignment, augmented_alignment, eps=1e-8):
    """Symmetric KL divergence between paired alignment distributions.

    Both tensors have shape ``[N_query, N_way, query_patch, support_patch]``
    and are already softmax-normalized along the support-patch dimension.
    A symmetric form lets both views learn rather than treating either view as
    a fixed teacher.
    """
    if alignment.shape != augmented_alignment.shape:
        raise ValueError(
            'Alignment shapes must match, got {} and {}.'.format(
                tuple(alignment.shape), tuple(augmented_alignment.shape)
            )
        )

    alignment = alignment.clamp_min(eps)
    augmented_alignment = augmented_alignment.clamp_min(eps)
    forward_kl = (
        alignment * (alignment.log() - augmented_alignment.log())
    ).sum(dim=-1)
    reverse_kl = (
        augmented_alignment * (augmented_alignment.log() - alignment.log())
    ).sum(dim=-1)

    return 0.5 * (forward_kl + reverse_kl).mean()


def default_train(train_loader,model,
    optimizer,writer,iter_counter,consistency_weight=0.0):

    if consistency_weight < 0:
        raise ValueError('consistency_weight must be non-negative')

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('W1',model.w1.item(),iter_counter)
    writer.add_scalar('W2',model.w2.item(),iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)

    avg_loss = 0
    avg_cls_loss = 0
    avg_consistency_loss = 0
    avg_acc = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1

        if consistency_weight > 0:
            if not isinstance(inp, (tuple, list)) or len(inp) != 2:
                raise ValueError(
                    'Alignment consistency requires paired training views. '
                    'Create the loader with paired_views=True.'
                )
            inp, augmented_inp = inp
            inp = inp.cuda()
            augmented_inp = augmented_inp.cuda()
            log_prediction, alignment = model(inp, return_alignment=True)
            _, augmented_alignment = model(
                augmented_inp,
                return_alignment=True,
            )
            consistency_loss = alignment_consistency_loss(
                alignment,
                augmented_alignment,
            )
        else:
            inp = inp.cuda()
            log_prediction = model(inp)
            consistency_loss = loss = torch.zeros(
                (),
                device=inp.device,
                dtype=log_prediction.dtype,
            )

        classification_loss = criterion(log_prediction,target)
        loss = classification_loss + consistency_weight * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        classification_loss_value = classification_loss.item()
        consistency_loss_value = consistency_loss.item()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

        avg_acc += acc
        avg_loss += loss_value
        avg_cls_loss += classification_loss_value
        avg_consistency_loss += consistency_loss_value

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)
    avg_cls_loss = avg_cls_loss/(i+1)
    avg_consistency_loss = avg_consistency_loss/(i+1)

    writer.add_scalar('total_loss',avg_loss,iter_counter)
    writer.add_scalar('classification_loss',avg_cls_loss,iter_counter)
    writer.add_scalar('alignment_consistency_loss',avg_consistency_loss,iter_counter)
    # Retained for compatibility with existing TensorBoard dashboards.
    writer.add_scalar('proto_loss',avg_cls_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc
