'''
MSSA_PLUS_HYBRID: MSSA_PLUS with a structured classifier for ROBOT compound faults.

Drop this file into models/MSSA_PLUS_HYBRID.py and run with --model_name MSSA_PLUS_HYBRID.

It keeps the 8-way label-powerset classifier for compatibility with the existing
TL-Fault-Diagnosis-Library pipeline, but adds a 3-output axis-decoupling classifier:

    class 0: normal    -> [0, 0, 0]
    class 1: axis4     -> [1, 0, 0]
    class 2: axis5     -> [0, 1, 0]
    class 3: axis6     -> [0, 0, 1]
    class 4: axis45    -> [1, 1, 0]
    class 5: axis46    -> [1, 0, 1]
    class 6: axis56    -> [0, 1, 1]
    class 7: axis456   -> [1, 1, 1]

Training losses:
  - source 8-class CE loss,
  - source 3-axis BCE loss,
  - source exact-combination NLL from axis probabilities,
  - class-head / axis-head consistency,
  - ensemble pseudo-label LMMD,
  - target classifier consistency,
  - target MCC regularization.

Inference blends the 8-way softmax distribution and the 8-way distribution implied
by the 3 sigmoid axis outputs.
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import modules
from train_utils import TrainerBase


# Canonical ROBOT label order. The project's docs define:
# 0 normal, 1 axis4, 2 axis5, 3 axis6, 4 axis45, 5 axis46, 6 axis56, 7 axis456.
ROBOT_CLASS_TO_AXIS = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
], dtype=torch.float32)


def _label_set_tensor(label_set, device):
    return torch.as_tensor(label_set, dtype=torch.long, device=device)


def source_labels_to_axis_targets(local_labels, label_set=None):
    """Convert local class indices to [axis4, axis5, axis6] binary targets."""
    device = local_labels.device
    mapping = ROBOT_CLASS_TO_AXIS.to(device)
    if label_set is not None:
        actual_labels = _label_set_tensor(label_set, device)[local_labels.long()]
    else:
        actual_labels = local_labels.long()
    actual_labels = actual_labels.clamp(min=0, max=7)
    return mapping[actual_labels]


def reorder_canonical_probs_to_label_set(prob_canonical, label_set):
    """Return probabilities in the branch's local class order."""
    idx = _label_set_tensor(label_set, prob_canonical.device).clamp(min=0, max=7)
    return prob_canonical[:, idx]


def axis_logits_to_class_prob(axis_logits, eps=1e-7):
    """Convert 3 sigmoid axis logits into canonical 8-class combination probabilities."""
    device = axis_logits.device
    combos = ROBOT_CLASS_TO_AXIS.to(device).view(1, 8, 3)
    p = torch.sigmoid(axis_logits).clamp(eps, 1.0 - eps).view(-1, 1, 3)
    comp_prob = combos * p + (1.0 - combos) * (1.0 - p)
    class_prob = comp_prob.prod(dim=2)
    return class_prob / class_prob.sum(dim=1, keepdim=True).clamp_min(eps)


def sharpen_prob(prob, temperature=0.5, eps=1e-6):
    if temperature <= 0:
        return prob
    out = prob.clamp_min(eps).pow(1.0 / temperature)
    return out / out.sum(dim=1, keepdim=True).clamp_min(eps)


class LMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-6):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.eps = eps

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma is not None:
            bandwidth = torch.as_tensor(self.fix_sigma, dtype=source.dtype, device=source.device)
        else:
            denom = max(n_samples * n_samples - n_samples, 1)
            bandwidth = torch.sum(l2_distance.detach()) / denom
        bandwidth = bandwidth.clamp_min(self.eps)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-l2_distance / bw.clamp_min(self.eps)) for bw in bandwidth_list)

    def _weights(self, s_label, t_prob, class_num, t_weight=None):
        device = t_prob.device
        ns = s_label.size(0)
        nt = t_prob.size(0)
        c = int(class_num)
        s_onehot = F.one_hot(s_label.long(), num_classes=c).float().to(device)
        t_soft = t_prob.float()
        if t_weight is not None:
            t_soft = t_soft * t_weight.float().view(nt, 1)
        s_sum = s_onehot.sum(dim=0)
        t_sum = t_soft.sum(dim=0)
        common = (s_sum > 0) & (t_sum > self.eps)
        if common.sum() == 0:
            return None
        s_vec = s_onehot[:, common] / s_sum[common].clamp_min(self.eps).view(1, -1)
        t_vec = t_soft[:, common] / t_sum[common].clamp_min(self.eps).view(1, -1)
        common_num = float(common.sum().item())
        return (
            torch.mm(s_vec, s_vec.t()) / common_num,
            torch.mm(t_vec, t_vec.t()) / common_num,
            torch.mm(s_vec, t_vec.t()) / common_num,
        )

    def get_loss(self, source, target, s_label, t_prob, class_num, t_weight=None):
        ns = source.size(0)
        nt = target.size(0)
        weights = self._weights(s_label, t_prob, class_num, t_weight=t_weight)
        if weights is None:
            return source.new_tensor(0.0)
        weight_ss, weight_tt, weight_st = weights
        kernels = self.gaussian_kernel(source, target)
        if torch.isnan(kernels).any() or torch.isinf(kernels).any():
            return source.new_tensor(0.0)
        ss = kernels[:ns, :ns]
        tt = kernels[ns:ns + nt, ns:ns + nt]
        st = kernels[:ns, ns:ns + nt]
        return torch.sum(weight_ss * ss + weight_tt * tt - 2.0 * weight_st * st)


def mcc_loss_from_prob(prob, temperature=1.0, eps=1e-6):
    """MCC on a probability matrix. Temperature is applied by re-sharpening probabilities."""
    if temperature != 1.0:
        prob = sharpen_prob(prob, temperature=temperature, eps=eps)
    prob = prob.clamp_min(eps)
    entropy = -torch.sum(prob * torch.log(prob), dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = (weight / weight.sum().clamp_min(eps) * prob.size(0)).view(-1, 1)
    prob = prob * weight
    class_confusion = torch.mm(prob.t(), prob)
    class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True).clamp_min(eps)
    return (class_confusion.sum() - torch.trace(class_confusion)) / prob.size(1)


def consistency_loss_from_probs(prob_list):
    if len(prob_list) <= 1:
        return prob_list[0].new_tensor(0.0)
    if any(prob.size(1) != prob_list[0].size(1) for prob in prob_list):
        return prob_list[0].new_tensor(0.0)
    mean_prob = torch.stack(prob_list, dim=0).mean(dim=0).detach()
    return sum(F.mse_loss(prob, mean_prob) for prob in prob_list) / len(prob_list)


class Trainer(TrainerBase):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=args.in_channel).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=args.in_channel, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

        self.Fs = nn.ModuleList([
            modules.MLP(input_size=self.G.out_dim, dropout=args.dropout, num_layer=2, output_layer=False)
            for _ in range(self.num_source)
        ]).to(self.device)

        # 8-way label-powerset head.
        self.Cs = nn.ModuleList([
            modules.MLP(input_size=self.Fs[i].feature_dim, output_size=args.num_classes[i], num_layer=1, last=None)
            for i in range(self.num_source)
        ]).to(self.device)

        # 3-way axis decoupling head: axis4 / axis5 / axis6.
        self.As = nn.ModuleList([
            modules.MLP(input_size=self.Fs[i].feature_dim, output_size=3, num_layer=1, last=None)
            for i in range(self.num_source)
        ]).to(self.device)

        self.lmmd = LMMD_loss()
        self.num_class = args.num_classes
        self._init_data()
        self.src = ['concat_source'] if args.train_mode == 'source_combine' else args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs, self.As])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum(len(self.dataloaders[s]) for s in self.src)

        self.source_dist_ema = torch.ones(self.num_source, dtype=torch.float32)
        self.source_ema_momentum = float(getattr(args, 'source_ema_momentum', 0.95))

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'As': self.As.state_dict(),
            'source_dist_ema': self.source_dist_ema,
        }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])
        if 'As' in ckpt:
            self.As.load_state_dict(ckpt['As'])
        if 'source_dist_ema' in ckpt:
            self.source_dist_ema = ckpt['source_dist_ema'].float().cpu()

    def _set_to_train(self):
        self.G.train(); self.Fs.train(); self.Cs.train(); self.As.train()

    def _set_to_eval(self):
        self.G.eval(); self.Fs.eval(); self.Cs.eval(); self.As.eval()

    def _tradeoff_at(self, idx, default=0.0):
        if hasattr(self, 'tradeoff') and idx < len(self.tradeoff):
            return self.tradeoff[idx]
        return default

    def _same_class_space(self):
        return all(nc == self.num_class[0] for nc in self.num_class)

    def _axis_blend(self):
        return float(getattr(self.args, 'axis_blend', 0.35))

    def _branch_prob(self, class_logits, axis_logits, branch_idx, axis_blend=None):
        if axis_blend is None:
            axis_blend = self._axis_blend()
        p_cls = F.softmax(class_logits, dim=1)
        p_axis_canon = axis_logits_to_class_prob(axis_logits)
        p_axis = reorder_canonical_probs_to_label_set(p_axis_canon, self.args.label_sets[branch_idx])
        return (1.0 - axis_blend) * p_cls + axis_blend * p_axis

    def _branch_outputs(self, feat, branch_idx):
        z = self.Fs[branch_idx](feat)
        return z, self.Cs[branch_idx](z), self.As[branch_idx](z)

    def _target_probs_all(self, feat_t):
        probs, class_logits, axis_logits = [], [], []
        for i in range(self.num_source):
            _, c_logit, a_logit = self._branch_outputs(feat_t, i)
            class_logits.append(c_logit)
            axis_logits.append(a_logit)
            probs.append(self._branch_prob(c_logit, a_logit, i))
        return probs, class_logits, axis_logits

    def _ensemble_pseudo_prob(self, prob_list, fallback_prob):
        if self._same_class_space() and all(prob.size(1) == prob_list[0].size(1) for prob in prob_list):
            prob = torch.stack([p.detach() for p in prob_list], dim=0).mean(dim=0)
        else:
            prob = fallback_prob.detach()
        temp = float(getattr(self.args, 'pseudo_temperature', 0.5))
        return sharpen_prob(prob, temperature=temp)

    def _source_weights(self):
        dist = self.source_dist_ema.to(self.device)
        if (not torch.isfinite(dist).all()) or self.num_source <= 1:
            return [1.0] * self.num_source
        if torch.max(dist) - torch.min(dist) < 1e-6:
            return [1.0] * self.num_source
        dist_norm = (dist - dist.min()) / (dist.max() - dist.min()).clamp_min(1e-6)
        tau = float(getattr(self.args, 'source_weight_tau', 2.0))
        weights = F.softmax(-tau * dist_norm, dim=0)
        return [float(w.item()) for w in weights]

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        pseudo_threshold = float(getattr(self.args, 'pseudo_threshold', 0.0))
        label_smoothing = float(getattr(self.args, 'label_smoothing', 0.0))

        axis_loss_weight = float(getattr(self.args, 'axis_loss_weight', 0.3))
        axis_nll_weight = float(getattr(self.args, 'axis_nll_weight', 0.2))
        head_cons_weight = float(getattr(self.args, 'head_cons_weight', 0.1))
        target_head_cons_weight = float(getattr(self.args, 'target_head_cons_weight', 0.05))
        target_consistency_weight = float(getattr(self.args, 'consistency_weight', 0.2))
        mcc_weight = float(getattr(self.args, 'mcc_weight', 0.1))

        for i in tqdm(range(self.num_iter), ascii=True):
            cur_src_idx = int(i % self.num_source)
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx])

            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            feat = self.G(data)
            feat_s_base, feat_t_base = feat.chunk(2, dim=0)

            f_s = self.Fs[cur_src_idx](feat_s_base)
            f_t = self.Fs[cur_src_idx](feat_t_base)
            y_s = self.Cs[cur_src_idx](f_s)
            y_t = self.Cs[cur_src_idx](f_t)
            a_s = self.As[cur_src_idx](f_s)
            a_t = self.As[cur_src_idx](f_t)

            if label_smoothing > 0:
                loss_cls = F.cross_entropy(y_s, source_labels, label_smoothing=label_smoothing)
            else:
                loss_cls = F.cross_entropy(y_s, source_labels)

            axis_targets = source_labels_to_axis_targets(source_labels, self.args.label_sets[cur_src_idx])
            loss_axis = F.binary_cross_entropy_with_logits(a_s, axis_targets)

            p_axis_s = reorder_canonical_probs_to_label_set(
                axis_logits_to_class_prob(a_s), self.args.label_sets[cur_src_idx]
            )
            loss_axis_nll = F.nll_loss(torch.log(p_axis_s.clamp_min(1e-6)), source_labels)
            loss_head_cons_s = F.mse_loss(F.softmax(y_s, dim=1), p_axis_s.detach())

            probs_t_all, class_logits_t_all, axis_logits_t_all = self._target_probs_all(feat_t_base)
            fallback_prob = self._branch_prob(y_t, a_t, cur_src_idx)
            t_prob = self._ensemble_pseudo_prob(probs_t_all, fallback_prob)
            confidence, _ = torch.max(t_prob, dim=1)
            if pseudo_threshold > 0:
                t_weight = confidence.detach() * (confidence.detach() >= pseudo_threshold).float()
            else:
                t_weight = confidence.detach()

            loss_mmd = self.lmmd.get_loss(
                f_s, f_t, source_labels, t_prob,
                self.num_class[cur_src_idx], t_weight=t_weight
            )

            loss_target_cons = consistency_loss_from_probs(probs_t_all)
            loss_mcc = sum(mcc_loss_from_prob(p) for p in probs_t_all) / len(probs_t_all)

            target_head_cons_terms = []
            for b_idx, (c_logit, a_logit) in enumerate(zip(class_logits_t_all, axis_logits_t_all)):
                p_cls = F.softmax(c_logit, dim=1)
                p_axis = reorder_canonical_probs_to_label_set(
                    axis_logits_to_class_prob(a_logit), self.args.label_sets[b_idx]
                )
                target_head_cons_terms.append(F.mse_loss(p_cls, p_axis))
            loss_head_cons_t = sum(target_head_cons_terms) / len(target_head_cons_terms)

            loss = (
                loss_cls
                + axis_loss_weight * loss_axis
                + axis_nll_weight * loss_axis_nll
                + head_cons_weight * loss_head_cons_s
                + self._tradeoff_at(0, 1.0) * loss_mmd
                + self._tradeoff_at(1, 1.0) * target_consistency_weight * loss_target_cons
                + self._tradeoff_at(2, 1.0) * mcc_weight * loss_mcc
                + target_head_cons_weight * loss_head_cons_t
            )

            source_prob = self._branch_prob(y_s, a_s, cur_src_idx)
            epoch_acc['Source Data'] += self._get_accuracy(source_prob, source_labels)
            epoch_loss['Source Classifier'] += loss_cls.detach()
            epoch_loss['Axis BCE'] += loss_axis.detach()
            epoch_loss['Axis NLL'] += loss_axis_nll.detach()
            epoch_loss['Head Cons S'] += loss_head_cons_s.detach()
            epoch_loss['MMD'] += loss_mmd.detach()
            epoch_loss['Target Consistency'] += loss_target_cons.detach()
            epoch_loss['Target MCC'] += loss_mcc.detach()
            epoch_loss['Head Cons T'] += loss_head_cons_t.detach()
            epoch_loss['Pseudo Conf'] += confidence.detach().mean()
            epoch_loss['Pseudo Used'] += (t_weight > 0).float().mean()

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                d = loss_mmd.detach().float().cpu()
                if torch.isfinite(d):
                    old = self.source_dist_ema[cur_src_idx]
                    self.source_dist_ema[cur_src_idx] = self.source_ema_momentum * old + (1.0 - self.source_ema_momentum) * d

        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)
        probs_tgt = []
        for i in range(self.num_source):
            _, c_logit, a_logit = self._branch_outputs(feat_tgt, i)
            probs_tgt.append(self._branch_prob(c_logit, a_logit, i))
        actual_pred = self._combine_prediction(
            probs_tgt,
            idx=list(range(self.num_source)),
            weights=self._source_weights()
        )
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]
        total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(
                actual_pred, actual_labels, return_acc=False,
                idx=list(range(self.num_source)), mode='closed-set'
            )
            correct['Closed-set-acc'] += output[0]
            total['Closed-set-acc'] += output[1]
        return correct, total
