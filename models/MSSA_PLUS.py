'''
MSSA_PLUS: accuracy-oriented extension for ROBOT multi-source UDA.

Drop this file into models/MSSA_PLUS.py and run with --model_name MSSA_PLUS.
It keeps the original MSSA structure but adds:
1) confidence-weighted ensemble pseudo labels for LMMD,
2) source reliability weighting for multi-source prediction,
3) target classifier-consistency regularization,
4) MCC target regularization,
5) optional source label smoothing.
'''

import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import modules
from train_utils import TrainerBase


class LMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-6):
        super(LMMD_loss, self).__init__()
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
        """Build LMMD class weights in torch.

        s_label: [Ns], source class index in the current source label space.
        t_prob:  [Nt, C], target soft pseudo label in the same label space.
        t_weight:[Nt], optional confidence weight. Low-confidence target samples can be
                 down-weighted without changing batch shape.
        """
        device = t_prob.device
        ns = s_label.size(0)
        nt = t_prob.size(0)
        c = int(class_num)

        s_onehot = F.one_hot(s_label, num_classes=c).float().to(device)
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

        weight_ss = torch.mm(s_vec, s_vec.t()) / common_num
        weight_tt = torch.mm(t_vec, t_vec.t()) / common_num
        weight_st = torch.mm(s_vec, t_vec.t()) / common_num
        return weight_ss, weight_tt, weight_st

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


def sharpen_prob(prob, temperature=0.5, eps=1e-6):
    if temperature <= 0:
        return prob
    out = prob.clamp_min(eps).pow(1.0 / temperature)
    return out / out.sum(dim=1, keepdim=True).clamp_min(eps)


def mcc_loss(logits, temperature=2.5, eps=1e-6):
    """Minimum Class Confusion loss for unlabeled target predictions."""
    prob = F.softmax(logits / temperature, dim=1)
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(eps)), dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = (weight / weight.sum().clamp_min(eps) * logits.size(0)).view(-1, 1)
    prob = prob * weight
    class_confusion = torch.mm(prob.t(), prob)
    class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True).clamp_min(eps)
    return (class_confusion.sum() - torch.trace(class_confusion)) / logits.size(1)


def consistency_loss_from_logits(logits_list):
    if len(logits_list) <= 1:
        return logits_list[0].new_tensor(0.0)
    # Only valid when all classifiers share the same class space.
    if any(logits.size(1) != logits_list[0].size(1) for logits in logits_list):
        return logits_list[0].new_tensor(0.0)
    probs = [F.softmax(logits, dim=1) for logits in logits_list]
    mean_prob = torch.stack(probs, dim=0).mean(dim=0).detach()
    return sum(F.mse_loss(prob, mean_prob) for prob in probs) / len(probs)


class Trainer(TrainerBase):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
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
        self.Cs = nn.ModuleList([
            modules.MLP(input_size=self.Fs[i].feature_dim, output_size=args.num_classes[i], num_layer=1, last=None)
            for i in range(self.num_source)
        ]).to(self.device)

        self.lmmd = LMMD_loss()
        self.num_class = args.num_classes
        self._init_data()
        self.src = ['concat_source'] if args.train_mode == 'source_combine' else args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum(len(self.dataloaders[s]) for s in self.src)

        # EMA of source-target LMMD distance. Smaller means more reliable source.
        self.source_dist_ema = torch.ones(self.num_source, dtype=torch.float32)
        self.source_ema_momentum = float(getattr(args, 'source_ema_momentum', 0.95))

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'source_dist_ema': self.source_dist_ema,
        }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])
        if 'source_dist_ema' in ckpt:
            self.source_dist_ema = ckpt['source_dist_ema'].float().cpu()

    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()

    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()

    def _tradeoff_at(self, idx, default=0.0):
        if hasattr(self, 'tradeoff') and idx < len(self.tradeoff):
            return self.tradeoff[idx]
        return default

    def _same_class_space(self):
        return all(nc == self.num_class[0] for nc in self.num_class)

    def _target_logits_all(self, feat_t):
        return [self.Cs[i](self.Fs[i](feat_t)) for i in range(self.num_source)]

    def _ensemble_pseudo_prob(self, logits_list, fallback_logits):
        if self._same_class_space() and all(logits.size(1) == logits_list[0].size(1) for logits in logits_list):
            prob = torch.stack([F.softmax(logits.detach(), dim=1) for logits in logits_list], dim=0).mean(dim=0)
        else:
            prob = F.softmax(fallback_logits.detach(), dim=1)
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
        lambda_cons = float(getattr(self.args, 'consistency_weight', 0.2))
        lambda_mcc = float(getattr(self.args, 'mcc_weight', 0.1))
        pseudo_threshold = float(getattr(self.args, 'pseudo_threshold', 0.0))
        label_smoothing = float(getattr(self.args, 'label_smoothing', 0.0))

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

            if label_smoothing > 0:
                loss_cls = F.cross_entropy(y_s, source_labels, label_smoothing=label_smoothing)
            else:
                loss_cls = F.cross_entropy(y_s, source_labels)

            all_logits_t = self._target_logits_all(feat_t_base)
            t_prob = self._ensemble_pseudo_prob(all_logits_t, y_t)
            confidence, _ = torch.max(t_prob, dim=1)
            if pseudo_threshold > 0:
                t_weight = confidence.detach() * (confidence.detach() >= pseudo_threshold).float()
            else:
                t_weight = confidence.detach()

            loss_mmd = self.lmmd.get_loss(
                f_s, f_t, source_labels, t_prob,
                self.num_class[cur_src_idx], t_weight=t_weight
            )
            loss_cons = consistency_loss_from_logits(all_logits_t)
            loss_mcc = sum(mcc_loss(logits) for logits in all_logits_t) / len(all_logits_t)

            loss = (
                loss_cls
                + self._tradeoff_at(0, 1.0) * loss_mmd
                + self._tradeoff_at(1, 1.0) * lambda_cons * loss_cons
                + self._tradeoff_at(2, 1.0) * lambda_mcc * loss_mcc
            )

            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source Classifier'] += loss_cls.detach()
            epoch_loss['MMD'] += loss_mmd.detach()
            epoch_loss['Target Consistency'] += loss_cons.detach()
            epoch_loss['Target MCC'] += loss_mcc.detach()
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
        logits_tgt = [F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1) for i in range(self.num_source)]
        actual_pred = self._combine_prediction(
            logits_tgt,
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
