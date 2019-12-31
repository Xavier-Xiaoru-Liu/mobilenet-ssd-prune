'''
Prune lib fir general prepose
Copyright (c) Xiaoru Liu, 2019
'''

import torch
import torch.nn
import numpy as np


# Varies for different model
# mapping the name of conv to its corresponding bn
def mapping(conv_name):
    try:
        return conv_name[:-1] + str(int(conv_name[-1])+1)
    except ValueError:
        return 'None'


class InfoStruct(object):

    def __init__(self, module, pre_f_cls, f_cls, b_cls):

        # init
        self.module = module
        self.pre_f_cls = pre_f_cls
        self.f_cls = f_cls
        self.b_cls = b_cls

        # the inputs channel number
        self.in_channel_num = pre_f_cls.channel_num

        # the outputs channel number
        self.out_channel_num = b_cls.channel_num

        # forward statistic
        self.forward_mean = None
        self.variance = None
        self.forward_cov = None

        # forward info
        self.zero_variance_masked_zero = None
        self.zero_variance_masked_one = None
        self.de_correlation_variance = None

        # raw score for rldr-pruning
        self.alpha = None
        self.normalized_alpha = None
        self.stack_op_for_weight = None

        # backward statistic
        self.grad_mean = None
        self.grad_cov = None
        self.adjust_matrix = None

        # raw score for crldr-pruning
        self.beta = None

        # weights
        self.weight = None

        # corresponding bn layer
        self.bn_module = None

    def compute_score(self):
        # for a group of given self.forward_mean, self.forward_cov, self.grad_mean, self.grad_cov
        # compute de_correlation_variance
        repaired_forward_cov = self.forward_cov + torch.diag(self.zero_variance_masked_one)

        f_cov_inverse = repaired_forward_cov.inverse()
        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.de_correlation_variance = repaired_alpha * self.zero_variance_masked_zero

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        # fetch weights
        self.weight = self.module.weight.detach()

        # score for rldr-pruning
        self.alpha = \
            torch.sum(torch.pow(torch.squeeze(self.weight), 2), dim=0).to(torch.double) * self.de_correlation_variance
        self.normalized_alpha = self.alpha / torch.norm(self.alpha)

        # cascade effects
        eig_value, eig_vec = torch.eig(self.grad_cov, eigenvectors=True)
        self.adjust_matrix = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t()).to(torch.float)

        # score for crldr-pruning
        square_sum_gamma_matrix = \
            torch.sum(torch.pow(torch.mm(self.adjust_matrix, torch.squeeze(self.weight)), 2), dim=0)
        self.beta = square_sum_gamma_matrix.to(torch.double) * self.de_correlation_variance
        return self.normalized_alpha, self.beta

    def compute_statistic(self):
        # compute forward statistic
        self.forward_mean = self.f_cls.module.sum_mean / self.f_cls.module.counter
        self.forward_cov = (self.f_cls.module.sum_covariance / self.f_cls.module.counter) - \
            torch.mm(self.forward_mean.view(-1, 1), self.forward_mean.view(1, -1))

        # compute backward statistic
        self.grad_mean = self.b_cls.module.b_sum_mean / self.b_cls.module.b_counter
        self.grad_cov = (self.b_cls.module.b_sum_covariance / self.b_cls.module.b_counter) - \
            torch.mm(self.grad_mean.view(-1, 1), self.grad_mean.view(1, -1))

        # equal 0 where variance of an activate is 0
        self.variance = torch.diag(self.forward_cov)
        self.zero_variance_masked_zero = torch.sign(self.variance)

        # where 0 var compensate 1
        self.zero_variance_masked_one = - self.zero_variance_masked_zero + 1

    def clear_zero_variance(self):

        # according to zero variance mask, remove all the channels with 0 variance,
        # this function first update [masks] in pre_forward_hook,
        # then update parameters in [bn module] or biases in the last layer

        verify = int(torch.sum(self.pre_f_cls.read_mask() - self.zero_variance_masked_zero.to(torch.float) * self.pre_f_cls.read_mask()))
        tmp_verify = int(torch.sum(self.pre_f_cls.read_mask() - self.zero_variance_masked_zero.to(torch.float)))

        if verify != tmp_verify:
            raise Exception('mask zero but variance not zero.')

        if verify != 0:

            print('Number of more zero variance channel: ', verify)

            # update mask
            self.pre_f_cls.load_mask(self.zero_variance_masked_zero.to(torch.float))

            # update weight
            if len(self.module.weight.shape) == 4:
                self.module.weight.data[:, :, 0, 0] = \
                    torch.squeeze(self.module.weight) * self.zero_variance_masked_zero.to(torch.float)
            elif len(self.module.weight.shape) == 2:
                self.module.weight.data[:, :, 0, 0] = torch.squeeze(
                    self.module.weight) * self.zero_variance_masked_zero.to(torch.float)

            # update bn
            if self.bn_module is None:
                self.module.bias.data = torch.squeeze(torch.mm(torch.squeeze(self.module.weight),
                                                               self.forward_mean.to(torch.float).view(-1, 1)))
            else:
                self.bn_module.running_mean.data = torch.squeeze(torch.mm(torch.squeeze(self.module.weight),
                                                                          self.forward_mean.to(torch.float).view(-1, 1)))

    def prune_then_modify(self, index_of_channel):

        # update [mask]
        channel_mask = self.pre_f_cls.read_mask()
        channel_mask[index_of_channel] = 0
        self.pre_f_cls.load_mask(channel_mask)

        # update [weights]

        new_weight = \
            torch.squeeze(self.weight) - \
            torch.mm(self.weight[:, index_of_channel].view(-1, 1).to(torch.double),
                     self.stack_op_for_weight[index_of_channel, :].view(1, -1)).to(torch.float)

        if self.f_cls.dim == 4:
            self.weight[:, :, 0, 0] = new_weight
        else:
            self.weight[:, :] = new_weight
        self.module.weight.data = self.weight

        # update [bn]

        if self.bn_module is None:
            print('Modify biases in', self.module)
            connections = torch.squeeze(self.weight[:, index_of_channel])
            repair_base = connections * self.forward_mean[index_of_channel].to(torch.float)
            self.module.bias.data -= repair_base
        else:
            self.bn_module.running_mean.data = \
                torch.squeeze(torch.mm(new_weight, self.forward_mean.to(torch.float).view(-1, 1)))
            self.bn_module.running_var.data = \
                torch.diag(torch.mm(torch.mm(new_weight, self.forward_cov.to(torch.float)), new_weight.t()))

        # update statistic
        self.forward_cov[:, index_of_channel] = 0
        self.forward_cov[index_of_channel, :] = 0

        self.zero_variance_masked_zero = channel_mask.to(torch.double)
        self.zero_variance_masked_one = (1 - channel_mask).to(torch.double)

    def minimum_score(self, method):

        if method == 'rldr':
            score = self.normalized_alpha
        elif method == 'crldr':
            score = self.beta
        else:
            raise Exception('method must be rldr or crldr')

        sorted_index = torch.argsort(score)

        channel_mask = self.pre_f_cls.read_mask()
        for index in list(np.array(sorted_index.cpu())):
            index = int(index)
            if int(channel_mask[index]) != 0:
                return index, float(score[index])

    def reset(self):
        self.f_cls.reset()
        self.b_cls.reset()

    def query_channel_num(self):

        channel_mask = self.pre_f_cls.read_mask()

        return int(torch.sum(channel_mask).cpu()), int(channel_mask.shape[0])


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples = samples.to(torch.half).to(torch.double)
    samples_num = list(samples.shape)[0]
    counter += samples_num
    sum_mean += torch.sum(samples, dim=0)
    sum_covar += torch.mm(samples.permute(1, 0), samples)


class ForwardStatisticHook(object):

    def __init__(self, name, module, dim=4):
        self.name = name
        self.dim = dim
        self.module = module

        if dim == 4:
            channel_num = module.in_channels
        elif dim == 2:
            channel_num = module.in_features
        else:
            raise Exception('dim must be 2 or 4')

        self.channel_num = channel_num

        module.register_buffer('sum_mean', torch.zeros(channel_num, dtype=torch.double))
        module.register_buffer('sum_covariance', torch.zeros([channel_num, channel_num], dtype=torch.double))
        module.register_buffer('counter', torch.zeros(1, dtype=torch.double))

    def __call__(self, module, inputs, output) -> None:
        with torch.no_grad():
            channel_num = self.channel_num
            # from [N,C,W,H] to [N*W*H,C]
            if self.dim == 4:
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = inputs[0]
            compute_statistic_and_update(samples, module.sum_mean, module.sum_covariance, module.counter)

    def reset(self):
        self.module.sum_mean.zero_()
        self.module.sum_covariance.zero_()
        self.module.counter.zero_()


class BackwardStatisticHook(object):

    def __init__(self, name, module, dim=4):
        self.name = name
        self.dim = dim
        self.module = module

        if dim == 4:
            channel_num = module.out_channels
        elif dim == 2:
            channel_num = module.out_features
        else:
            raise Exception('dim must be 2 or 4')

        self.channel_num = channel_num

        module.register_buffer('b_sum_mean', torch.zeros(channel_num, dtype=torch.double))
        module.register_buffer('b_sum_covariance', torch.zeros([channel_num, channel_num], dtype=torch.double))
        module.register_buffer('b_counter', torch.zeros(1, dtype=torch.double))

    def __call__(self, module, grad_input, grad_output) -> None:
        with torch.no_grad():
            channel_num = self.channel_num

            if self.dim == 4:
                samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = grad_output[0]
            compute_statistic_and_update(samples, module.b_sum_mean, module.b_sum_covariance, module.b_counter)

    def reset(self):
        self.module.b_sum_mean.zero_()
        self.module.b_sum_covariance.zero_()
        self.module.b_counter.zero_()


class PreForwardHook(object):

    def __init__(self, name, module, dim=4):
        self.name = name
        self.dim = dim
        self.module = module
        if dim == 4:
            self.channel_num = module.in_channels
        elif dim == 2:
            self.channel_num = module.in_features
        module.register_buffer('mask', torch.ones(self.channel_num))

    def __call__(self, module, inputs):
        if self.dim == 4:
            modified = torch.mul(inputs[0].permute([0, 2, 3, 1]), module.mask)
            return modified.permute([0, 3, 1, 2])
        elif self.dim == 2:
            return torch.mul(inputs[0], module.mask)
        else:
            raise Exception

    def read_mask(self):
        return self.module.mask.data

    def load_mask(self, data):
        self.module.mask.data = data


class MaskManager(object):

    def __init__(self, using_statistic=True):

        self.name_to_statistic = {}
        self.name_module = {}
        self.bn_name = {}
        self.statistic = using_statistic

    def __call__(self, model):

        for name, sub_module in model.named_modules():

            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    pre_hook_cls = PreForwardHook(name, sub_module)
                    sub_module.register_forward_pre_hook(pre_hook_cls)
                    self.name_module[name] = sub_module
                    if self.statistic:
                        hook_cls = ForwardStatisticHook(name, sub_module)
                        back_hook_cls = BackwardStatisticHook(name, sub_module)
                        sub_module.register_forward_hook(hook_cls)
                        sub_module.register_backward_hook(back_hook_cls)
                        self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)

            elif isinstance(sub_module, torch.nn.Linear):
                pre_hook_cls = PreForwardHook(name, sub_module, dim=2)
                sub_module.register_forward_pre_hook(pre_hook_cls)
                self.name_module[name] = sub_module
                if self.statistic:
                    hook_cls = ForwardStatisticHook(name, sub_module, dim=2)
                    back_hook_cls = BackwardStatisticHook(name, sub_module, dim=2)
                    sub_module.register_forward_hook(hook_cls)
                    sub_module.register_backward_hook(back_hook_cls)
                    self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)

            elif isinstance(sub_module, torch.nn.BatchNorm1d) or isinstance(sub_module, torch.nn.BatchNorm2d):
                self.bn_name[name] = sub_module
                # print('bn', name)

    def computer_statistic(self):

        with torch.no_grad():

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                if mapping(name) in self.bn_name:
                    info.bn_module = self.bn_name[mapping(name)]

                info.compute_statistic()

                info.clear_zero_variance()

                info.compute_score()

    def prune(self, pruned_num, method='rldr'):

        for _ in range(pruned_num):

            min_score = 1000
            the_info = None
            the_name = None
            index = None

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                idx, score = info.minimum_score(method)
                if score < min_score:
                    min_score = score
                    the_info = info
                    the_name = name
                    index = idx

            print('pruned score: ', min_score, 'name: ', the_name)
            the_info.prune_then_modify(index)
            the_info.compute_score()

    def prune_by_threshold(self, threshold, method='rldr'):

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]

            idx, score = info.minimum_score(method)
            while score < threshold:
                print('pruned score: ', score, 'name: ', name)
                info.prune_then_modify(idx)
                info.compute_score()
                idx, score = info.minimum_score(method)

    def pruning_overview(self):

        if self.statistic:
            all_channel_num = 0
            remained_channel_num = 0

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]
                r, a = info.query_channel_num()
                all_channel_num += a
                remained_channel_num += r

            print('channel number: ', remained_channel_num, '/', all_channel_num)
            return remained_channel_num, all_channel_num
        else:
            all_channel_num = 0
            remained_channel_num = 0

            for name in self.name_module:
                channel_mask = self.name_module[name].mask.data
                r = int(torch.sum(channel_mask).cpu())
                a = int(channel_mask.shape[0])

                all_channel_num += a
                remained_channel_num += r
                print(name, r, '/', a)

            print('channel number: ', remained_channel_num, '/', all_channel_num)

    def reset(self):

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]

            info.reset()

    def visualize(self):

        from matplotlib import pyplot as plt
        i = 1
        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            plt.subplot(10, 15, i)
            plt.imshow(np.array(forward_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            plt.subplot(10, 15, i)
            plt.imshow(np.array(grad_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 150:
                break
        plt.show()
