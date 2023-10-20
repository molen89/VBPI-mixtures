import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import random
import numpy as np
from utils import namenum
from base_branchModel import BaseModel
from vector_sbnModel import SBN
from phyloModel import PHY
import logging
from deep_branchModel import DeepModel
import wandb


class mixTreeBranchVBPI(nn.Module):
    EPS = np.finfo(float).eps

    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel,
                 scale=0.1, psp=True, feature_dim=2, S=2, use_nf=False, hidden_sizes=[50], flow_type='planar', mixture_type="multi", num_of_layers_nf=16):
        super().__init__()
        print("Init mixTreeBranchVBPI")
        self.taxa = taxa
        self.ntips = len(data)
        self.scale = scale
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale)
        self.log_p_tau = - np.sum(np.log(np.arange(3, 2*self.ntips-3, 2)))

        self.log_w_tilde = torch.log(torch.ones(S)/S)

        self.use_nf = use_nf
        self.mixture_type = mixture_type
        if mixture_type == "multi":
            self.S_tree = S
            self.S_branch = S
        elif mixture_type == "multi_branch":
            self.S_tree = 1
            self.S_branch = S
        elif mixture_type == "multi_tree":
            self.S_tree = S
            self.S_branch = 1

        self.S = torch.tensor(S)
        self.logS = torch.log(torch.tensor(S))
        self.tree_model = [SBN(taxa, rootsplit_supp_dict[s],
                               subsplit_supp_dict[s]) for s in range(self.S_tree)]

        self.rs_embedding_map = [
            self.tree_model[s].rs_map for s in range(self.S_tree)]
        self.ss_embedding_map = [
            self.tree_model[s].ss_map for s in range(self.S_tree)]

        base_model = [
            BaseModel(self.ntips, self.rs_embedding_map[0 if self.S_tree == 1 else s],
                      self.ss_embedding_map[0 if self.S_tree == 1 else s], psp=psp, feature_dim=feature_dim)
            for s in range(self.S_branch)]

        if self.use_nf:
            print("using NF")
            self.branch_model = DeepModel(self.ntips, base_model, psp=psp, hidden_sizes=hidden_sizes,
                                          feature_dim=feature_dim, flow_type=flow_type, num_of_layers_nf=num_of_layers_nf)
        else:
            self.branch_model = base_model

        logging.info("Model initialized finished")

    def print_parameters(self, add_wandb=False, lf=1000):
        logging.info('Parameter Info:')

        if self.use_nf:
            for bm in self.branch_model.mean_std_encoder.mean_std_model:
                if add_wandb:
                    wandb.watch(bm, log_freq=lf)
                for param in bm.parameters():
                    logging.info(
                        f"base-model: {param.dtype}, {param.size()}, {torch.norm(param)}")

            if add_wandb:
                wandb.watch(self.branch_model, log_freq=lf)
            for param in self.branch_model.parameters():
                logging.info(
                    f"flow-model: {param.dtype}, {param.size()}, {torch.norm(param)}")
        else:
            for bm in self.branch_model:
                if add_wandb:
                    wandb.watch(bm, log_freq=lf)
                for param in bm.parameters():
                    logging.info(
                        f"branch-model: {param.dtype}, {param.size()}, {torch.norm(param)}")

        for sbn in self.tree_model:
            if add_wandb:
                wandb.watch(sbn, log_freq=lf)
            for param in sbn.parameters():
                logging.info(
                    f"tree model: {param.dtype}, {param.size()}, {torch.norm(param)}")

    def load_from(self, state_dict_path):
        print("loading...")
        if self.use_nf:
            nf_path = state_dict_path.replace('.pt', f'_nf.pt')
            self.branch_model.load_state_dict(torch.load(nf_path))
            for s in range(self.S_tree):
                sbn_path = state_dict_path.replace('.pt', f'_sbn_{s}.pt')
                self.tree_model[s].load_state_dict(torch.load(sbn_path))
            for s in range(self.S_branch):
                bm_path = state_dict_path.replace('.pt', f'_bm_{s}.pt')
                self.branch_model.mean_std_encoder.mean_std_model[s].load_state_dict(
                    torch.load(bm_path))

        else:

            for s in range(self.S_tree):
                sbn_path = state_dict_path.replace('.pt', f'_sbn_{s}.pt')
                self.tree_model[s].load_state_dict(torch.load(sbn_path))
            for s in range(self.S_branch):
                bm_path = state_dict_path.replace('.pt', f'_bm_{s}.pt')
                self.branch_model[s].load_state_dict(torch.load(bm_path))

        self.eval()
        [sbn.update_CPDs() for sbn in self.tree_model]

    def save(self, save_to_path):
        print(f"saving to {save_to_path}")
        if self.use_nf:  # mix of base with flows
            nf_path = save_to_path.replace('.pt', f'_nf.pt')
            torch.save(self.branch_model.state_dict(), nf_path)

            for s in range(self.S_tree):
                sbn_path = save_to_path.replace('.pt', f'_sbn_{s}.pt')
                torch.save(self.tree_model[s].state_dict(), sbn_path)
            for s in range(self.S_branch):
                bm_path = save_to_path.replace('.pt', f'_bm_{s}.pt')
                torch.save(
                    self.branch_model.mean_std_encoder.mean_std_model[s].state_dict(), bm_path)

        else:  # mix of base
            for s in range(self.S_tree):
                sbn_path = save_to_path.replace('.pt', f'_sbn_{s}.pt')
                torch.save(self.tree_model[s].state_dict(), sbn_path)
            for s in range(self.S_branch):
                bm_path = save_to_path.replace('.pt', f'_bm_{s}.pt')
                torch.save(self.branch_model[s].state_dict(), bm_path)

    def logq_tree_s(self, s, tree):
        return self.tree_model[s](tree)

    def sample_branch(self, samp_trees, s):
        n_particles = len(samp_trees)

        # MIXTURE OF BASE WITH NF
        if self.use_nf:

            samp_log_branch_0, samp_log_branch, logq_branch_0, logq_det = self.branch_model(
                samp_trees, s, return_components=True)

            def logq_tree_branch(tree, log_branch_0):

                logq_t_tmp = [self.tree_model[j](
                    tree) for j in range(self.S_tree)]
                logq_b_tmp = [self.branch_model.mean_std_encoder.mean_std_model[j].logq_branch_tree(
                    tree, log_branch_0, shift=0.0) for j in range(self.S_branch)]
                return torch.stack(logq_t_tmp), torch.stack(logq_b_tmp)

            logq_t, logq_b = zip(
                *[logq_tree_branch(samp_trees[k], samp_log_branch_0[k]) for k in range(n_particles)])

        # MIXTURE OF BASE
        else:
            samp_log_branch, logq_branch_0 = self.branch_model[s](samp_trees)

            def logq_tree_branch(tree, log_branch):
                logq_t_tmp = [self.tree_model[j](
                    tree) for j in range(self.S_tree)]
                logq_b_tmp = [self.branch_model[j].logq_branch_tree(
                    tree, log_branch) for j in range(self.S_branch)]
                return torch.stack(logq_t_tmp), torch.stack(logq_b_tmp)

            logq_t, logq_b = zip(
                *[logq_tree_branch(samp_trees[k], samp_log_branch[k]) for k in range(n_particles)])
            logq_det = torch.zeros(n_particles)

        return samp_log_branch, torch.stack(logq_t), torch.stack(logq_b), logq_det

    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10, S_tmp=None):
        rtrick_loss_and_vimco_second_term = 0
        vimco_loss_first_term = 0
        lower_bound_monitoring = []
        max_logll_monitoring = []
        logqt_monitoring = []
        logqb_monitoring = []
        logqdet_monitoring = []
        if S_tmp is None:
            S_tmp = range(self.S)

        for s in S_tmp:
            s_t = 0 if self.S_tree == 1 else s
            s_b = 0 if self.S_branch == 1 else s

            samp_trees = [self.tree_model[s_t].sample_tree()
                          for particle in range(n_particles)]
            [namenum(tree, self.taxa) for tree in samp_trees]

            # (K,n_branches), (K,S),(K,S),(K), if not nf then logq_det=0
            samp_log_branch, logq_t, logq_b, logq_det = self.sample_branch(
                samp_trees, s_b)
            logq_tree_branch = torch.logsumexp(
                logq_t + logq_b, 1) + logq_det - self.logS  # (K)

            # likelihood
            logll = torch.stack([self.phylo_model.loglikelihood(
                log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
            logp_prior = self.phylo_model.logprior(samp_log_branch)

            # only for monotoring
            lower_bound_monitoring.append(torch.logsumexp(
                logll + logp_prior - logq_tree_branch + self.log_p_tau - math.log(n_particles), 0).detach())
            max_logll_monitoring.append(torch.max(logll).detach())
            logqt_monitoring.append(logq_t.detach().mean())
            logqb_monitoring.append(logq_b.detach().mean())
            logqdet_monitoring.append(logq_det.detach().mean())

            # loss branches r-trick
            logp_joint_temp = inverse_temp * logll + logp_prior
            rtrick_loss_and_vimco_second_term += (torch.logsumexp(logp_joint_temp - logq_tree_branch - math.log(
                n_particles), dim=0)) / self.S  # hat L (tempered lower bound)

            # loss tree vimco
            logp_joint = inverse_temp * logll + logp_prior
            logq_tree_s = torch.stack(
                [self.logq_tree_s(s_t, tree) for tree in samp_trees])
            l_signal = logp_joint - logq_tree_branch  # f(τ,q)
            if n_particles > 1:
                mean_exclude_signal = (
                    torch.sum(l_signal) - l_signal) / (n_particles - 1.)  # hat_f(τ,q)
                control_variates = torch.logsumexp(
                    l_signal.view(-1, 1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(
                        n_particles), dim=0)  # log 1/K (Σf + hat_f)
            else:
                control_variates = torch.zeros(n_particles)

            temp_lower_bound = torch.logsumexp(
                l_signal - math.log(n_particles), dim=0)  # hat L
            vimco_loss_first_term += torch.sum(
                (temp_lower_bound - control_variates).detach() * logq_tree_s, dim=0) / self.S

        loss = -rtrick_loss_and_vimco_second_term - vimco_loss_first_term
        return loss, rtrick_loss_and_vimco_second_term, vimco_loss_first_term, lower_bound_monitoring, max_logll_monitoring, torch.stack(logqt_monitoring), torch.stack(logqb_monitoring), torch.stack(logqdet_monitoring)

    def lower_bound(self, n_particles=1, n_runs=1000, return_mean=True, S_tmp=None):
        lower_bounds = []
        lll = []
        logpp = []
        logqt = []
        logqb = []
        logqdet = []
        if S_tmp is None:
            S_tmp = range(self.S)

        with torch.no_grad():
            for run in range(n_runs):
                miselbo = 0
                for s in S_tmp:
                    s_t = 0 if self.S_tree == 1 else s
                    s_b = 0 if self.S_branch == 1 else s

                    samp_trees = [self.tree_model[s_t].sample_tree()
                                  for particle in range(n_particles)]  # shape=(K)
                    [namenum(tree, self.taxa) for tree in samp_trees]

                    # (K,n_branches), (K,S),(K,S),(K), if not nf then logq_det=0
                    samp_log_branch, logq_t, logq_b, logq_det = self.sample_branch(
                        samp_trees, s_b)
                    logq_tree_branch = torch.logsumexp(
                        logq_t + logq_b, 1) + logq_det - self.logS  # (K)

                    logll = torch.stack([self.phylo_model.loglikelihood(
                        log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                    logp_prior = self.phylo_model.logprior(samp_log_branch)

                    miselbo += torch.logsumexp(logll + logp_prior - logq_tree_branch +
                                               self.log_p_tau - math.log(n_particles), 0) / self.S
                    lll.append(logll.mean())
                    logpp.append(logp_prior.mean())
                    logqt.append(logq_t.mean())
                    logqb.append(logq_b.mean())
                    logqdet.append(logq_det)
                lower_bounds.append(miselbo)

        if return_mean:
            return torch.stack(lower_bounds).mean().item(), torch.stack(lower_bounds).std().item()
        else:
            return torch.stack(lower_bounds), torch.stack(lll), torch.stack(logpp), torch.stack(logqt), torch.stack(logqb), torch.stack(logqdet)

    def lower_bound_miselbo(self, n_particles=1, n_runs=1000, opt_w=False, stepz=0.001, iters=100000, opt="sgd", S_tmp=None):
        lower_bounds = []
        lll = torch.zeros((n_runs, n_particles, self.S))
        logpp = torch.zeros((n_runs, n_particles, self.S))
        logqt = torch.zeros((n_runs, n_particles, self.S, self.S))
        logqb = torch.zeros((n_runs, n_particles, self.S, self.S))
        logqdet = torch.zeros((n_runs, n_particles, self.S))

        if S_tmp is None:
            S_tmp = range(self.S)
        with torch.no_grad():
            for run in range(n_runs):
                miselbo = 0
                for s in S_tmp:
                    s_t = 0 if self.S_tree == 1 else s
                    s_b = 0 if self.S_branch == 1 else s

                    samp_trees = [self.tree_model[s_t].sample_tree()
                                  for particle in range(n_particles)]  # shape=(K)
                    [namenum(tree, self.taxa) for tree in samp_trees]

                    # (K,n_branches), (K,S),(K,S),(K), if not nf then logq_det=0
                    samp_log_branch, logq_t, logq_b, logq_det = self.sample_branch(
                        samp_trees, s_b)
                    logq_tree_branch = torch.logsumexp(
                        logq_t + logq_b, 1) + logq_det - self.logS  # (K)

                    logll = torch.stack([self.phylo_model.loglikelihood(
                        log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                    logp_prior = self.phylo_model.logprior(samp_log_branch)

                    miselbo += torch.logsumexp(logll + logp_prior - logq_tree_branch +
                                               self.log_p_tau - math.log(n_particles), 0) / self.S
                    lll[run, :, s] = logll
                    logpp[run, :, s] = logp_prior + self.log_p_tau
                    logqt[run, :, s, :] = logq_t
                    logqb[run, :, s, :] = logq_b
                    logqdet[run, :, s] = logq_det
                lower_bounds.append(miselbo)

        sgd_iterations = iters if opt_w else 1
        log_w = torch.nn.parameter.Parameter(self.log_w_tilde)
        if opt == "sgd":
            optimizer = torch.optim.SGD([{'params': log_w, 'lr': stepz}])
        elif opt == "adam":
            optimizer = torch.optim.Adam([{'params': log_w, 'lr': stepz}])

        neg_weighted_miselbos = []
        log_w_tildes = []

        for i in range(sgd_iterations):
            log_w_tilde = log_w - torch.logsumexp(log_w, dim=-1)
            log_denominator = torch.logsumexp(
                logqb + logqt + log_w_tilde, dim=-1) + logqdet
            log_numerator = lll + logpp
            log_f = log_numerator - log_denominator
            L_hat = torch.logsumexp(log_f - math.log(n_particles), dim=1)
            neg_weighted_miselbo = - \
                torch.sum(log_w_tilde.exp() * torch.mean(L_hat, 0))

            neg_weighted_miselbos.append(neg_weighted_miselbo.detach().item())
            log_w_tildes.append(log_w_tilde.detach())

            if opt_w:
                neg_weighted_miselbo.backward()
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            idx = np.argmin(neg_weighted_miselbos)
            log_w_tilde = log_w_tildes[idx]
            log_denominator = torch.logsumexp(
                logqb + logqt + log_w_tilde, dim=-1) + logqdet
            log_numerator = lll + logpp
            log_f = log_numerator - log_denominator
            L_hat = torch.logsumexp(log_f - math.log(n_particles), dim=1)
            weighted_lower_bounds = torch.sum(
                log_w_tilde.exp() * L_hat, dim=-1)
            mean_weighted_lower_bound = torch.mean(
                weighted_lower_bounds).item()
            std_weighted_lower_bound = torch.std(weighted_lower_bounds).item()

        logging.info(f"w_tilde = {log_w_tilde.exp()}")
        logging.info(f"logw_tilde = {log_w_tilde}")

        self.log_w_tilde = log_w_tilde.detach()
        return torch.stack(lower_bounds).mean().item(), torch.stack(lower_bounds).std().item(), mean_weighted_lower_bound, std_weighted_lower_bound, self.log_w_tilde

    def sample_topologies(self, s, n_particles=1):
        s_t = 0 if self.S_tree == 1 else s

        samp = [self.tree_model[s_t].sample_tree().detach()
                for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp]
        return samp

    def sample_trees(self, s, n_particles=1):
        s_t = 0 if self.S_tree == 1 else s

        samp_trees = [self.tree_model[s_t].sample_tree()
                      for particle in range(n_particles)]  # t_s ~ q_s(t)
        [namenum(tree, self.taxa) for tree in samp_trees]

        samp_log_branch, _, _, _ = self.sample_branch(samp_trees, s_t)

        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in
                             zip(*[samp_log_branch, samp_trees])])  # p(Y|t_s, b)
        logq_tree = torch.tensor([self.logq_tree(tree) for tree in samp_trees])

        return samp_trees, samp_log_branch, logll, logq_tree

    def learn(self, stepsz, maxiter=100000, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, n_particles=10,
              init_inverse_temp=0.001, warm_start_interval=50000, checkpoint_freq=-1, method='vimco', save_to_path=None, optimizer='adam'):

        print(f"Saving model at {save_to_path}")
        self.print_parameters(add_wandb=True, lf=test_freq)
        lbs, lls = [], []
        test_lb = []

        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}

        opt = [{'params': sbn.parameters(), 'lr': stepsz['tree']}
               for sbn in self.tree_model]
        if self.use_nf:
            opt += [{'params': bm.parameters(), 'lr': stepsz['branch']}
                    for bm in self.branch_model.mean_std_encoder.mean_std_model]  # base models
            # flow model
            opt += [{'params': self.branch_model.parameters(),
                     'lr': stepsz['branch']}]
        else:
            opt += [{'params': bm.parameters(), 'lr': stepsz['branch']}
                    for bm in self.branch_model]

        if optimizer == "adam":
            optimizer = torch.optim.Adam(opt)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(opt)

        logging.info(optimizer)

        run_time_tot = 0.
        for it in range(1, maxiter+1):
            inverse_temp = min(1., init_inverse_temp + it *
                               1.0/warm_start_interval)  # beta warmup
            iter_time = -time.time()
            if method == 'vimco':
                loss, rtrick_loss_and_vimco_second_term, vimco_loss_first_term, lower_bound_monitoring, max_logll_monitoring, logqt_monitoring, logqb_monitoring, logqdet_monitoring = self.vimco_lower_bound(
                    inverse_temp=inverse_temp, n_particles=n_particles)
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            [sbn.update_CPDs() for sbn in self.tree_model]

            iter_time += time.time()
            run_time_tot += iter_time

            lbs.append(np.mean(lower_bound_monitoring))
            lls.append(np.max(max_logll_monitoring))

            wandb.log({"iter": it,
                       "rtrick_vimcosecond_loss": -rtrick_loss_and_vimco_second_term,
                       "vimco_first_loss": -vimco_loss_first_term,
                       "joint_loss": loss.detach().item(),
                       "train lower bound - avg": lbs[-1],
                       "train lower bound - std": np.std(lower_bound_monitoring),
                       "train max loglikelihood": lls[-1],
                       "train logqt": logqt_monitoring.mean(),
                       "train logqb": logqb_monitoring.mean(),
                       "train logqdet": logqdet_monitoring.mean(),
                       "train iter time": iter_time,
                       },
                      step=it
                      )

            if it % test_freq == 0:
                with torch.no_grad():
                    logging.info(f'Iter {it}:({iter_time:.1f}s) (tot: {run_time_tot:.1f}s)'
                                 f'Lower Bound: {np.mean(lbs):.4f}±{np.std(lbs):.2f} | '
                                 f'Loglikelihood: {np.max(lls):.4f}')

                    if it % lb_test_freq == 0:
                        test_lower_bounds, test_lll, test_logpp, test_logqt, test_logqb, test_logqdet = self.lower_bound(
                            n_particles=1, n_runs=1000, return_mean=False)

                        test_lb.append(test_lower_bounds.mean())
                        logging.info(
                            f'>>> Iter {it}:({iter_time:.1f}s) (tot: {run_time_tot:.1f}s) Test Lower Bound: {test_lb[-1]:.4f}±{test_lower_bounds.std():.2f}')
                        self.print_parameters()

                        wandb.log({"iter": it,
                                   "test lower bound - avg": test_lb[-1],
                                   "test lower bound - std": test_lower_bounds.std(),
                                   "test logll": test_lll.mean(),
                                   "test logpp": test_logpp.mean(),
                                   "test logqt": test_logqt.mean(),
                                   "test logqb": test_logqb.mean(),
                                   "test logqdet": test_logqdet.mean(),
                                   },
                                  step=it
                                  )
                        wandb.log({f"lr(g{i})": g['lr'] for i, g in enumerate(
                            optimizer.param_groups)}, step=it)

                    lbs, lls = [], []

            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate

            if checkpoint_freq > 0:
                if it % checkpoint_freq == 0 and save_to_path is not None:
                    self.save(save_to_path.replace(
                        '.pt', 'checkpoint_{}.pt'.format(it)))

        wandb.log({"iter": it,
                   "run time total": run_time_tot,
                   },
                  step=it
                  )

        if save_to_path is not None:
            self.save(save_to_path)

        return test_lb
