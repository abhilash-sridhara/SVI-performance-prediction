# Based on Lalor's implementation of SVI for IRT estimation https://github.com/nd-ball/py-irt

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
import numpy as np
from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.optim import Adam, SGD 
from functools import partial
from scipy.special import expit 


class SVIIRT:
    def __init__(self, device, num_items, num_models,verbose=False):

        self.device = device
        self.num_items = num_items
        self.num_models = num_models        
        self.verbose = verbose

    def model_vague(self, models, items, obs):
        t_dims = pyro.plate('theta_dim', self.num_models, device=self.device)        
        b_dims = pyro.plate('beta_dim',self.num_items,device=self.device)
        with t_dims:
            ability = pyro.sample('theta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device))) 

        with b_dims:
            diff = pyro.sample('beta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))
            disc = pyro.sample('alpha', dist.Normal(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))

        with pyro.plate('observe_data', obs.size(0), device=self.device):
            logit_trasforms = (ability[models] - diff[items])*disc[items]
            pyro.sample("obs", dist.Bernoulli(logits=logit_trasforms), obs=obs)

    def guide_vague(self, models, items, obs):
        # register learnable params in the param store
        m_theta_param = pyro.param("loc_ability", torch.zeros((self.num_models), device=self.device))
        s_theta_param = pyro.param("scale_ability", torch.ones((self.num_models), device=self.device),
                            constraint=constraints.positive)
        m_b_param = pyro.param("loc_diff", torch.zeros((self.num_items), device=self.device))
        s_b_param = pyro.param("scale_diff", torch.empty((self.num_items), device=self.device).fill_(1.5),
                                constraint=constraints.positive)

        m_a_param = pyro.param("loc_alpha", torch.ones((self.num_items), device=self.device))
        s_a_param = pyro.param("scale_alpha", torch.empty((self.num_items), device=self.device).fill_(0.85),
                                constraint=constraints.positive)

        # guide distributions
        t_dims = pyro.plate('theta_dim', self.num_models, device=self.device)        
        b_dims = pyro.plate('beta_dim',self.num_items,device=self.device)
        with t_dims:
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample('theta', dist_theta)
        with b_dims:
            dist_b = dist.Normal(m_b_param, s_b_param)
            dist_a = dist.Normal(m_a_param, s_a_param)
            pyro.sample('beta', dist_b)
            pyro.sample('alpha',dist_a)

    def fit(self, models, items, responses, num_epochs):
        optim = Adam({'lr': 0.1})
        svi = SVI(self.model_vague, self.guide_vague, optim, loss=Trace_ELBO())
        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        # if self.verbose:
        #     print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        # values = ['loc_diff', 'scale_diff', 'loc_ability', 'scale_ability']



    def summary(self, traces, sites):
        marginal = EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1) \
                [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats

class SVIIPFC:
    def __init__(self, device, num_items, num_skills, verbose=False):
        self.device = device
        self.num_items = num_items
        self.verbose = verbose
        self.num_skills = num_skills
       
    def model_vague(self, items, obs, skills, skill_attempts, skill_wins, item_attempts,
                    item_wins, total_attempts, total_wins):
        counter = np.arange(items.shape[0], dtype=np.int16)
        b_dims = pyro.plate('beta_dim', self.num_items, device=self.device)
        skill_dims = pyro.plate(
            'skill_dim', self.num_skills, device=self.device)

        with b_dims:
            diff = pyro.sample('beta', dist.Normal(torch.tensor(
                0., device=self.device), torch.tensor(1.e1, device=self.device)))

        with skill_dims:
            psi_skill = pyro.sample('psi_skill',dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))
            skill_w = pyro.sample('sk_w',dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))
            skill_a = pyro.sample('sk_a',dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))

        p_skill = psi_skill.repeat((items.shape[0], 1))
        skill_w = skill_w.repeat((items.shape[0], 1))
        skill_a = skill_a.repeat((items.shape[0], 1))

        item_w = pyro.sample('i_w', dist.Normal(torch.tensor(
            0., device=self.device), torch.tensor(1., device=self.device)))
        item_a = pyro.sample('i_a', dist.Normal(torch.tensor(
            0., device=self.device), torch.tensor(1., device=self.device)))
        total_w = pyro.sample('t_w', dist.Normal(torch.tensor(
            0., device=self.device), torch.tensor(1., device=self.device)))
        total_a = pyro.sample('t_a', dist.Normal(torch.tensor(
            0., device=self.device), torch.tensor(1., device=self.device)))
        with pyro.plate('observe_data', obs.size(0), device=self.device):
            logit_trasforms = diff[items]
            logit_trasforms -= torch.sum(skills[counter]*(p_skill), dim=1)
            logit_trasforms -= torch.sum(skill_wins[counter]*(skill_w), dim=1) + torch.sum(
                skill_attempts[counter]*(skill_a), dim=1)
            logit_trasforms -= torch.flatten(item_attempts) * \
                item_a + torch.flatten(item_wins)*item_w
            logit_trasforms -= torch.flatten(total_attempts) * \
                total_a + torch.flatten(total_wins)*total_w
    
    def fit(self, items, obs, skills, skill_attempts, skill_wins, item_attempts, item_wins, 
            total_attempts, total_wins, num_epochs=800):
        optim = Adam({'lr': 0.15})
        svi = SVI(self.model_vague, self.guide_vague, optim, loss=Trace_ELBO())        
        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(items,obs,skills,skill_attempts,skill_wins,item_attempts,item_wins,total_attempts,total_wins)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        # if self.verbose:
        #     print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        # values = ['loc_diff', 'scale_diff', 'loc_ability', 'scale_ability']

    def guide_vague(self, items, obs, skills, skill_attempts, skill_wins, item_attempts,
                    item_wins, total_attempts, total_wins):
        # register learnable params in the param store

        m_b_param = pyro.param("loc_diff", torch.zeros((self.num_items), device=self.device))
        s_b_param = pyro.param("scale_diff", torch.empty((self.num_items), device=self.device).fill_(1.),
                               constraint=constraints.positive)
        m_sk_param = pyro.param('loc_skill',torch.zeros((self.num_skills), device=self.device))
        s_sk_param = pyro.param("scale_skill", torch.empty((self.num_skills), device=self.device).fill_(1./self.num_skills),
                                constraint=constraints.positive)
        m_sk_w_param = pyro.param('loc_sk_win',torch.zeros((self.num_skills), device=self.device))
        s_sk_w_param = pyro.param("scale_sk_win", torch.empty((self.num_skills), device=self.device).fill_(1./self.num_skills),
                                  constraint=constraints.positive)
        m_sk_a_param = pyro.param('loc_sk_attempt',torch.zeros((self.num_skills), device=self.device))
        s_sk_a_param = pyro.param("scale_sk_attempt", torch.empty((self.num_skills), device=self.device).fill_(1./self.num_skills),
                                  constraint=constraints.positive)
        m_ic_a_param = pyro.param('loc_ic_attempt',torch.zeros((1), device=self.device))
        s_ic_a_param = pyro.param("scale_ic_attempt", torch.empty((1), device=self.device).fill_(.1),
                                  constraint=constraints.positive)
        m_tc_a_param = pyro.param('loc_tc_attempt',torch.zeros((1), device=self.device))
        s_tc_a_param = pyro.param("scale_tc_attempt", torch.empty((1), device=self.device).fill_(.1),
                                  constraint=constraints.positive)
        m_ic_w_param = pyro.param('loc_ic_win',torch.zeros((1), device=self.device))
        s_ic_w_param = pyro.param("scale_ic_win", torch.empty((1), device=self.device).fill_(.1),
                                  constraint=constraints.positive)
        m_tc_w_param = pyro.param('loc_tc_win',torch.zeros((1), device=self.device))
        s_tc_w_param = pyro.param("scale_tc_win", torch.empty((1), device=self.device).fill_(.1),
                                  constraint=constraints.positive)
       
        # guide distributions
           
        b_dims = pyro.plate('beta_dim', self.num_items, device=self.device)
        skill_dims = pyro.plate(
            'skill_dim', self.num_skills, device=self.device)
        
        with b_dims:
            dist_b = dist.Normal(m_b_param, s_b_param)
            pyro.sample('beta', dist_b)

        with skill_dims:
            dist_skill = dist.Normal(m_sk_param, s_sk_param)
            pyro.sample('psi_skill', dist_skill)
            dist_sk_w = dist.Normal(m_sk_w_param, s_sk_w_param)
            pyro.sample('sk_w', dist_sk_w)
            dist_sk_a = dist.Normal(m_sk_a_param, s_sk_a_param)
            pyro.sample('sk_a', dist_sk_a)

        dist_ic_w = dist.Normal(m_ic_w_param, s_ic_w_param)
        pyro.sample('i_w', dist_ic_w)
        dist_ic_a = dist.Normal(m_ic_a_param, s_ic_a_param)
        pyro.sample('i_a', dist_ic_a)
        dist_tc_w = dist.Normal(m_tc_w_param, s_tc_w_param)
        pyro.sample('t_w', dist_tc_w)
        dist_tc_a = dist.Normal(m_tc_a_param, s_tc_a_param)
        pyro.sample('t_a', dist_tc_a)
