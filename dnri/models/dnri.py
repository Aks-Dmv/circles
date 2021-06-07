import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.normal import Normal
from copy import deepcopy
import numpy as np
from . import model_utils
from .model_utils import RefNRIMLP, encode_onehot
import math


class DNRI(nn.Module):
    def __init__(self, params):
        super(DNRI, self).__init__()
        # Model Params
        self.num_vars = params['num_vars']
        self.encoder = DNRI_Encoder(params)
        decoder_type = params.get('decoder_type', None)
        if decoder_type == 'ref_mlp':
            self.decoder = DNRI_MLP_Decoder(params)
        else:
            self.decoder = DNRI_Decoder(params)
        self.decoder_targ = deepcopy(self.decoder)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.decoder_targ.parameters():
            p.requires_grad = False

        self.discrim = MLP_Discriminator(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        
        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.add_uniform_prior = True #params.get('add_uniform_prior')
        if self.add_uniform_prior:
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(self.num_edge_types)
                prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
                prior[0] = params['no_edge_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(self.num_edge_types)
                prior.fill(1.0/self.num_edge_types)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior

    def single_step_Critic(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden, hidden_pred = self.decoder(inputs, decoder_hidden, edges)

        q1_critic, q2_critic = self.decoder.q_net(hidden_pred.detach(), predictions.detach())

        q1_target, q2_target = self.decoder_targ.q_net(hidden_pred.detach(), predictions.detach())
        q_pi_targ = torch.min(q1_target, q2_target)
        
        return predictions, decoder_hidden, q1_critic, q2_critic, q_pi_targ, edges
    
    def single_step_Actor(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden, hidden_pred = self.decoder(inputs, decoder_hidden, edges)
        
        q1_policy, q2_policy = self.decoder.q_net(hidden_pred.detach(), predictions)
        q_pi_critic = torch.min(q1_policy, q2_policy)
        
        return predictions, decoder_hidden, q_pi_critic, edges
    
    def single_step_forward(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden, _ = self.decoder(inputs, decoder_hidden, edges)
        return predictions, decoder_hidden, edges
    
    
    def calculate_loss_discrim(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []

        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, _ = self.encoder(inputs)
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions

            current_p_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)

        target = inputs[:, 1:, :, :]

        loss_fn = nn.BCEWithLogitsLoss()
        # gen_states = self.discrim(all_predictions).flatten()
        # print("fake ",torch.sigmoid(gen_states).mean().item())
        # real_states = self.discrim(target).flatten()
        # print("real ",torch.sigmoid(real_states).mean().item())
        # loss_discrim = loss_fn(gen_states, torch.zeros_like(gen_states)) + loss_fn(real_states, torch.ones_like(real_states))

        gen_states = self.discrim(all_predictions)
        real_states = self.discrim(target)

        # gradient penalty
        alpha = torch.rand(target.shape[0], 1, 1, 1).cuda()
        interpolates = ( alpha * target.detach() + (1 - alpha) * all_predictions.detach() ).requires_grad_(True)
        d_interpolates = self.discrim(interpolates)
        gp_grad = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        gp_grad = torch.reshape(gp_grad, (gp_grad.shape[0], -1))
        gp_grad_norm = gp_grad.norm(2, dim=1)
        gradient_penalty = torch.mean((gp_grad_norm - 1) ** 2)

        # loss_discrim = (0.55-torch.mean(real_states))**2 + (0.45-torch.mean(gen_states))**2 + 30 * gradient_penalty
        loss_discrim = loss_fn(gen_states, torch.zeros_like(gen_states)) + loss_fn(real_states, torch.ones_like(real_states)) + 0.01 * gradient_penalty
        print("fake ", torch.mean(torch.sigmoid(gen_states)).item(), " / real ", torch.mean(torch.sigmoid(real_states)).item() )

        return loss_discrim
    

    def calculate_loss_q(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []

        all_q1_c = []
        all_q2_c = []
        all_q_target = []

        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, _ = self.encoder(inputs)
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        
        # we change the number of steps from (num_time_steps-1), to num_time_steps
        # as we need to get the q_target values of the next states
        for step in range(num_time_steps):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions

            current_p_logits = prior_logits[:, step]
            predictions, decoder_hidden, q1_critic, q2_critic, q_pi_targ, edges = self.single_step_Critic(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_q1_c.append(q1_critic)
            all_q2_c.append(q2_critic)
            all_q_target.append(q_pi_targ)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        all_q1_c = torch.stack(all_q1_c, dim=1)
        all_q2_c = torch.stack(all_q2_c, dim=1)
        all_q_target = torch.stack(all_q_target, dim=1)

        target = inputs[:, 1:, :, :]
        # removed the last all_predictions as the last for looping was essentially to calculate q_target and log_pi
        loss_nll = self.nll(all_predictions[:, :-1], target) # old version
        # reward_discrim = 10 * self.discrim(all_predictions[:, :-1]).detach() # wgan reward
        reward_discrim = -torch.log(1-torch.sigmoid(self.discrim(all_predictions[:, :-1]))+1e-5) # reward = log(D); D = rho_E/(rho_E + rho_pi)
        # for i in range(loss_nll.shape[1]):
        #     print(all_predictions[0, i,0].cpu().detach().numpy(), target[0,i,0].cpu().detach().numpy())

        # critic loss
        print("reward ",reward_discrim.tolist()[0][30][0])
        gamma = 0.25
        rewards_to_go = reward_discrim + gamma * (all_q_target[:, 1:].mean(dim=-1))
        rewards_to_go[:, -1] = reward_discrim[:, -1] # assuming finite-horizon MDP
        loss_critic = ((all_q1_c[:, :-1].mean(dim=-1) - rewards_to_go.detach())**2).mean() + ((all_q2_c[:, :-1].mean(dim=-1) - rewards_to_go.detach())**2).mean()

        return loss_critic, loss_nll.mean(dim=-1).mean(dim=-1)
    
    def calculate_loss_pi(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_q_pi = []
        all_q_g = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, _ = self.encoder(inputs)
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions

            current_p_logits = prior_logits[:, step]
            predictions, decoder_hidden, q_pi_critic, edges = self.single_step_Actor(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_q_pi.append(q_pi_critic)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        all_q_pi = torch.stack(all_q_pi, dim=1)

        # actor loss
        # for i in range(0, inputs.shape[1]-1, 7):
        #     print(all_predictions[0,i].mean().cpu().item(), inputs[0,i].mean().cpu().item())
        loss_policy = - all_q_pi.mean(dim=-1)

        prob_pr = F.softmax(prior_logits, dim=-1)
        if self.add_uniform_prior:
            loss_kl = self.kl_categorical_avg(prob_pr)

        loss = loss_policy.mean() + loss_kl.mean()

        if return_edges:
            return loss, loss_policy, loss_kl, edges
        elif return_logits:
            return loss, loss_policy, loss_kl, prior_logits, all_predictions
        else:
            return loss, loss_policy, loss_kl


    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        all_predictions = []
        all_edges = []
        prior_logits, prior_hidden = self.encoder(inputs[:, :-1])
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
            if return_everything:
                all_edges.append(edges)
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            # predictions = predictions + torch.normal(0., 0.05, size=(predictions.shape[0],predictions.shape[1], predictions.shape[2])).cuda()
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
            predictions, decoder_hidden, edges = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
            all_predictions.append(predictions)
            all_edges.append(edges)
        
        predictions = torch.stack(all_predictions, dim=1)
        if return_edges:
            edges = torch.stack(all_edges, dim=1)
            return predictions, edges
        else:
            return predictions

    def copy_states(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            current_state = (state[0].clone(), state[1].clone())
        else:
            current_state = state.clone()
        return current_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        print("INPUT SHAPE: ",inputs.shape)
        prior_logits, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        all_timestep_edges = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            current_batch_edges = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior = self.encoder.copy_states(prior_hidden)
                tmp_decoder = self.copy_states(decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
                if return_edges:
                    current_batch_edges.append(current_edge_logits.cpu())
            batch_prior_hidden = self.encoder.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            if return_edges:
                current_batch_edges = torch.cat(current_batch_edges, 0)
                current_timestep_edges = [current_batch_edges]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
                if return_edges:
                    current_timestep_edges.append(current_batch_edge_logits.cpu())
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
            if return_edges:
                all_timestep_edges.append(torch.stack(current_timestep_edges, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            edge_result = torch.cat(all_timestep_edges, dim=0)
            return result.unsqueeze(0), edge_result.unsqueeze(0)
        else:
            return result.unsqueeze(0)

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) )#.view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))


    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1)#.mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class DNRI_Encoder(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(DNRI_Encoder, self).__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)


        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?


    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result



    def forward(self, inputs):
        if self.training or not self.save_eval_memory:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            x = inputs.transpose(2, 1).contiguous()
            # New shape: [num_sims, num_atoms, num_timesteps, num_dims]
            x = self.mlp1(x)  # 2-layer ELU net per node
            x = self.node2edge(x)
            x = self.mlp2(x)
            x_skip = x
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        
            
            # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
            # RNN aggregation
            old_shape = x.shape
            x = x.contiguous().view(-1, old_shape[2], old_shape[3])
            forward_x, prior_state = self.forward_rnn(x)
            timesteps = old_shape[2]
            
            #x: [batch*num_edges, num_timesteps, hidden_size]
            prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            return prior_result, prior_state
        else:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            all_x = []
            all_forward_x = []
            all_prior_result = []
            prior_state = None
            for timestep in range(num_timesteps):
                x = inputs[:, timestep]
                #x = inputs.transpose(2, 1).contiguous()
                x = self.mlp1(x)  # 2-layer ELU net per node
                x = self.node2edge(x)
                x = self.mlp2(x)
                x_skip = x
                x = self.edge2node(x)
                x = self.mlp3(x)
                x = self.node2edge(x)
                x = torch.cat((x, x_skip), dim=-1)  # Skip connection
                x = self.mlp4(x)
            
                
                # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
                # RNN aggregation
                old_shape = x.shape
                x = x.contiguous().view(-1, 1, old_shape[-1])
                forward_x, prior_state = self.forward_rnn(x, prior_state)
                all_x.append(x.cpu())
                all_forward_x.append(forward_x.cpu())
                all_prior_result.append(self.prior_fc_out(forward_x).view(old_shape[0], 1, old_shape[1], self.num_edges).cpu())
            prior_result = torch.cat(all_prior_result, dim=1).cuda(non_blocking=True)
            return prior_result, prior_state

    def single_step_forward(self, inputs, prior_state):
        # Inputs is shape [batch, num_vars, input_size]
        x = self.mlp1(inputs)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape
        x  = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                       prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (prior_state[0].view(old_prior_shape), prior_state[1].view(old_prior_shape))
        return prior_result, prior_state

class MLP_Discriminator(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(MLP_Discriminator, self).__init__()
        num_vars = params['num_vars']
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)

        self.discrim_fc_out1 = nn.Linear(hidden_size, hidden_size)
        self.discrim_fc_out2 = nn.Linear(hidden_size, 1)

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?

    def forward(self, inputs):
        # Inputs is shape [batch, num_timesteps, num_vars, input_size]
        num_timesteps = inputs.size(1)
        x = inputs.transpose(2, 1).contiguous()
        # New shape: [batch, num_vars, num_timesteps, num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.discrim_fc_out1(F.relu(x))
        x = self.discrim_fc_out2(F.relu(x)).transpose(2, 1).contiguous().mean(dim=-1)
        # At this point, x should be [batch, num_timesteps, num_vars]

        return x


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.q1_layer1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.q2_layer1 = nn.Linear(obs_dim + act_dim, hidden_size)

        self.q1_layer2 = nn.Linear(hidden_size, hidden_size)
        self.q2_layer2 = nn.Linear(hidden_size, hidden_size)
        self.q1_layer3 = nn.Linear(hidden_size, hidden_size)
        self.q2_layer3 = nn.Linear(hidden_size, hidden_size)

        self.q1_layer4 = nn.Linear(hidden_size, 1)
        self.q2_layer4 = nn.Linear(hidden_size, 1)

    def forward(self, q_input, pi_action):
        q1_l1 = self.q1_layer1( torch.cat([q_input, pi_action], dim=-1) )
        q2_l1 = self.q2_layer1( torch.cat([q_input, pi_action], dim=-1) )
        q1_l2 = self.q1_layer2( F.relu(q1_l1) )
        q2_l2 = self.q2_layer2( F.relu(q2_l1) )
        q1_l3 = self.q1_layer3( F.relu(q1_l2) )
        q2_l3 = self.q2_layer3( F.relu(q2_l2) )
        q1 = self.q1_layer4( F.relu(q1_l3) )
        q2 = self.q2_layer4( F.relu(q2_l3) )
        return q1, q2
    
class DNRI_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_Decoder, self).__init__()
        self.num_vars = num_vars =  params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        
        self.mu_layer = nn.Linear(n_hid, out_size)
        self.log_std_layer = nn.Linear(n_hid, out_size)

        self.q_net = MLPQFunction(n_hid, out_size, n_hid)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        
        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape, device=inputs.device)
        
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i+1]
            all_msgs += msg/norm

        # This step sums all of the messages per node
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1) # Average

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)

        mu = self.mu_layer(pred)
        log_std = self.log_std_layer(pred)
        log_std = torch.clamp(log_std - 3.5, -4, -3)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if self.training:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        return pi_action, hidden, pred.clone()


class DNRI_MLP_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_MLP_Decoder, self).__init__()
        num_vars = params['num_vars']
        edge_types = params['num_edge_types']
        n_hid = params['decoder_hidden']
        msg_hid = int(params['decoder_hidden']/8)
        msg_out = msg_hid #TODO: make this a param
        skip_first = params['skip_first']
        n_in_node = params['input_size']

        do_prob = params['decoder_dropout']
        in_size = n_in_node
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_size, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        out_size = n_in_node
        self.out_fc1 = nn.Linear(in_size + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        
        self.mu_layer = nn.Linear(n_hid, out_size)
        self.log_std_layer = nn.Linear(n_hid, out_size)

        self.q_net = MLPQFunction(n_hid, out_size, n_hid)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return None

    def forward(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
        else:
            p = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        
        mu = self.mu_layer(pred)
        log_std = self.log_std_layer(pred)
        log_std = torch.clamp(log_std - 3.5, -4, -3)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if self.training:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        # Predict position/velocity difference
        return pi_action, None, pred.clone()
