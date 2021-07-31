
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer, get_sync_dir
import utils
import model
import param
from param import args
from collections import defaultdict
from nav_graph import NavGraph


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
        self.nav_graph = NavGraph(env, args.batchSize, args.gnn_dim, args.gnn_order)
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.nav_graph.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                if args.dataset == 'R2R':
                    for traj in self.rollout(**kwargs):
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                elif args.dataset == 'SOON':
                    trajs, pre_heading, pre_elevation = self.rollout(**kwargs)
                    for index in range(len(trajs)):
                        self.loss = 0
                        self.results[trajs[index]['instr_id']] = {}
                        self.results[trajs[index]['instr_id']]['path'] = trajs[index]['path']
                        self.results[trajs[index]['instr_id']]['obj_heading'] = pre_heading[index].tolist()
                        self.results[trajs[index]['instr_id']]['obj_elevation'] = pre_elevation[index].tolist()
        else:   # Do a full round
            while True:
                if args.dataset == 'R2R':
                    for traj in self.rollout(**kwargs):
                        if traj['instr_id'] in self.results:
                            looped = True
                        else:
                            self.loss = 0
                            self.results[traj['instr_id']] = traj['path']
                    if looped:
                        break
                elif args.dataset == 'SOON':
                    if not args.compute_bbox:
                        trajs, pre_heading, pre_elevation = self.rollout(**kwargs)
                        for index in range(len(trajs)):
                            if trajs[index]['instr_id'] in self.results:
                                looped = True
                            else:
                                self.loss = 0
                                self.results[trajs[index]['instr_id']] = {}
                                self.results[trajs[index]['instr_id']]['path'] = trajs[index]['path']
                                self.results[trajs[index]['instr_id']]['obj_heading'] = pre_heading[index].tolist()
                                self.results[trajs[index]['instr_id']]['obj_elevation'] = pre_elevation[index].tolist()
                    else:
                        trajs, pre_bboxes, pre_num_headings, pre_num_elevations = self.rollout(**kwargs)
                        for index in range(len(trajs)):
                            if trajs[index]['instr_id'] in self.results:
                                looped = True
                            else:
                                self.loss = 0
                                self.results[trajs[index]['instr_id']] = {}
                                self.results[trajs[index]['instr_id']]['path'] = trajs[index]['path']
                                self.results[trajs[index]['instr_id']]['pre_bbox'] = pre_bboxes[index]
                                self.results[trajs[index]['instr_id']]['pre_num_heading'] = pre_num_headings[index].argmax()
                                self.results[trajs[index]['instr_id']]['pre_num_elevation'] = pre_num_elevations[index].argmax()
                    if looped:
                        break
                

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20, dataset='R2R'):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = env.feature_size
        self.time = {}

        # Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        self.encoder = model.EncoderLSTM(tok.vocab_size(), args.wemb, enc_hidden_size, padding_idx,
                                         args.dropout, bidirectional=args.bidir).cuda()
        hidden_size = args.rnn_dim
        gnn_size = args.gnn_dim
        feature_size = self.feature_size + args.angle_feat_size
        self.vis_encoder = model.VisEncoder(hidden_size, feature_size, gnn_size, args.dropout).cuda()
        self.can_encoder = model.CanEncoder(gnn_size, feature_size, args.dropout).cuda()
        self.graph_encoder = model.GraphConvolution(args.gnn_dim, args.gnn_dim).cuda()
        # self.decoder = model.AttnDecoderLSTM(args.aemb, args.rnn_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.decoder = model.GraphDecoderLSTM(args.aemb, args.gnn_dim, args.rnn_dim, args.dropout).cuda()
        self.critic = model.Critic().cuda()

        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        graph_encoder_model = list(self.vis_encoder.parameters()) + list(self.can_encoder.parameters()) + list(self.graph_encoder.parameters())
        self.graph_encoder_optimizer = args.optimizer(graph_encoder_model, lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)

        # aggregation
        self.models = [self.encoder, self.vis_encoder, self.can_encoder, self.graph_encoder, self.decoder, self.critic]
        self.all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("vis_encoder", self.vis_encoder, self.graph_encoder_optimizer),
                     ("can_encoder", self.can_encoder, self.graph_encoder_optimizer),
                     ("graph_encoder", self.graph_encoder, self.graph_encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        self.optimizers = [self.encoder_optimizer, self.graph_encoder_optimizer, self.decoder_optimizer, self.critic_optimizer]

        self.dataset = args.dataset
        if self.dataset == 'SOON' and not args.compute_bbox:
            self.heading_indicator = model.OrientationIndicator().cuda()
            self.elevation_indicator = model.OrientationIndicator().cuda()
            self.bce_loss = nn.BCELoss()
            self.orientation_optimizer = args.optimizer(
                list(self.heading_indicator.parameters())
                + list(self.elevation_indicator.parameters()), lr=args.lr)
            
            self.models += [self.heading_indicator, self.elevation_indicator]
            self.all_tuple += [("heading_indicator", self.heading_indicator, self.orientation_optimizer), 
            ("elevation_indicator", self.elevation_indicator, self.orientation_optimizer)]
            self.optimizers += [self.orientation_optimizer]

        if self.dataset == 'SOON' and args.compute_bbox:
            self.bbox_indicator = model.BboxIndicator().cuda()
            self.num_heading_indicator = model.HeadingNumIndicator().cuda()
            self.num_elevation_indicator = model.ElevationNumIndicator().cuda()
            self.mse_loss = nn.MSELoss()

            self.bbox_optimizer = args.optimizer(
                list(self.bbox_indicator.parameters()) + list(self.num_heading_indicator.parameters()) + \
                list(self.num_elevation_indicator.parameters()), lr=args.lr)
            
            self.models += [self.bbox_indicator, self.num_heading_indicator, self.num_elevation_indicator]
            self.all_tuple += [("bbox_indicator", self.bbox_indicator, self.bbox_optimizer), 
            ("num_heading_indicator", self.num_heading_indicator, self.bbox_optimizer), 
            ("num_elevation_indicator", self.num_elevation_indicator, self.bbox_optimizer)]
            self.optimizers += [self.bbox_optimizer]

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

        if self.dataset == 'SOON':
            if not args.compute_bbox:
                self.all_tuple.append(("heading_predictor", self.heading_indicator, self.orientation_optimizer))
                self.all_tuple.append(("elevation_predictor", self.elevation_indicator, self.orientation_optimizer))
            else:
                self.all_tuple.append(("bbox_predictor", self.bbox_indicator, self.bbox_optimizer))
                self.all_tuple.append(("num_heading_indicator", self.num_heading_indicator, self.bbox_optimizer))
                self.all_tuple.append(("num_elevation_indicator", self.num_elevation_indicator, self.bbox_optimizer))

    def set_env(self, env):
        self.nav_graph.env = env

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    # def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
    #     """
    #     Interface between Panoramic view and Egocentric view 
    #     It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
    #     """
    #     def take_action(i, idx, name):
    #         if type(name) is int:       # Go to the next view
    #             self.nav_graph.env.env.sims[idx].makeAction(name, 0, 0)
    #         else:                       # Adjust
    #             self.nav_graph.env.env.sims[idx].makeAction(*self.nav_graph.env_actions[name])
    #         state = self.nav_graph.env.env.sims[idx].getState()
    #         if traj is not None:
    #             traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
    #     if perm_idx is None:
    #         perm_idx = range(len(perm_obs))
    #     for i, idx in enumerate(perm_idx):
    #         action = a_t[i]
    #         if action != -1:            # -1 is the <stop> action
    #             select_candidate = perm_obs[i]['candidate'][action]
    #             src_point = perm_obs[i]['viewIndex']
    #             trg_point = select_candidate['pointId']
    #             src_level = (src_point ) // 12   # The point idx started from 0
    #             trg_level = (trg_point ) // 12
    #             while src_level < trg_level:    # Tune up
    #                 take_action(i, idx, 'up')
    #                 src_level += 1
    #             while src_level > trg_level:    # Tune down
    #                 take_action(i, idx, 'down')
    #                 src_level -= 1
    #             while self.nav_graph.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
    #                 take_action(i, idx, 'right')
    #             assert select_candidate['viewpointId'] == \
    #                    self.nav_graph.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
    #             take_action(i, idx, select_candidate['idx'])

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False
        if reset:
            # Reset env
            obs = np.array(self.nav_graph.env.reset())
        else:
            obs = np.array(self.nav_graph.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.nav_graph.env.batch.copy()
            speaker.env = self.nav_graph.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.nav_graph.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx_mask = seq_mask

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        rl_ml_loss = 0.

        prev_h = h_t
        # initialize graph
        self.nav_graph.reset()
        last_feat = torch.zeros(batch_size, args.rnn_dim).cuda()
        for t in range(self.episode_len):
            input_a_t, f_t, _, _ = self.get_input_feat(perm_obs)
            if speaker is not None:       # Apply the env drop mask to the feat
                # candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
            
            vis_encoder_begin = time.time()
            attn_feat = self.vis_encoder(f_t, ctx, already_dropfeat=(speaker is not None))
            vis_encoder_time = time.time() - vis_encoder_begin
            self.time_add("vis_encoder_time", vis_encoder_time)
            # viewpoints = [ob["viewpoint"] for ob in perm_obs]
            # print(viewpoints)
            update_graph_begin = time.time()
            self.nav_graph.update_graph(perm_obs, attn_feat, self.can_encoder)
            update_graph_time = time.time() - update_graph_begin
            self.time_add("update_graph_time", update_graph_time)

            graph_obs_begin = time.time()
            batch_adj, batch_feat, batch_cur_id, batch_candidates, candidate_hashs, candidate_leng = self.nav_graph.graph_obs()
            graph_obs_time = time.time() - graph_obs_begin
            self.time_add("graph_obs_time", graph_obs_time)

            graph_encoder_begin = time.time()
            batch_feat = self.graph_encoder(batch_adj, batch_feat)
            graph_encoder_time = time.time() - graph_encoder_begin
            self.time_add("graph_encoder_time", graph_encoder_time)

            decoder_begin = time.time()
            batch_cur_id = batch_cur_id.unsqueeze(1).unsqueeze(2)
            indices = batch_cur_id.expand(batch_cur_id.size(0), 1, batch_feat.size(2))
            batch_cur_feat = batch_feat.gather(1, indices).squeeze() # find the gnn feature of the current position
            h_t, c_t, logit, prev_h = self.decoder(input_a_t, batch_cur_feat, batch_candidates, prev_h, h_t, c_t, ctx, ctx_mask)
            decoder_time = time.time() - decoder_begin
            self.time_add("decoder_time", decoder_time)
            
            # self.graph_encoder 
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))

            # Supervised training
            # target = self._teacher_action(perm_obs, ended)
            teacher_action_begin = time.time()
            if self.feedback == 'teacher' or self.feedback == "sample":
                try:
                    target, deviations = self.nav_graph.teacher_action(perm_obs, perm_idx, ended, traj, train_rl=train_rl, debug={self.feedback})
                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()
                ml_loss += self.criterion(logit, target)
            else:
                target = None
            teacher_action_time = time.time() - teacher_action_begin
            self.time_add("teacher_action_time", teacher_action_time)
        
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                # sample_a_t = c.sample().detach()
                _mask = (deviations > args.rl_ml_th).int()
                target = target * (1-_mask) + args.ignoreid * _mask
                policy_log_probs.append(c.log_prob(a_t))
                
                # calculate rl_ml loss
                # import ipdb; ipdb.set_trace()
                rl_ml_loss += self.criterion(logit, target)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            decoder_begin = time.time()
            # Make action and get the new state
            # self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            self.nav_graph.make_graph_action(cpu_a_t, perm_obs, candidate_hashs, perm_idx, traj, target)
            obs = np.array(self.nav_graph.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu
            decoder_time = time.time() - decoder_begin
            self.time_add("decoder_time", decoder_time)

            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            raise NameError("The action doesn't change the move")
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # update the last not ended prev_h
            ended_tensor = torch.from_numpy(ended).cuda().unsqueeze(1).float()
            last_feat = (1-ended_tensor).float() * prev_h + ended_tensor * last_feat

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if speaker is not None:
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
            # last_h_, _, _, _ = self.decoder(input_a_t, f_t, candidate_feat,
            #                                 h_t, h1, c_t,
            #                                 ctx, ctx_mask,
            #                                 speaker is not None)
            last_h_, _, _, _ = self.decoder(input_a_t, batch_cur_feat, batch_candidates, prev_h, h_t, c_t, ctx, ctx_mask)
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss * args.rl_weight
        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
        
        if train_rl is not None and args.rl_ml_weight > 0:
            self.loss += rl_ml_loss * args.rl_ml_weight / batch_size
        
        if self.dataset == 'SOON':
            
            if not args.compute_bbox:
                pre_heading = self.heading_indicator(last_feat)
                pre_elevation = self.elevation_indicator(last_feat)
                if args.det_loss and not train_rl:
                    inds = []
                    gt_headings = []
                    gt_elevations = []
                    for index in range(len(traj)):
                        pre_traj = traj[index]
                        pre_last_view = pre_traj['path'][-1][0]
                        gt_image_ids = perm_obs[index]['image_ids']
                        if pre_last_view in gt_image_ids:
                            ins_ind = gt_image_ids.index(pre_last_view)
                            inds.append(index)
                            gt_headings.append(perm_obs[index]['obj_headings'][ins_ind])
                            gt_elevations.append(perm_obs[index]['obj_elevations'][ins_ind])
                    if len(inds) > 0:
                        # t1 = pre_heading[inds]
                        # t2 = torch.Tensor(gt_headings).cuda()
                        # print(t1[0].item(), t2[0].item())
                        # import ipdb; ipdb.set_trace()
                        heading_loss = self.bce_loss(pre_heading[inds], torch.Tensor(gt_headings).cuda())
                        elevation_loss = self.bce_loss(pre_elevation[inds], torch.Tensor(gt_elevations).cuda())
                        self.logs['heading_loss'].append(heading_loss.detach())
                        self.logs['elevation_loss'].append(elevation_loss.detach())
                        # print(f'heading_loss {heading_loss.detach()}, elevation_loss {elevation_loss.detach()}')
                        self.loss += heading_loss
                        self.loss += elevation_loss
            else:
                offset_headings = []
                offset_elevations = []
                # vl_feat = h1
                for index in range(len(traj)):
                    num = len(perm_obs[index]['offset_headings'])
                    ind = random.randint(0, num - 1)
                    offset_headings.append(perm_obs[index]['offset_headings'][ind])
                    offset_elevations.append(perm_obs[index]['offset_elevations'][ind])

                offset_headings_tensor = torch.FloatTensor(offset_headings).cuda()
                offset_elevations_tensor = torch.FloatTensor(offset_elevations).cuda()
                vl_feat = torch.cat([vl_feat, offset_headings_tensor.unsqueeze(dim=1),
                                     offset_elevations_tensor.unsqueeze(dim=1)], dim=1)
                pred_bboxes_2d = self.bbox_indicator(vl_feat)
                pred_num_headings = self.num_heading_indicator(vl_feat)
                pred_num_elevations = self.num_elevation_indicator(vl_feat)

                if not train_rl:
                    inds = []
                    gt_bboxes_2d = []
                    gt_num_headings = []
                    gt_num_elevations = []
                    for index in range(len(traj)):
                        pre_traj = traj[index]
                        pre_last_view = pre_traj['path'][-1][0]
                        gt_image_ids = perm_obs[index]['image_ids']
                        if pre_last_view in gt_image_ids:
                            ins_ind = gt_image_ids.index(pre_last_view)
                            inds.append(index)
                            gt_bboxes_2d.append(perm_obs[index]['bboxes_2d'][ins_ind])
                            gt_num_headings.append(perm_obs[index]['num_headings'][ins_ind])
                            gt_num_elevations.append(perm_obs[index]['num_elevations'][ins_ind])

                    if len(inds) > 0:
                        gt_bboxes_2d_tensor = torch.FloatTensor(gt_bboxes_2d).cuda()
                        gt_num_headings = np.array(gt_num_headings).astype(np.int)
                        gt_num_elevations = np.array(gt_num_elevations).astype(np.int)
                        bbox_loss = self.mse_loss(pred_bboxes_2d[inds], gt_bboxes_2d_tensor)
                        num_heading_loss = F.cross_entropy(pred_num_headings[inds], torch.from_numpy(gt_num_headings).cuda())
                        num_elevation_loss = F.cross_entropy(pred_num_elevations[inds], torch.from_numpy(gt_num_elevations).cuda())
                        self.logs['bbox_loss'].append(bbox_loss.detach())
                        self.logs['num_heading_loss'].append(num_heading_loss.detach())
                        self.logs['num_elevation_loss'].append(num_elevation_loss.detach())
                        self.loss += bbox_loss
                        self.loss += num_heading_loss
                        self.loss += num_elevation_loss
        
        if type(self.loss) is int or type(self.loss) is float:  # For safety, it will be activated if no losses are added
            assert not (self.loss > 0)
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        # self.time_print()
        # return traj
        if self.dataset == 'SOON':
            if not args.compute_bbox:
                return traj, pre_heading.detach().cpu().numpy(), pre_elevation.detach().cpu().numpy()
            else:
                return traj, pred_bboxes_2d.detach().cpu().numpy(), pred_num_headings.detach().cpu().numpy(), pred_num_elevations.detach().cpu().numpy()
        else:
            return traj

    def time_add(self, name, value):
        if name not in self.time:
            self.time[name] = 0
        self.time[name] += value
    
    def time_print(self):
        _str = ""
        for k, v in self.time.items():
            _str += f"{k}: {v:.2f}, "
        print(_str)

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.nav_graph.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        # Dijk Algorithm
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.nav_graph.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.nav_graph.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.nav_graph.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                                      h_t, h1, c_t,
                                                      ctx, ctx_mask,
                                                      False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item() 
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.nav_graph.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            for model in self.models:
                model.train()
        else:
            for model in self.models:
                model.eval()
        super(Seq2SeqAgent, self).test(iters)

    def do_train(self):
        for model in self.models:
            model.train()
    
    def do_eval(self):
        for model in self.models:
            model.eval()

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        self.do_train()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        for optimizer in self.optimizers:
            optimizer.step()
        # self.encoder_optimizer.step()
        # self.graph_encoder_optimizer.step()
        # self.decoder_optimizer.step()
        # self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.do_train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            self.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.optim_step()

            # self.loss.backward()

            # torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            # torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

            # self.encoder_optimizer.step()
            # self.graph_encoder_optimizer.step()
            # self.decoder_optimizer.step()
            # self.critic_optimizer.step()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        
        for param in self.all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        for param in self.all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

