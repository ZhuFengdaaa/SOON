import numpy as np
import scipy.sparse as sp
import torch
import math
from param import args
import networkx as nx
import time
# from torch.utils.data import Dataset, DataLoader
# from multiprocessing import Pool

class NavGraph():
    def __init__(self, env, batch_size, gnn_dim, gnn_order=1):
        self.V = None
        self.E = None
        self.env = env
        self.obs = None
        self.batch_size = batch_size
        self.gnn_dim = gnn_dim
        self.gnn_order = gnn_order
        self.time = {}
        # self.dataloader = DataLoader(self, num_workers = 16, shuffle=False, batch_size = batch_size)
        # self.pool = Pool(8)
    
    def reset(self):
        self.V = [{} for _ in range(self.batch_size)]
        self.E = [[] for _ in range(self.batch_size)]
        self.G = [nx.Graph() for _ in range(self.batch_size)]
        self.V_idx = [0 for _ in range(self.batch_size)]
        self.V_emb = [{} for _ in range(self.batch_size)] # embeded by attention
        self.C_feat = [{} for _ in range(self.batch_size)]
        self.C_emb = [{} for _ in range(self.batch_size)]
        self.feat_type = [{} for _ in range(self.batch_size)] # 1 visited, 2 candidate
        self.current_pos = ["" for _ in range(self.batch_size)]
    
    # def __len__(self):
    #     return self.batch_size
    
    # def __getitem__(self, idx):
    #     return self.construct_adj(idx)

    def viewpoint(self, i, hash):
        if hash not in self.V[i]:
            self.V[i][hash] = self.V_idx[i]
            self.G[i].add_node(self.V_idx[i])
            self.V_idx[i] += 1
        return self.V[i][hash]

    def update_vfeat(self, i, hash, feat):
        id = self.viewpoint(i,hash)
        if hash not in self.V_emb[i]:
            self.V_emb[i][id] = feat
            self.feat_type[i][id] = 1
        else:
            # visit this point multiple times
            import pdb; pdb.set_trace()
    
    def update_cfeat(self, i, hash, feat):
        id = self.viewpoint(i,hash)
        # if i == 0:
            # import pdb; pdb.set_trace()
        if id not in self.feat_type[i]:
            self.C_feat[i][id] = []
            self.feat_type[i][id] = 2
        elif self.feat_type[i][id] == 1:
            # visited
            return 
        self.C_feat[i][id].append(torch.from_numpy(feat))
        self.lazy_idx.append((i, id))
        self.lazy_feats.append(torch.mean(torch.stack(self.C_feat[i][id], dim=0), dim=0)) # mean for simplicity

    def add_edge(self, i, u, v):
        # u_id = self.V[i][u]
        # v_id = self.V[i][v]
        self.E[i].append((u, v))
        self.G[i].add_edge(u, v)
    
    def parallel_construct_adj(self):
        ob_views = [ob["viewpoint"] for ob in self.obs]
        res = self.pool.starmap(construct_adj, ((i, self.E, self.V, ob_views) for i in range(self.batch_size)))
        return res
    
    def construct_can(self, i):
        # candidates
        candidate_hash = []
        candidate_feat = []
        for hash in self.V[i].keys():
            # is candidate
            id = self.V[i][hash]
            if self.feat_type[i][id] == 2:
                candidate_hash.append((id, hash))
                candidate_feat.append(self.C_emb[i][id])
        return candidate_hash, candidate_feat

    def update_graph(self, obs, attn_feat, encoder):
        self.lazy_idx = []
        self.lazy_feats = []
        self.obs = obs
        for i, ob in enumerate(obs):
            cur_hash = ob["viewpoint"]
            self.current_pos[i] = cur_hash
            cur_id = self.viewpoint(i, cur_hash)
            self.update_vfeat(i, cur_hash, attn_feat[i])
            
            # update connectivity
            for candidate in ob["candidate"]:
                can_hash = candidate["viewpointId"]
                can_id = self.viewpoint(i, can_hash)
                self.add_edge(i, cur_id, can_id)
                self.update_cfeat(i, can_hash, candidate["feature"])
        
        # retrieve C_emb
        if len(self.lazy_feats) > 0:
            self.lazy_feats = encoder(torch.stack(self.lazy_feats, dim=0).cuda())
            for _i, (i, id) in enumerate(self.lazy_idx):
                self.C_emb[i][id] = self.lazy_feats[_i]
    
    def graph_candidate(self):
        max_can = 0
        candidate_hashs = []
        candidate_feats = []
        for i in range(self.batch_size):
            can_hashs, can_feats = self.construct_can(i)
            can_len = len(can_hashs)
            max_can = max(max_can, can_len)
            candidate_hashs.append(can_hashs)
            candidate_feats.append(can_feats)
        
        batch_candidates = []
        for candidates in candidate_feats:
            len_pad = max_can - len(candidates) + 1 # 1 for STOP action
            # try:
            #     c = candidates[0].shape[0] # 256
            # except:
            #     import pdb; pdb.set_trace()
            if len(candidates) > 0:
                candidates = torch.stack(candidates, dim=0).cuda()
                pad_tensor = torch.zeros(len_pad, args.gnn_dim).cuda()
                candidates = torch.cat([candidates, pad_tensor])
            else:
                candidates = torch.zeros(len_pad, args.gnn_dim).cuda()
            batch_candidates.append(candidates)
        batch_candidates = torch.stack(batch_candidates, dim=0)
        candidate_leng = [len(candidate)+1 for candidate in candidate_hashs]

        return batch_candidates, candidate_hashs, candidate_leng

    def construct_adj(self, i):
        edges = np.array(self.E[i])
        v_num = len(self.V[i])
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(v_num, v_num),
            dtype=np.float32)
        adj = torch.tensor(adj.todense()).cuda() + torch.eye(adj.shape[0]).cuda()
        # # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # # eliminate duplicate edges
        # adj.sum_duplicates()
        # adj.data[:] = 1
        # # normalize
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        cur_id = self.V[i][self.obs[i]["viewpoint"]]
        return adj, cur_id
        # return raw_adj.to_dense(), cur_id # for parallel

    def graph_obs(self):
        # stage_10_begin = time.time()
        # # res = next(self.dataloader)
        # res = next(iter(self.dataloader))
        # import ipdb; ipdb.set_trace()
        # stage_10_time = time.time() - stage_10_begin   
        # self.time_add("stage_10_time", stage_10_time)

        # stage_11_begin = time.time()
        # res = self.parallel_construct_adj()
        # batch_adj = [_res[1] for _res in res]
        # batch_cur_id = [_res[2] for _res in res]
        # _xs = [adj.shape[0] for adj in batch_adj]
        # max_x = max(_xs)
        # # retrieve adj, feat
        # stage_11_time = time.time() - stage_11_begin   
        # self.time_add("stage_11_time", stage_11_time)
       
        batch_adj = []
        batch_adj1 = []
        batch_cur_id = []
        max_x, max_y = 0, 0
        for i in range(self.batch_size):
            adj, cur_id = self.construct_adj(i)
            # adj = adj.cuda()
            _x, _y = adj.size()
            max_x = max(max_x, _x)
            max_y = max(max_y, _y)
            batch_adj.append(adj)
            batch_cur_id.append(cur_id)
        
        batch_cur_id = torch.tensor(batch_cur_id).cuda()
        assert max_x == max_y
        _shape = (max_x, max_y)
        for i, adj in enumerate(batch_adj):
            # _indices = adj._indices()
            # _values = adj._values()
            # adj = torch.sparse.FloatTensor(_indices, _values, _shape)
            # batch_adj[i] = adj
            _w, _h = adj.shape
            pad = torch.zeros(*_shape).cuda()
            pad[:_w, :_h] = adj
            batch_adj[i] = pad
            
        batch_adj = torch.stack(batch_adj, dim=0).cuda()
        # build symmetric adjacency matrix
        batch_adj_T = batch_adj.transpose(1,2)
        batch_adj1 = batch_adj + batch_adj_T
        batch_adj2 = (batch_adj1 > 0).float()
        # _values = batch_adj1.values()
        # _values = torch.ones(_values.shape).cuda()
        # _indices = batch_adj1.indices()
        # _shape = batch_adj1.shape
        # batch_adj2 = torch.sparse.FloatTensor(_indices, _values, _shape)
        # normalize
        # batch_adj2_sum = torch.sparse.sum(batch_adj2, 2)
        batch_adj2_sum = torch.sum(batch_adj2, 2)
        batch_adj2_norm = 1 / batch_adj2_sum
        batch_adj2_norm[batch_adj2_norm == float('inf')] = 0 
        # _indices = batch_adj2_sum.nonzero() # 2({B, E}) * |E|
        # _indices = torch.cat([_indices, _indices[:,1].unsqueeze(1)], dim=1)
        # _values = 1 / batch_adj2_sum.values()
        batch_adj2_diag = batch_diagonal(batch_adj2_norm)
        # batch_adj2_diag = torch.diag(batch_adj2_norm) # diag
        batch_adj2 = torch.bmm(batch_adj2, batch_adj2_diag)
        batch_adj2 = torch.transpose(batch_adj2, 1,2)
        batch_adj = batch_adj2

        featuress = []
        for i in range(self.batch_size):
            features = []
            for j in range(self.V_idx[i]):
                assert j in self.feat_type[i]
                if self.feat_type[i][j] == 1:
                    features.append(self.V_emb[i][j].unsqueeze(0))
                elif self.feat_type[i][j] == 2:
                    features.append(self.C_emb[i][j].unsqueeze(0))
                else:
                    print("ERROR: cannot find an embedding")
                    import pdb; pdb.set_trace()
            # for j in range(max_x - self.V_idx[i]): # padding
                # features[i].append(torch.zeros(self.gnn_dim).cuda())
            # for j in range(max_x - self.V_idx[i]): # padding
            features.append(torch.zeros(max_x - self.V_idx[i], self.gnn_dim).cuda())
            featuress.append(torch.cat(features, dim=0))
        batch_feat = torch.stack(featuress, dim=0)

        batch_candidates, candidate_hashs, candidate_leng = self.graph_candidate()
        # self.time_print()
        return batch_adj, batch_feat, batch_cur_id, batch_candidates, candidate_hashs, candidate_leng
    
    def teacher_action(self, obs, perm_idx, ended, traj=None, train_rl=None, debug={}):
        batch_candidates, candidate_hashs, candidate_leng = self.graph_candidate()
        a = np.zeros(len(obs), dtype=np.int64)
        deviations = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            item = self.env.batch[perm_idx[i]]
            deviation = 1000 # max int
            for node_hash in item['path']:
                dist = self._shortest_discrete_dist(item["scan"], obs[i]["viewpoint"], node_hash)
                deviation = min(deviation, dist)
            deviations[i] = deviation
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                if train_rl == False:
                    # supervised learning
                    teacher_hash = ob['teacher']
                    for k, candidate_hash in enumerate(candidate_hashs[i]):
                        if candidate_hash[1] == teacher_hash:   # Next view point
                            a[i] = k
                            break
                    else:   # Stop here
                        # print("Cannot find the teacher hash among candidates")
                        # import pdb; pdb.set_trace()
                        # assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                        if ob['teacher'] != ob['viewpoint']:
                            print("Cannot find the teacher hash and not stop")
                            import pdb; pdb.set_trace()
                            assert False
                        a[i] = len(candidate_hashs[i])
                else:
                    item = self.env.batch[perm_idx[i]]
                    is_set = False
                    if args.dataset == "SOON":
                        targets = [bbox["image_id"] for bbox in item["bboxes"]]
                        if obs[i]["viewpoint"] in targets:
                            a[i] = len(candidate_hashs[i]) # stop
                            is_set = True
                        else:
                            _min = float("inf")
                            for k, candidate_hash in enumerate(candidate_hashs[i]):        
                                dists = []
                                try:
                                    for target in targets:
                                        dist = self._shortest_dist(item["scan"], candidate_hash[1], target)
                                        dists.append(dist)
                                except:
                                    pass
                                if len(dists) == 0:
                                    continue
                                else:
                                    min_dist = min(dists)
                                    if min_dist < _min:
                                        _min = min_dist
                                        a[i] = k
                                        is_set = True
                        if is_set == False:
                            a[i] = len(candidate_hashs[i]) # stop
                    else:
                        if item['path'][-1] == obs[i]["viewpoint"]:
                            a[i] = len(candidate_hashs[i]) # stop
                            is_set = True
                        else:
                            _min = float("inf")
                            for k, candidate_hash in enumerate(candidate_hashs[i]):        
                                dist = self._shortest_dist(item["scan"], candidate_hash[1], item['path'][-1])
                                if dist < _min:
                                    _min = dist
                                    a[i] = k
                                    is_set = True
                        if is_set == False:
                            # import ipdb; ipdb.set_trace()
                            a[i] = len(candidate_hashs[i]) # stop

        return torch.from_numpy(a).cuda(), torch.from_numpy(deviations).cuda()

    def _shortest_dist(self, scanId, curViewpointId, goalViewpointId):
        return self.env.dists[scanId][curViewpointId][goalViewpointId]
    
    def _shortest_discrete_dist(self, scanId, curViewpointId, goalViewpointId):
        return self.env.discrete_dists[scanId][curViewpointId][goalViewpointId]
    
    def _shortest_path(self, scanId, curViewpointId, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if curViewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.env.paths[scanId][curViewpointId][goalViewpointId]
        return path     

    def make_graph_action(self, a_t, perm_obs, candidate_hashs, perm_idx=None, traj=None, _target=None):
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            # STOP 
            if action != -1:
                select_hash = candidate_hashs[i][action][1]
                src_point = self.V[i][perm_obs[i]['viewpoint']]
                trg_point = self.V[i][select_hash]
                prev_view = perm_obs[i]["viewpoint"]
                shortest_path = nx.shortest_path(self.G[i], source=src_point, target=trg_point)
                navigate_nodes = shortest_path[1:]
                for node in navigate_nodes:
                    # for candidate in candidate_hashs[i]:
                    is_navigate = False
                    for v_hash in self.V[i]:
                        v_idx = self.V[i][v_hash]
                        # candidate: (id, hash)
                        if node == v_idx:
                            target_hash = v_hash
                            perm_obs = self.env._update_obs(perm_obs, perm_idx[i], i)
                            # chack candidate avaliable
                            s_point, t_point, select_candidate = None, None, None
                            s_point = perm_obs[i]['viewIndex']
                            for target in perm_obs[i]['candidate']:
                                if target["viewpointId"] == target_hash:
                                    t_point = target['pointId']
                                    select_candidate = target
                            if t_point is None:
                                print("t_point not found")
                                import pdb; pdb.set_trace()
                            self.make_pano_action(s_point, t_point, select_candidate, idx) # idx: index of the env before perm
                            state = self.env.env.sims[idx].getState()
                            if traj is not None:
                                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                            # update perm_obs
                            is_navigate = True
                            break
                    if is_navigate == False:
                        print("navigate fail")
                        import pdb; pdb.set_trace()       
    
    def make_pano_action(self, src_point, trg_point, select_candidate, idx):
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
        """
        Interface between graph view and Egocentric view 
        It will convert the action graph view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(idx, name):
            # print(idx, name)
            # print("before", self.env.env.sims[idx].getState().location.viewpointId)
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*env_actions[name])
            # print("after", self.env.env.sims[idx].getState().location.viewpointId)
        # assert trg_point is not None, "trg_point not found"
        src_level = (src_point ) // 12   # The point idx started from 0
        trg_level = (trg_point ) // 12
        while src_level < trg_level:    # Tune up
            take_action(idx, 'up')
            src_level += 1
        while src_level > trg_level:    # Tune down
            take_action(idx, 'down')
            src_level -= 1
        while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
            take_action(idx, 'right')
        assert select_candidate['viewpointId'] == \
                self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
        take_action(idx, select_candidate['idx'])
    
    def time_add(self, name, value):
        if name not in self.time:
            self.time[name] = 0
        self.time[name] += value
    
    def time_print(self):
        _str = ""
        for k, v in self.time.items():
            _str += f"{k}: {v:.2f}, "
        print(_str)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_adj(i, E, V, ob_views):
    edges = np.array(E[i])
    v_num = len(V[i])
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(v_num, v_num),
        dtype=np.float32)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    cur_id = V[i][ob_views[i]]
    return i, adj.to_dense(), cur_id

def batch_diagonal(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N) 
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims).cuda()
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal 
    output.as_strided(input.size(), strides ).copy_(input)
    return output   