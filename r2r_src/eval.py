''' Evaluation of agent trajectories '''

import copy
import json
import os
import sys
import math
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# from matplotlib import path as mlpath

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent
from param import args


DATASET = 'R2R'
RESULT_DIR = 'tasks/R2R/results/'


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok, dataset='R2R'):
        self.dataset = dataset
        global DATASET
        DATASET = self.dataset
        global RESULT_DIR
        RESULT_DIR = 'tasks/' + DATASET + '/results/'

        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for split in splits:
            split_data = []
            for item in load_datasets([split], dataset):
                split_data.append(item)
                if scans is not None:
                    if DATASET == 'R2R' and item['scan'] not in scans:
                        continue
                    elif DATASET == 'SOON' and item['bboxes'][0]['scan'] not in scans:
                        continue
                self.gt[str(item['path_id'])] = copy.deepcopy(item)
                self.scans.append(item['scan'])
                if DATASET == 'SOON':
                    new_instrs = []
                    instructions = copy.deepcopy(item['instructions'])
                    for i in range(len(instructions)):
                        new_instrs.append(instructions[i][4])
                    self.gt[str(item['path_id'])]['instructions'] = new_instrs
                self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
            with open(f"eval1_{split}.json", "w") as f:
                json.dump(split_data, f)
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _image_bbox_overlap(self, boxes, query_boxes, criterion=-1):
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=boxes.dtype)
        for k in range(K):
            # x on the left of center: < 0, otherwise > 0
            # y below center: < 0, otherwise > 0
            qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 1] - query_boxes[k, 3]))
            for n in range(N):
                iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]))
                if iw > 0:
                    ih = math.fabs(max(boxes[n, 3], query_boxes[k, 3]) - min(boxes[n, 1], query_boxes[n, 1]))
                    if ih > 0:
                        if criterion == -1:
                            ua = ((boxes[n, 2] - boxes[n, 0]) * (boxes[n, 1] - boxes[n, 3]) + qbox_area - iw * ih)
                        elif criterion == 0:
                            ua = ((boxes[n, 2] - boxes[n, 0]) * (boxes[n, 1] - boxes[n, 3]))
                        elif criterion == 1:
                            ua = qbox_area
                        else:
                            ua = 1.0
                        overlaps[n, k] = iw * ih / ua
        return overlaps

    def _score_item(self, instr_id, path, heading=None, elevation=None, pre_bbox=None, pre_num_heading=None, pre_num_elevation=None):
        heading = np.array(heading)
        elevation = np.array(elevation)
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        if self.dataset == 'SOON':
            if not args.compute_bbox:
                pre_point = Point(heading, elevation)
                # pre_point = np.array([heading, elevation]).reshape(1, 2)
                success = False
                for bbox in gt['bboxes']:
                    if bbox['image_id'] == final_position:
                        goal = final_position
                        gt_heading = bbox['heading']
                        gt_elevation = bbox['elevation']
                        # gt_point = Point(gt_heading, gt_elevation)
                        gt_poly = Polygon([(bbox['target']['left_top']['heading'], bbox['target']['left_top']['elevation']),
                                           (bbox['target']['right_top']['heading'], bbox['target']['right_top']['elevation']),
                                           (bbox['target']['right_bottom']['heading'], bbox['target']['right_bottom']['elevation']),
                                           (bbox['target']['left_bottom']['heading'], bbox['target']['left_bottom']['elevation'])])
                        # gt_poly = mlpath.Path(np.array([(bbox['target']['left_top']['heading'], bbox['target']['left_top']['elevation']),
                        #                      (bbox['target']['right_top']['heading'], bbox['target']['right_top']['elevation']),
                        #                      (bbox['target']['right_bottom']['heading'], bbox['target']['right_bottom']['elevation']),
                        #                      (bbox['target']['left_bottom']['heading'], bbox['target']['left_bottom']['elevation'])]))

                        self.scores['heading_errors'].append(math.fabs((gt_heading - heading)))
                        self.scores['elevation_errors'].append(math.fabs((gt_elevation - elevation)))
                        if gt_poly.contains(pre_point):
                            # point_inds = (gt_poly.contains_points(pre_point) > 0).nonzero()[0]
                            # if point_inds.shape[0] > 0:
                            self.scores['det_success_num'].append(1.)
                            # self.scores['point_det_errors'].append(pre_point.distance(gt_point))
                            self.scores['point_det_errors'].append(math.hypot(gt_heading - heading, gt_elevation - elevation))
                            success = True
                        break
                if not success:
                    self.scores['det_success_num'].append(0.)
            else:
                if pre_bbox is not None:
                    pre_bbox = pre_bbox.reshape((1, 4))
                    success = False
                    for bbox in gt['bboxes']:
                        if bbox['image_id'] == final_position:
                            goal = final_position
                            gt_bboxes_2d = np.array([bbox['mouse_left_top']['x'], bbox['mouse_left_top']['y'],
                                                     bbox['mouse_right_bottom']['x'], bbox['mouse_right_bottom']['y']]).reshape((1, 4))
                            overlap = self._image_bbox_overlap(gt_bboxes_2d, pre_bbox)   # 1 x 1
                            self.scores['bbox_iou'].append(overlap[0, 0])
                            if overlap[0, 0] > 0.5 and pre_num_heading == bbox['num_heading'] and \
                                    pre_num_elevation == bbox['num_elevation']:
                                self.scores['det_success_num'].append(1.)
                                success = True
                                break
                    if not success:
                        self.scores['det_success_num'].append(0.)

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])

        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        self.scores['goal_progress'].append(
            self.distances[gt['scan']][start][goal] - self.distances[gt['scan']][final_position][goal]
        )

    def score(self, output_file, env_name=None):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                if self.dataset == 'R2R':
                    self._score_item(item['instr_id'], item['trajectory'])
                elif self.dataset == 'SOON':
                    if not args.compute_bbox:
                        self._score_item(item['instr_id'], item['trajectory']['path'],
                                         item['trajectory']['obj_heading'],
                                         item['trajectory']['obj_elevation'])
                    else:
                        self._score_item(item['instr_id'], item['trajectory']['path'],
                                         pre_bbox=item['trajectory']['pre_bbox'],
                                         pre_num_heading=item['trajectory']['pre_num_heading'],
                                         pre_num_elevation=item['trajectory']['pre_num_elevation'])
        if 'train' not in self.splits and self.dataset != 'SOON':  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                           % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['nav_errors']) == len(self.instr_ids)

        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        if self.dataset == 'SOON':
            if not args.compute_bbox:
                score_summary['heading_errors'] = np.average(self.scores['heading_errors'])
                score_summary['elevation_errors'] = np.average(self.scores['elevation_errors'])
                score_summary['point_det_errors'] = np.average(self.scores['point_det_errors'])
                score_summary['goal_progress'] = np.average(self.scores['goal_progress'])
            else:
                score_summary['bbox_iou'] = np.average(self.scores['bbox_iou'])

        if self.dataset == 'SOON':
            det_num_successes = len([i for i in self.scores['det_success_num'] if i > 0.])
            score_summary['det_success_rate'] = float(det_num_successes) / float(len(self.scores['det_success_num']))
            num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
            score_summary['nav_success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        else:
            num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
            score_summary['success_rate'] = float(num_successes) / float(len(self.scores['nav_errors']))

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)
        if self.dataset == 'SOON':
            if not args.one_image:
                success_rate = np.array(self.scores['det_success_num']) * np.array(spl)
                score_summary['success_rate'] = np.average(success_rate)
            else:
                score_summary['success_rate'] = score_summary['det_success_rate']
            score_summary['goal_progress'] = np.average(self.scores['goal_progress'])
        return score_summary, self.scores

    def bleu_score(self, path2inst):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            candidates.append([self.tok.index_to_word[word_id] for word_id in inst])

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions


def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    if DATASET == 'R2R':
        splits = ['train', 'val_seen', 'val_unseen', 'test']
    elif DATASET == 'SOON':
        splits = ['train', 'val_seen_instrs', 'val_unseen_instrs', 'val_unseen_house', 'test']
    for split in splits:
        env = R2RBatch(None, batch_size=1, splits=[split], dataset=DATASET)
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    if DATASET == 'R2R':
        splits = ['val_seen', 'val_unseen']
    elif DATASET == 'SOON':
        splits = ['val_seen_instrs', 'val_unseen_instrs', 'val_unseen_house']
    for outfile in outfiles:
        for split in splits:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    eval_simple_agents()





