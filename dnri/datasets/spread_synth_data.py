import numpy as np
import torch
from torch.utils.data import Dataset
from dnri.utils import data_utils
import copy

import argparse, os

# if you want to visualize tests, uncomment
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import copy

class SmallSynthData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'val_feats')
            edge_path = os.path.join(data_path, 'val_edges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')
        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)
        self.same_norm = params['same_data_norm']
        self.no_norm = params['no_data_norm']
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min)*2/(self.feat_max-self.feat_min) - 1
        else:
            self.loc_max = train_data[:, :, :, :2].max()
            self.loc_min = train_data[:, :, :, :2].min()
            self.vel_max = train_data[:, :, :, 2:].max()
            self.vel_min = train_data[:, :, :, 2:].min()
            self.feats[:,:,:, :2] = (self.feats[:,:,:,:2]-self.loc_min)*2/(self.loc_max - self.loc_min) - 1
            self.feats[:,:,:,2:] = (self.feats[:,:,:,2:]-self.vel_min)*2/(self.vel_max-self.vel_min)-1

    def unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        else:
            result1 = (data[:, :, :, :2] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            result2 = (data[:, :, :, 2:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
            return np.concatenate([result1, result2], axis=-1)


    def __getitem__(self, idx):
        return {'inputs': self.feats[idx], 'edges':self.edges[idx]}

    def __len__(self):
        return len(self.feats)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_val', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--num_time_steps', type=int, default=50)
    parser.add_argument('--pull_factor', type=float, default=0.3)
    parser.add_argument('--push_factor', type=float, default=0.3)

    args = parser.parse_args()
    # np.random.seed(1)
    all_data = []
    all_edges = []
    num_sims = args.num_train + args.num_val + args.num_test
    flip_count = 0
    total_steps = 0
    ################### 1) Spread code
    # landmark_locs = np.random.uniform(-2, 2, size=(num_sims, 6, 2))
    # for sim in range(num_sims):
    #     agent_goals = np.zeros(3)

    #     p_vels = np.random.uniform(-0.1, 0.1, size=(3, 2))
    #     p_locs = np.random.uniform(-2, 2, size=(3, 2))

    #     agent_locs = copy.deepcopy(p_locs)
    #     landmarks_left = copy.deepcopy(landmark_locs[sim, :3])
    #     blue_is= [0, 1, 2]
    #     for _ in range(3):
    #         blue_i = blue_is[0]

    #         dists_ag = [np.linalg.norm(landmark_locs[sim, blue_i] - al) for al in agent_locs]
    #         selected_agent = np.argmin(np.array(dists_ag))

    #         dists_land = [np.linalg.norm(agent_locs[selected_agent]  - l) for l in landmarks_left]
    #         selected_landmark = np.argmin(np.array(dists_land))

    #         if dists_ag[selected_agent] < dists_land[selected_landmark]:
    #             agent_goals[selected_agent] = blue_i
    #             agent_locs[selected_agent] = 10
    #             landmarks_left[blue_i] = 10
    #             blue_is.remove(blue_i)
    #         else:
    #             agent_goals[selected_agent] = selected_landmark
    #             agent_locs[selected_agent] = 10
    #             landmarks_left[selected_landmark] = 10
    #             blue_is.remove(selected_landmark)

    #     current_feats = []
    #     current_edges = []
    #     for time_step in range(args.num_time_steps):
    #         current_edge = np.array([1]*6)
    #         current_edges.append(current_edge)

    #         """
    #         edge to node convertion (i.e. 0th row means 
    #                                 edge sends 0th node data to 1st node)
    #                                 3rd row means 
    #                                 edge sends 1st node data to 2nd node)
    #         [0, 1, 0],
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, 1, 0]

    #         """
    #         for i in range(3):
    #             norm = np.linalg.norm( landmark_locs[sim, int(agent_goals[i])] - p_locs[i] )
    #             dir_1 = ( landmark_locs[sim, int(agent_goals[i])] - p_locs[i] )#/norm
    #             p_vels[i] = args.push_factor*dir_1
    #             if norm>0.5:
    #                 p_vels[i] = 0.5*p_vels[i]/ norm
    #             p_locs[i] += p_vels[i]

    #         p1_feat = np.concatenate([p_locs[0], p_vels[0], landmark_locs[sim].flatten(), landmark_locs[sim,3]])
    #         p2_feat = np.concatenate([p_locs[1], p_vels[1], landmark_locs[sim].flatten(), landmark_locs[sim,4]])
    #         p3_feat = np.concatenate([p_locs[2], p_vels[2], landmark_locs[sim].flatten(), landmark_locs[sim,5]])
    #         new_feat = np.stack([p1_feat, p2_feat, p3_feat])
    #         current_feats.append(new_feat)
    #     all_data.append(np.stack(current_feats))
    #     all_edges.append(np.stack(current_edges))
        
    # all_data = np.stack(all_data)

    # # for visualization of (1), uncomment the code below
    # # unnormalized_gt = all_data # dataset.unnormalize(all_data)
    # # fig, ax = plt.subplots()
    # # def update(frame):
    # #     ax.clear()
    # #     ax.plot(unnormalized_gt[0, frame, 0, 0], unnormalized_gt[0, frame, 0, 1], 'bo', alpha=0.5)
    # #     ax.plot(unnormalized_gt[0, frame, 1, 0], unnormalized_gt[0, frame, 1, 1], 'ro', alpha=0.5)
    # #     ax.plot(unnormalized_gt[0, frame, 2, 0], unnormalized_gt[0, frame, 2, 1], 'go', alpha=0.5)
    # #     for i in range(3):
    # #         ax.plot(landmark_locs[0, i, 0], landmark_locs[0, i, 1],'bx')
    # #         ax.plot(landmark_locs[0, i+3, 0], landmark_locs[0, i+3, 1],'rx')
        
    # #     ax.set_xlim(-2.5, 2.5)
    # #     ax.set_ylim(-2.5, 2.5)
    # # ani = animation.FuncAnimation(fig, update, interval=100, frames=30)
    # # path = os.path.join('./', 'pred_trajectory_1.mp4')
    # # ani.save(path, codec='mpeg4')

    ################### 2) Multi-Modal Simple
    landmark_locs = np.array([[0., 2.], [0., -2.]])
    for sim in range(num_sims):
        p_vels = np.random.uniform(-0.1, 0.1, size=(3, 2))
        p_locs = np.random.uniform(-1.5, 1.5, size=(3, 2))

        landmark_i = int(sim%2)

        current_feats = []
        current_edges = []
        for time_step in range(args.num_time_steps):
            current_edge = np.array([1]*6) # look at this later
            current_edges.append(current_edge)

            """
            edge to node convertion (i.e. 0th row means 
                                    edge sends 0th node data to 1st node)
                                    3rd row means 
                                    edge sends 1st node data to 2nd node)
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]

            """
            for i in range(3):
                norm = np.linalg.norm( landmark_locs[landmark_i] - p_locs[i] )
                dir_1 = ( landmark_locs[landmark_i] - p_locs[i] )#/norm
                p_vels[i] = args.push_factor*dir_1
                if norm>0.3:
                    p_vels[i] = 0.3*p_vels[i]/ norm
                p_locs[i] += p_vels[i]

            p1_feat = np.concatenate([p_locs[0], p_vels[0]])
            p2_feat = np.concatenate([p_locs[1], p_vels[1]])
            p3_feat = np.concatenate([p_locs[2], p_vels[2]])
            new_feat = np.stack([p1_feat, p2_feat, p3_feat])
            current_feats.append(new_feat)
        all_data.append(np.stack(current_feats))
        all_edges.append(np.stack(current_edges))
        
    all_data = np.stack(all_data)

    # for visualization of (2), uncomment the code below
    # unnormalized_gt = all_data # dataset.unnormalize(all_data)
    # fig, ax = plt.subplots()
    # def update(frame):
    #     ax.clear()
    #     ax.plot(unnormalized_gt[0, frame, 0, 0], unnormalized_gt[0, frame, 0, 1], 'bo', alpha=0.5)
    #     ax.plot(unnormalized_gt[0, frame, 1, 0], unnormalized_gt[0, frame, 1, 1], 'ro', alpha=0.5)
    #     ax.plot(unnormalized_gt[0, frame, 2, 0], unnormalized_gt[0, frame, 2, 1], 'go', alpha=0.5)

    #     ax.plot(landmark_locs[0, 0], landmark_locs[0, 1],'bx')
    #     ax.plot(landmark_locs[1, 0], landmark_locs[1, 1],'rx')
        
    #     ax.set_xlim(-2.5, 2.5)
    #     ax.set_ylim(-2.5, 2.5)
    # ani = animation.FuncAnimation(fig, update, interval=100, frames=30)
    # path = os.path.join('./', 'pred_trajectory_1.mp4')
    # ani.save(path, codec='mpeg4')

    train_data = torch.FloatTensor(all_data[:args.num_train])
    val_data = torch.FloatTensor(all_data[args.num_train:args.num_train+args.num_val])
    test_data = torch.FloatTensor(all_data[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_feats')
    torch.save(train_data, train_path)
    val_path = os.path.join(args.output_dir, 'val_feats')
    torch.save(val_data, val_path)
    test_path = os.path.join(args.output_dir, 'test_feats')
    torch.save(test_data, test_path)

    train_edges = torch.FloatTensor(all_edges[:args.num_train])
    val_edges = torch.FloatTensor(all_edges[args.num_train:args.num_train+args.num_val])
    test_edges = torch.FloatTensor(all_edges[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_edges')
    torch.save(train_edges, train_path)
    val_path = os.path.join(args.output_dir, 'val_edges')
    torch.save(val_edges, val_path)
    test_path = os.path.join(args.output_dir, 'test_edges')
    torch.save(test_edges, test_path)
