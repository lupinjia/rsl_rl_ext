import torch
import numpy as np

class Symmetry():
    def __init__(self, cfg):
        # convert to a list
        self.mirror_indices = [
            cfg["mirror_indices"]["sideneg_obs_inds"] + cfg["mirror_indices"]["neg_obs_inds"],
            cfg["mirror_indices"]["right_obs_inds"],
            cfg["mirror_indices"]["left_obs_inds"],
            cfg["mirror_indices"]["neg_act_inds"] + cfg["mirror_indices"]["sideneg_act_inds"],
            cfg["mirror_indices"]["right_act_inds"],
            cfg["mirror_indices"]["left_act_inds"],
        ]

    def get_mirror_function(self):

        # negation means the sign should reverse(multiply -1)
        negation_obs_indices = self.mirror_indices[0]
        right_obs_indices = self.mirror_indices[1]
        left_obs_indices = self.mirror_indices[2]
        negation_action_indices = self.mirror_indices[3]
        right_action_indices = self.mirror_indices[4]
        left_action_indices = self.mirror_indices[5]

        def mirror_function(trajectory_samples, return_both=False):
            observations_batch = trajectory_samples[0]
            states_batch = trajectory_samples[1]
            actions_batch = trajectory_samples[2]
            value_preds_batch = trajectory_samples[3]
            return_batch = trajectory_samples[4]
            masks_batch = trajectory_samples[5]
            old_action_log_probs_batch = trajectory_samples[6]
            adv_targ = trajectory_samples[7]

            def swap_lr(t, r, l):
                t[:, np.concatenate((r, l))] = t[:, np.concatenate((l, r))]

            # Only observation and action needs to be mirrored
            observations_clone = observations_batch.clone()
            actions_clone = actions_batch.clone()

            observations_clone[:, negation_obs_indices] *= -1
            swap_lr(observations_clone, right_obs_indices, left_obs_indices)

            actions_clone[:, negation_action_indices] *= -1
            swap_lr(actions_clone, right_action_indices, left_action_indices)

            if return_both:
                # Others need to be repeated
                observations_batch = torch.cat([observations_batch, observations_clone])
                actions_batch = torch.cat([actions_batch, actions_clone])
                states_batch = states_batch.repeat((2, 1))
                value_preds_batch = value_preds_batch.repeat((2, 1))
                return_batch = return_batch.repeat((2, 1))
                masks_batch = masks_batch.repeat((2, 1))
                old_action_log_probs_batch = old_action_log_probs_batch.repeat((2, 1))
                adv_targ = adv_targ.repeat((2, 1))
            else:
                observations_batch = observations_clone
                actions_batch = actions_clone

            return (
                observations_batch,
                states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

        return mirror_function
    
    def mirror_action(self, actions):
        negation_action_indices = self.mirror_indices[3]
        right_action_indices = self.mirror_indices[4]
        left_action_indices = self.mirror_indices[5]
        
        def swap_lr(t, r, l):
                t[:, np.concatenate((r, l))] = t[:, np.concatenate((l, r))]
        
        actions_clone = actions.clone()