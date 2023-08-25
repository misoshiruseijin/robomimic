"""
next_obs prediction models, used in HBC / IRIS.
"""
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

import robomimic.models.obs_nets as ObsNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, Algo
from robomimic.utils.obs_utils import ImageModality


@register_algo_factory_func("error_detector")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the ErrorDetector algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.conditional:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return ConditionalErrorDetector, {}
    else:
        print("??????????????????????????")
        return ErrorDetector, {}
    
class ErrorDetector(Algo):
    """
    Implements conditional error detector
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        super(ErrorDetector, self).__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = VAENets.VAE(
            input_shapes=self.obs_shapes,
            output_shapes=self.obs_shapes,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()

        # remove temporal batches for all except scalar signals (to be compatible with model outputs)
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ErrorDetector, self).train_on_batch(batch, epoch, validate=validate)

            # batch variables
            obs = batch["obs"]

            vae_outputs = self.nets["policy"](
                inputs=obs, # encoder takes observations
                outputs=obs, # reconstruct observations
            )
            recons_loss = vae_outputs["reconstruction_loss"]
            kl_loss = vae_outputs["kl_loss"]
            goal_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
            info["recons_loss"] = recons_loss
            info["kl_loss"] = kl_loss
            info["goal_loss"] = goal_loss

            with torch.no_grad():
                info["encoder_variance"] = torch.exp(vae_outputs["encoder_params"]["logvar"])

            # VAE gradient step
            if not validate:
                goal_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["policy"],
                    optim=self.optimizers["policy"],
                    loss=goal_loss,
                )
                info["goal_grad_norms"] = goal_grad_norms

        return info


    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["reconstruction_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        loss_log = super(ErrorDetector, self).log_info(info)

        loss_log["KL_Loss"] = info["kl_loss"].item()
        loss_log["Reconstruction_Loss"] = info["recons_loss"].item()

        return loss_log

    def get_reconstruction_loss(self, batch):
        """
        Get policy action outputs.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            recons_loss (float): reconstruction loss
        """
        with torch.no_grad():
            obs = batch["obs"]
            vae_outputs = self.nets["policy"](
                inputs=obs, # encoder takes observations
                outputs=obs, # reconstruct observations
            )
            recons_loss = vae_outputs["reconstruction_loss"]
        return recons_loss.item()
    
    def get_reconstructed_obs(self, batch, file_path=None):
        """
        Get policy action outputs.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            reconstruction (np.array): reconstructed image
        """
        with torch.no_grad():
            obs = batch["obs"]
            vae_outputs = self.nets["policy"](
                inputs=obs, # encoder takes observations
                outputs=obs, # reconstruct observations
            )
            reconstruction = ImageModality._default_obs_unprocessor(vae_outputs["decoder_outputs"]["rgb"])
            reconstruction = torch.squeeze(reconstruction, 0)
        return reconstruction.cpu().detach().numpy()

class ConditionalErrorDetector(Algo):
    """
    Implements conditional error detector
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        super(ConditionalErrorDetector, self).__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = VAENets.VAE(
            input_shapes=self.obs_shapes,
            output_shapes=self.obs_shapes,
            condition_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()

        # remove temporal batches for all except scalar signals (to be compatible with model outputs)
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, 0, :] for k in batch["next_obs"]}

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ConditionalErrorDetector, self).train_on_batch(batch, epoch, validate=validate)

            # batch variables
            obs = batch["obs"]
            next_obs = batch["next_obs"]

            vae_outputs = self.nets["policy"](
                inputs=next_obs, # encoder takes goal observations
                outputs=next_obs, # reconstruct goal observations
                conditions=obs, # condition on observations
            )
            recons_loss = vae_outputs["reconstruction_loss"]
            kl_loss = vae_outputs["kl_loss"]
            goal_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
            info["recons_loss"] = recons_loss
            info["kl_loss"] = kl_loss
            info["goal_loss"] = goal_loss

            with torch.no_grad():
                info["encoder_variance"] = torch.exp(vae_outputs["encoder_params"]["logvar"])

            # VAE gradient step
            if not validate:
                goal_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["policy"],
                    optim=self.optimizers["policy"],
                    loss=goal_loss,
                )
                info["goal_grad_norms"] = goal_grad_norms

        return info


    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["reconstruction_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        loss_log = super(ConditionalErrorDetector, self).log_info(info)

        loss_log["KL_Loss"] = info["kl_loss"].item()
        loss_log["Reconstruction_Loss"] = info["recons_loss"].item()

        return loss_log

    def get_reconstruction_loss(self, batch):
        """
        Get policy action outputs.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            recons_loss (float): reconstruction loss
        """
        with torch.no_grad():
            obs = batch["obs"]
            next_obs = batch["next_obs"]
            
            vae_outputs = self.nets["policy"](
                inputs=next_obs, # encoder takes goal observations
                outputs=next_obs, # reconstruct goal observations
                conditions=obs, # condition on observations
            )
            recons_loss = vae_outputs["reconstruction_loss"]
        return recons_loss.item()
    
    def get_reconstructed_obs(self, batch, file_path=None):
        """
        Get policy action outputs.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            reconstruction (np.array): reconstructed image
        """
        with torch.no_grad():
            obs = batch["obs"]
            next_obs = batch["next_obs"]

            vae_outputs = self.nets["policy"](
                inputs=next_obs, # encoder takes goal observations
                outputs=next_obs, # reconstruct goal observations
                conditions=obs, # condition on observations
            )
            reconstruction = ImageModality._default_obs_unprocessor(vae_outputs["decoder_outputs"]["rgb"])
            reconstruction = torch.squeeze(reconstruction, 0)
        return reconstruction.cpu().detach().numpy()