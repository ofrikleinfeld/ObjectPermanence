from typing import Dict, Any

import torch
import torch.nn as nn

from baselines.detector import CaterObjectDetector
from baselines.programmed_models import AbstractReasoner, ObjectDetectWithSiamTracker, HeuristicReasoner
from baselines.learned_models import AbstractCaterModel, BaselineLstm, NonLinearLstm, TransformerLstm, OPNet, OPNetLstmMlp
from baselines.DaSiamRPN.code.net import SiamRPNvot
from object_indices import OBJECTS_NAME_TO_IDX


class ModelsFactory(object):

    @staticmethod
    def get_tracker_model(model_name: str, model_weights: str, compute_device: torch.device) -> AbstractReasoner:
        snitch_index = 140

        if model_name == "detector_tracker":

            # initiate DaSiamRPN tracker and load its weights
            da_siam_net: nn.Module = SiamRPNvot()
            da_siam_net.load_state_dict(torch.load(model_weights))
            da_siam_net.to(compute_device)
            da_siam_net.eval()

            return ObjectDetectWithSiamTracker(snitch_index, da_siam_net, compute_device)

        elif model_name == "detector_heuristic":
            return HeuristicReasoner(index_to_track=snitch_index)

        else:
            raise AttributeError("Tracking model name is incorrect")

    @staticmethod
    def get_detector_model(model_name: str, model_weights: str = None) -> CaterObjectDetector:

        if model_name == "object_detector":
            return CaterObjectDetector(model_weights, OBJECTS_NAME_TO_IDX)

    @staticmethod
    def get_model(model_name: str, model_config: Dict[str, int], model_weights_path: str = None) -> AbstractCaterModel:
        if model_name == "baseline_lstm":
            model: AbstractCaterModel = BaselineLstm(model_config)

        elif model_name == "baseline_lstm_no_labels":
            model: AbstractCaterModel = BaselineLstm(model_config)

        elif model_name == "non_linear_lstm":
            model: AbstractCaterModel = NonLinearLstm(model_config)

        elif model_name == "non_linear_lstm_no_labels":
            model: AbstractCaterModel = NonLinearLstm(model_config)

        elif model_name == "transformer_lstm":
            model: AbstractCaterModel = TransformerLstm(model_config)

        elif model_name == "transformer_lstm_no_labels":
            model: AbstractCaterModel = TransformerLstm(model_config)

        elif model_name == "opnet":
            model: AbstractCaterModel = OPNet(model_config)

        elif model_name == "opent_no_labels":
            model: AbstractCaterModel = OPNet(model_config)

        elif model_name == "opnet_lstm_mlp":
            model: AbstractCaterModel = OPNetLstmMlp(model_config)

        elif model_name == "opnet_lstm_mlp_no_labels":
            model: AbstractCaterModel = OPNetLstmMlp(model_config)

        else:
            raise AttributeError("Model name is incorrect")

        if model_weights_path is not None:
            model.load_state_dict(torch.load(model_weights_path, map_location="cuda:0"))
            print(f"Loaded model parameters from {model_weights_path}")

        return model
