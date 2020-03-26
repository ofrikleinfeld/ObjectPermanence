from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractCaterModel(nn.Module):

    def __init__(self, config: Dict[str, int]):
        super().__init__()
        self.config: Dict[str, int] = config
        self.max_objects_in_frame = 15
        self.bb_in_dim = 5
        self.bb_out_dim = 4


class OPNet(AbstractCaterModel):
    def __init__(self, config: Dict[str, int]):
        super().__init__(config)
        self.bb_in_dim = 6
        object_to_track_dim = config["object_to_track_pred_dim"]

        object_to_track_lstm_in_dim = self.bb_in_dim * 15
        object_to_track_lstm_hidden_dim = config["object_to_track_hidden_dim"]
        video_lstm_hidden_dim: int = config["videos_hidden_dim"]

        # self.boxes_linear = nn.Linear(in_features=self.bb_in_dim, out_features=boxes_features_dim, bias=False)
        self.object_to_track_LSTM = nn.LSTM(input_size=object_to_track_lstm_in_dim, hidden_size=object_to_track_lstm_hidden_dim, num_layers=1, bidirectional=False, batch_first=True, bias=False)
        self.object_to_track_prediction = nn.Linear(in_features=object_to_track_lstm_hidden_dim, out_features=object_to_track_dim, bias=False)

        self.video_LSTM = nn.LSTM(input_size=self.bb_in_dim, hidden_size=video_lstm_hidden_dim, num_layers=1, bidirectional=False, batch_first=True, bias=False)
        self.prediction_layer = nn.Linear(in_features=video_lstm_hidden_dim, out_features=self.bb_out_dim, bias=False)

    def forward(self, boxes: torch.tensor) -> torch.tensor:
        batch_size, num_frames, num_objects, num_tracks_for_object = boxes.size()
        scene_features = boxes.view(batch_size, num_frames, -1)

        object_to_track_features, _ = self.object_to_track_LSTM(scene_features)
        object_to_track_prediction = self.object_to_track_prediction(object_to_track_features)
        object_to_track_index_probs = F.softmax(object_to_track_prediction, dim=-1)

        frames_boxes = torch.einsum("bfot,bfo->bft", boxes, object_to_track_index_probs)

        # sequence predictions using a single layer LSTM
        hidden, _ = self.video_LSTM(frames_boxes)
        y_boxes = self.prediction_layer(hidden)

        # transpose for CE loss that expects dimensions of batch, prediction, sequence
        object_to_track_prediction = object_to_track_prediction.permute(0, 2, 1).contiguous()

        return y_boxes, object_to_track_prediction


class OPNetLstmMlp(AbstractCaterModel):
    def __init__(self, config: Dict[str, int]):
        super().__init__(config)
        self.bb_in_dim = 6
        object_to_track_dim = config["object_to_track_pred_dim"]

        object_to_track_lstm_in_dim = self.bb_in_dim * 15
        object_to_track_lstm_hidden_dim = config["object_to_track_hidden_dim"]
        video_lstm_hidden_dim: int = config["videos_hidden_dim"]

        # self.boxes_linear = nn.Linear(in_features=self.bb_in_dim, out_features=boxes_features_dim, bias=False)
        self.object_to_track_LSTM = nn.LSTM(input_size=object_to_track_lstm_in_dim, hidden_size=object_to_track_lstm_hidden_dim, num_layers=1, bidirectional=False, batch_first=True, bias=False)
        self.object_to_track_prediction = nn.Linear(in_features=object_to_track_lstm_hidden_dim, out_features=object_to_track_dim, bias=False)

        self.hidden_layer = nn.Linear(in_features=self.bb_in_dim, out_features=video_lstm_hidden_dim, bias=False)
        self.prediction_layer = nn.Linear(in_features=video_lstm_hidden_dim, out_features=self.bb_out_dim, bias=False)

    def forward(self, boxes: torch.tensor) -> torch.tensor:
        batch_size, num_frames, num_objects, num_tracks_for_object = boxes.size()
        scene_features = boxes.view(batch_size, num_frames, -1)

        object_to_track_features, _ = self.object_to_track_LSTM(scene_features)
        object_to_track_prediction = self.object_to_track_prediction(object_to_track_features)
        object_to_track_index_probs = F.softmax(object_to_track_prediction, dim=-1)

        frames_boxes = torch.einsum("bfot,bfo->bft", boxes, object_to_track_index_probs)

        # replace LSTM with 1 layer MLP with the same size
        hidden = F.relu(self.hidden_layer(frames_boxes))
        y_boxes = self.prediction_layer(hidden)

        # transpose for CE loss that expects dimensions of batch, prediction, sequence
        object_to_track_prediction = object_to_track_prediction.permute(0, 2, 1).contiguous()

        return y_boxes, object_to_track_prediction


class BaselineLstm(AbstractCaterModel):

    def __init__(self, config: Dict[str, int]):
        super().__init__(config)

        lstm_in_dim = self.max_objects_in_frame * self.bb_in_dim
        lstm_hidden_dim: int = config["videos_hidden_dim"]

        self.video_LSTM = nn.LSTM(input_size=lstm_in_dim, hidden_size=lstm_hidden_dim,
                                  num_layers=1, bidirectional=False, batch_first=True, bias=False)
        self.predictions_layer = nn.Linear(in_features=lstm_hidden_dim, out_features=self.bb_out_dim, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # boxes dimension - batch, num_frames, num_objects_in_frame (15), boxes (5, including real/padding bit)
        batch_size, frames_dim, objects_dim, features_dim = x.size()

        # combine the objects dimensions and features dimensions into one dimension
        # representing a scene vector
        scene_features = x.view((batch_size, frames_dim, objects_dim * features_dim))

        # sequence predictions using a single layer LSTM
        hidden, _ = self.video_LSTM(scene_features)

        # transform to 4 number presenting x, y, x, y
        y_boxes = self.predictions_layer(hidden)

        return y_boxes


class NonLinearLstm(AbstractCaterModel):
    def __init__(self, config: Dict[str, int]):
        super().__init__(config)
        boxes_features_dim = config["boxes_features_dim"]
        lstm_hidden_dim: int = config["videos_hidden_dim"]

        lstm_in_dim = self.max_objects_in_frame * boxes_features_dim
        lstm_out_dim = lstm_hidden_dim

        self.boxes_linear = nn.Linear(in_features=self.bb_in_dim, out_features=boxes_features_dim, bias=False)
        self.video_LSTM = nn.LSTM(input_size=lstm_in_dim, hidden_size=lstm_hidden_dim,
                                  num_layers=2, bidirectional=False, batch_first=True, bias=False)
        self.predictions_layer = nn.Linear(in_features=lstm_out_dim, out_features=self.bb_out_dim, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # boxes dimension - batch, num_frames, num_objects_in_frame (15), boxes (5, including real/padding bit)
        batch_size, frames_dim, objects_dim, features_dim = x.size()

        boxes_features = F.relu(self.boxes_linear(x))

        # combine the objects dimensions and features dimensions into one dimension
        # representing a scene vector
        scene_features = boxes_features.view((batch_size, frames_dim, -1)).contiguous()

        # sequence predictions using a single layer LSTM
        hidden, _ = self.video_LSTM(scene_features)

        # transform to 4 number presenting x, y, x, y
        y_boxes = self.predictions_layer(hidden)

        return y_boxes


class TransformerLstm(AbstractCaterModel):
    def __init__(self, config: Dict[str, int]):
        super().__init__(config)
        boxes_features_dim = config["boxes_features_dim"]
        num_attention_heads = config["num_attention_heads"]
        num_attention_layers = config["num_attention_layers"]
        num_lstm_layers = config["num_lstm_layers"]
        bilstm_hidden_dim: int = config["lstm_hidden_dim"]

        bilstm_in_dim = boxes_features_dim
        bilstm_out_dim = bilstm_hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=boxes_features_dim, nhead=num_attention_heads)
        self.boxes_linear = nn.Linear(in_features=self.bb_in_dim, out_features=boxes_features_dim, bias=False)
        self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        self.video_LSTM = nn.LSTM(input_size=bilstm_in_dim, hidden_size=bilstm_hidden_dim,
                                  num_layers=num_lstm_layers, bidirectional=False, batch_first=True, bias=False)
        self.predictions_layer = nn.Linear(in_features=bilstm_out_dim, out_features=self.bb_out_dim, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # boxes dimension - batch, num_frames, num_objects_in_frame (15), boxes (5, including real/padding bit)
        batch_size, frames_dim, objects_dim, features_dim = x.size()

        boxes_features = F.relu(self.boxes_linear(x))

        # combine batch dimension and frame dimension
        # let the model encode for each frame which are the important objects bounding boxes
        # using self attention between each object and all the others
        objects_sequence = boxes_features.view((batch_size * frames_dim, objects_dim, -1)).contiguous()
        attended_objects = self.attention_encoder(objects_sequence)
        attended_snitch = attended_objects[:, 0, :]  # the snitch is always encoded as the first object in frame

        # transform back to batch, frames, features
        # this time features is the attended snitch
        scene_features = attended_snitch.view((batch_size, frames_dim, -1)).contiguous()

        # sequence predictions using BiLSTM
        hidden, _ = self.video_LSTM(scene_features)

        # transform to 4 number presenting x, y, x, y
        y_boxes = self.predictions_layer(hidden)

        return y_boxes
