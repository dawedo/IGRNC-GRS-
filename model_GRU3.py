import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.nn.init as init
from dgl.nn import GNNExplainer
from torch.utils.data import Dataset
import copy
import numpy as np

# Mafengwo dataset dimensions
NUM_USERS = 5275
NUM_ITEMS = 1513
NUM_GROUPS = 995

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes):
        super(HeteroGCNLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_feats, out_feats, weight=True, bias=True)
            for etype in etypes
        })

    def forward(self, graph, inputs):
        return self.conv(graph, inputs)

class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, etypes, num_users=NUM_USERS, num_items=NUM_ITEMS, device=device):
        super(HeteroGCN, self).__init__()
        self.layer1 = HeteroGCNLayer(in_feats, hidden_size, etypes)
        self.layer2 = HeteroGCNLayer(hidden_size, out_feats, etypes)
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        
        # Output layer for preference prediction
        self.preference_layer = nn.Linear(out_feats * 2, 1)
        init.xavier_uniform_(self.preference_layer.weight)

    def forward(self, graph, feat, user_ids=None, item_ids=None):
        h = self.layer1(graph, feat)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.layer2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        
        if user_ids is not None and item_ids is not None:
            user_embeddings = h['user'][user_ids]
            item_embeddings = h['item'][item_ids]
            concat_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
            preferences = torch.sigmoid(self.preference_layer(concat_embeddings))
            return preferences.squeeze()
        
        return h

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users=NUM_USERS, num_items=NUM_ITEMS, num_groups=NUM_GROUPS, 
                 group_user_ids=None, embedding_size=64, dropout_rate=0.0):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.embedding_size = embedding_size
        
        self.user_embedding = nn.Embedding(num_users + 1, embedding_size, padding_idx=num_users)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.group_embedding = nn.Embedding(num_groups, embedding_size)
        self.group_user_ids = group_user_ids
        #GRU 
        self.group_gru = nn.GRU(embedding_size, embedding_size, batch_first=True)

        self.fc1 = nn.Linear(embedding_size * 2, 64)
        self.norm1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.fc2 = nn.Linear(64, 32)
        self.norm2 = nn.LayerNorm(32)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.fc3 = nn.Linear(32, 1)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def update_embeddings(self, user_features, item_features):
        """Update embeddings with external features from GNNExplainer"""
        device = next(self.parameters()).device
        user_features = user_features.to(device)
        item_features = item_features.to(device)
        
        zeros_tensor = torch.zeros(1, self.embedding_size).to(device)
        updated_user_features = torch.cat((user_features, zeros_tensor), dim=0)
        
        self.user_embedding = nn.Embedding.from_pretrained(updated_user_features, 
                                                          padding_idx=self.num_users)
        self.item_embedding = nn.Embedding.from_pretrained(item_features)
        
        self.user_embedding = self.user_embedding.to(device)
        self.item_embedding = self.item_embedding.to(device)

    def forward(self, group_indices, item_indices):
        device = next(self.parameters()).device
        group_indices = group_indices.to(device)
        item_indices = item_indices.to(device)
        
        # if self.group_user_ids is not None:
        #     group_user_ids_device = self.group_user_ids.to(device)
        #     group_members = group_user_ids_device[group_indices]
        #     group_embedded = self.user_embedding(group_members)
        #     group_embedded = torch.sum(group_embedded, dim=1)
        # else:
        #     group_embedded = self.group_embedding(group_indices)
        if self.group_user_ids is not None:
            group_user_ids_device = self.group_user_ids.to(device)
            group_members = group_user_ids_device[group_indices]
            member_embeds = self.user_embedding(group_members)
            h0 = self.group_embedding(group_indices).unsqueeze(0)
            _, h_n = self.group_gru(member_embeds, h0)
            group_embedded = h_n.squeeze(0)
        else:
            group_embedded = self.group_embedding(group_indices)

        item_embedded = self.item_embedding(item_indices)
        
        concatenated = torch.cat([group_embedded, item_embedded], dim=1)

        x = torch.relu(self.fc1(concatenated))
        x = self.norm1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout2(x)

        output = torch.sigmoid(self.fc3(x))
        return output.squeeze()

class EXPLAINER:
    def __init__(self, features, device):
        self.FEATURES = copy.deepcopy(features)
        self.device = device
        self.initial_features = copy.deepcopy(features)
        # Z_INFLUENTIAL stores the delta (updated_features - initial_features).
        # Initialized to zero so the first embedding fusion is a no-op.
        self.Z_INFLUENTIAL = {
            'user': torch.zeros_like(features['user']),
            'item': torch.zeros_like(features['item']),
        }

    def explain(self, model, graph, num_epochs, graph_data):
        """
        Run GNNExplainer on the GCN model to get feature and edge importance masks.
        Update node features and Z_INFLUENTIAL based on the explanation.
        """
        try:
            class ModelWrapper(nn.Module):
                def __init__(self, original_model, graph_data):
                    super().__init__()
                    self.model = original_model
                    self.graph_data = graph_data
                
                def forward(self, graph, feat):
                    user_item_edges = self.graph_data[('user', 'interacts', 'item')]
                    if len(user_item_edges[0]) > 0:
                        sample_size = min(100, len(user_item_edges[0]))
                        user_ids = user_item_edges[0][:sample_size]
                        item_ids = user_item_edges[1][:sample_size]
                        preferences = self.model(graph, feat, user_ids, item_ids)
                        return torch.mean(preferences)
                    else:
                        return torch.tensor(0.0, requires_grad=True).to(feat['user'].device)
            
            wrapped_model = ModelWrapper(model, graph_data).to(self.device)
            
            try:
                explainer = GNNExplainer(wrapped_model, num_hops=1)
                feat_mask, edge_mask = explainer.explain_graph(graph, self.FEATURES, num_epochs=num_epochs)
            except TypeError:
                try:
                    explainer = GNNExplainer(wrapped_model, num_hops=1)
                    feat_mask, edge_mask = explainer.explain_graph(graph, self.FEATURES)
                except Exception:
                    self._alternative_explanation_approach(model, graph, graph_data)
                    return

            if isinstance(feat_mask, dict):
                user_feat_imp = feat_mask.get('user', torch.ones_like(self.FEATURES['user']))
                item_feat_imp = feat_mask.get('item', torch.ones_like(self.FEATURES['item']))
            else:
                user_feat_imp = torch.ones_like(self.FEATURES['user'])
                item_feat_imp = torch.ones_like(self.FEATURES['item'])

            if isinstance(edge_mask, dict):
                user_item_edge_imp = edge_mask.get(('user', 'interacts', 'item'), 
                                                 torch.ones(graph.number_of_edges(('user', 'interacts', 'item'))).to(self.device))
            else:
                user_item_edge_imp = torch.ones(graph.number_of_edges(('user', 'interacts', 'item'))).to(self.device)

            self._apply_explanation_updates(graph_data, user_item_edge_imp, user_feat_imp, item_feat_imp)
            
        except Exception as e:
            print(f"Warning: Explainer failed with error: {e}")
            try:
                self._alternative_explanation_approach(model, graph, graph_data)
            except Exception as e2:
                print(f"Warning: Alternative explainer also failed: {e2}")
                self._apply_fallback_updates()

    def _alternative_explanation_approach(self, model, graph, graph_data):
        """
        Gradient-based importance when GNNExplainer fails.
        """
        model.eval()
        
        user_item_edges = graph_data[('user', 'interacts', 'item')]
        if len(user_item_edges[0]) == 0:
            self._apply_fallback_updates()
            return
            
        sample_size = min(100, len(user_item_edges[0]))
        user_ids = user_item_edges[0][:sample_size]
        item_ids = user_item_edges[1][:sample_size]
        
        self.FEATURES['user'].requires_grad_(True)
        self.FEATURES['item'].requires_grad_(True)
        
        predictions = model(graph, self.FEATURES, user_ids, item_ids)
        loss = torch.mean(predictions)
        loss.backward()
        
        user_feat_imp = torch.abs(self.FEATURES['user'].grad)
        item_feat_imp = torch.abs(self.FEATURES['item'].grad)
        
        edge_imp = []
        for i in range(sample_size):
            user_idx = user_ids[i].item()
            item_idx = item_ids[i].item()
            importance = (torch.mean(user_feat_imp[user_idx]) + torch.mean(item_feat_imp[item_idx])) / 2
            edge_imp.append(importance)
        
        num_edges = graph.number_of_edges(('user', 'interacts', 'item'))
        if len(edge_imp) > 0:
            avg_importance = torch.mean(torch.stack(edge_imp))
            user_item_edge_imp = torch.full((num_edges,), avg_importance.item()).to(self.device)
        else:
            user_item_edge_imp = torch.ones(num_edges).to(self.device)
        
        self.FEATURES['user'].grad = None
        self.FEATURES['item'].grad = None
        self.FEATURES['user'].requires_grad_(False)
        self.FEATURES['item'].requires_grad_(False)
        
        self._apply_explanation_updates(graph_data, user_item_edge_imp, user_feat_imp, item_feat_imp)
        
        print("Using gradient-based alternative explanation approach")

    def _apply_explanation_updates(self, graph_data, user_item_edge_imp, user_feat_imp, item_feat_imp):
        """
        Apply explanation updates and store the resulting delta in Z_INFLUENTIAL.
        """
        user_item_edge = graph_data[('user', 'interacts', 'item')]
        num_edges = len(user_item_edge[0])
        
        new_user_features = self.initial_features['user'].clone()
        new_item_features = self.initial_features['item'].clone()
        
        importance_scale = 0.1
        
        for i in range(min(num_edges, len(user_item_edge_imp))):
            try:
                user_idx = user_item_edge[0][i].item()
                item_idx = user_item_edge[1][i].item()
                edge_imp = user_item_edge_imp[i].item()
                
                if (user_idx < new_user_features.size(0) and 
                    item_idx < new_item_features.size(0)):
                    
                    user_update = edge_imp * user_feat_imp[user_idx] * importance_scale
                    item_update = edge_imp * item_feat_imp[item_idx] * importance_scale
                    
                    new_user_features[user_idx] += user_update * new_user_features[user_idx]
                    new_item_features[item_idx] += item_update * new_item_features[item_idx]
                    
            except (IndexError, RuntimeError):
                continue

        # Normalize updated features
        self.FEATURES['user'] = F.normalize(new_user_features, p=2, dim=1).to(self.device)
        self.FEATURES['item'] = F.normalize(new_item_features, p=2, dim=1).to(self.device)

        # Store delta relative to initial features so the training loop can do:
        #   master_features[key] + explainer.Z_INFLUENTIAL[key]
        self.Z_INFLUENTIAL['user'] = (self.FEATURES['user'] - self.initial_features['user'].to(self.device)).detach()
        self.Z_INFLUENTIAL['item'] = (self.FEATURES['item'] - self.initial_features['item'].to(self.device)).detach()

    def _apply_fallback_updates(self):
        """Fallback: apply small random influence when explainer fails."""
        noise_scale = 0.01
        self.FEATURES['user'] = self.FEATURES['user'] + torch.randn_like(self.FEATURES['user']) * noise_scale
        self.FEATURES['item'] = self.FEATURES['item'] + torch.randn_like(self.FEATURES['item']) * noise_scale
        
        self.FEATURES['user'] = F.normalize(self.FEATURES['user'], p=2, dim=1).to(self.device)
        self.FEATURES['item'] = F.normalize(self.FEATURES['item'], p=2, dim=1).to(self.device)

        # Store delta
        self.Z_INFLUENTIAL['user'] = (self.FEATURES['user'] - self.initial_features['user'].to(self.device)).detach()
        self.Z_INFLUENTIAL['item'] = (self.FEATURES['item'] - self.initial_features['item'].to(self.device)).detach()

class CustomDataset(Dataset):
    def __init__(self, group_indices, item_indices, ratings):
        self.group_indices = group_indices
        self.item_indices = item_indices
        self.ratings = ratings

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        return self.group_indices[idx], self.item_indices[idx], self.ratings[idx]