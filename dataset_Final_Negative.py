import pandas as pd
import numpy as np
import torch
import os
import re
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional
import random

class GDataset:
    """Dataset handler for MAFENGWO group recommendation data"""
    
    def __init__(self, user_path: str, group_path: str, num_negatives: int = 4, random_seed: int = 42):
        """
        Initialize dataset
        
        Args:
            user_path: Path to user rating directory
            group_path: Path to group rating directory  
            num_negatives: Number of negative samples per positive sample during training
            random_seed: Random seed for reproducibility
        """
        self.user_path = user_path
        self.group_path = group_path
        self.num_negatives = num_negatives
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize data structures
        self.user_trainMatrix = {}  # (user, item) -> rating
        self.user_testRatings = []  # [(user, item), ...]
        self.user_testNegatives = []  # [negative_items, ...]
        
        self.group_trainMatrix = {}  # (group, item) -> rating
        self.group_testRatings = []  # [(group, item), ...]
        self.group_testNegatives = []  # [negative_items, ...]
        
        # Dataset constants - Mafengwo dimensions
        self.NUM_USERS = 5275
        self.NUM_ITEMS = 1513
        self.NUM_GROUPS = 995
        
        # Load all data
        self._load_user_data()
        self._load_group_data()
        
        print(f"Dataset loaded:")
        print(f"   User train interactions: {len(self.user_trainMatrix)}")
        print(f"   User test cases: {len(self.user_testRatings)}")
        print(f"   Group train interactions: {len(self.group_trainMatrix)}")
        print(f"   Group test cases: {len(self.group_testRatings)}")
    
    def _load_user_data(self):
        """Load user rating data"""
        print("Loading user rating data...")
        
        # Load training data
        train_file = os.path.join(self.user_path, "userRatingTrain.csv")
        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file, header=None, names=['user_id', 'item_id', 'rating'])
            for _, row in df_train.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = float(row['rating'])
                self.user_trainMatrix[(user_id, item_id)] = rating
        
        # Load test data
        test_file = os.path.join(self.user_path, "userRatingTest.csv")
        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file, header=None, names=['user_id', 'item_id', 'rating'])
            for _, row in df_test.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                self.user_testRatings.append((user_id, item_id))
        
        # Load negative samples
        neg_file = os.path.join(self.user_path, "userRatingNegative.csv")
        if os.path.exists(neg_file):
            self.user_testNegatives = self._load_negative_samples(neg_file)
    
    def _load_group_data(self):
        """Load group rating data"""
        print("Loading group rating data...")
        
        # Load training data
        train_file = os.path.join(self.group_path, "groupRatingTrain.csv")
        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file, header=None, names=['group_id', 'item_id', 'rating'])
            for _, row in df_train.iterrows():
                group_id = int(row['group_id'])
                item_id = int(row['item_id'])
                rating = float(row['rating'])
                self.group_trainMatrix[(group_id, item_id)] = rating
        
        # Load test data
        test_file = os.path.join(self.group_path, "groupRatingTest.csv")
        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file, header=None, names=['group_id', 'item_id', 'rating'])
            for _, row in df_test.iterrows():
                group_id = int(row['group_id'])
                item_id = int(row['item_id'])
                self.group_testRatings.append((group_id, item_id))
        
        # Load negative samples
        neg_file = os.path.join(self.group_path, "groupRatingNegative.csv")
        if os.path.exists(neg_file):
            self.group_testNegatives = self._load_negative_samples(neg_file)
    
    def _load_negative_samples(self, neg_file: str) -> List[List[int]]:
        """Load negative samples from file"""
        negative_samples = []
        
        try:
            with open(neg_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Parse format: (user_id,item_id) neg1 neg2 neg3 ...
                        match = re.match(r'\((\d+),(\d+)\)\s+(.+)', line)
                        if match:
                            neg_items_str = match.group(3)
                            neg_items = [int(x) for x in neg_items_str.split()]
                            negative_samples.append(neg_items)
                        else:
                            # Fallback: treat entire line as space-separated negative items
                            try:
                                neg_items = [int(x) for x in line.split() if x.isdigit()]
                                if neg_items:
                                    negative_samples.append(neg_items)
                            except:
                                negative_samples.append([])
        except Exception as e:
            print(f"Error loading negative samples from {neg_file}: {e}")
        
        return negative_samples
    
    def create_train_val_split(self, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[csr_matrix, csr_matrix]:
        """Create train/validation split from group training data using original indices"""
        np.random.seed(random_seed)
        
        # Use fixed dimensions to ensure consistency
        n_groups = self.NUM_GROUPS
        n_items = self.NUM_ITEMS
        
        # Collect all training data
        rows_train = []
        cols_train = []
        data_train = []
        rows_val = []
        cols_val = []
        data_val = []
        
        # Group data by group_id for splitting
        group_data = {}
        for (group_id, item_id), rating in self.group_trainMatrix.items():
            if group_id not in group_data:
                group_data[group_id] = []
            group_data[group_id].append((item_id, rating))
        
        # Split data for each group
        for group_id, items_ratings in group_data.items():
            if len(items_ratings) > 0:
                # Split items for this group
                n_val = max(1, int(len(items_ratings) * val_ratio))
                val_indices = np.random.choice(len(items_ratings), size=n_val, replace=False)
                val_indices_set = set(val_indices)
                
                for i, (item_id, rating) in enumerate(items_ratings):
                    if i in val_indices_set:
                        rows_val.append(group_id)
                        cols_val.append(item_id)
                        data_val.append(rating)
                    else:
                        rows_train.append(group_id)
                        cols_train.append(item_id)
                        data_train.append(rating)
        
        # FIXED: Get actual max indices from the data to handle sparse IDs
        max_group_idx = max(max(rows_train + rows_val), n_groups - 1) if (rows_train + rows_val) else n_groups - 1
        max_item_idx = max(max(cols_train + cols_val), n_items - 1) if (cols_train + cols_val) else n_items - 1
        
        # Create sparse matrices with dimensions that accommodate all indices
        matrix_shape = (max_group_idx + 1, max_item_idx + 1)
        
        train_matrix = csr_matrix((data_train, (rows_train, cols_train)), 
                                 shape=matrix_shape)
        val_matrix = csr_matrix((data_val, (rows_val, cols_val)), 
                               shape=matrix_shape)
        
        print(f"\nTrain/Val split created:")
        print(f"   Train interactions: {len(data_train)}")
        print(f"   Val interactions: {len(data_val)}")
        print(f"   Matrix shape: {matrix_shape}")
        
        return train_matrix, val_matrix
    
    def generate_training_samples(self, random_seed: int = 42):
        """Generate training samples with popularity-based hard negative sampling.

        Stage 1 (static hard negatives): instead of uniform random, negatives
        are drawn with probability proportional to item popularity (number of
        groups that interacted with the item).  Popular items are harder
        negatives because the model is more likely to confuse them with
        positives — they appear more often in training and share embeddings
        with many groups.
        """
        np.random.seed(random_seed)

        all_items = list(range(self.NUM_ITEMS))

        # ── Build item popularity weights (Stage 1) ───────────────────────────
        # popularity[i] = number of groups that interacted with item i
        popularity = np.zeros(self.NUM_ITEMS, dtype=np.float32)
        for (_, item_id) in self.group_trainMatrix.keys():
            popularity[item_id] += 1.0
        # Items with zero interactions get a small floor so they can still appear
        popularity = np.where(popularity == 0, 0.1, popularity)

        # Pre-build group -> interacted items map (avoids O(N) scan per group)
        group_items_map = {}
        for (group_id, item_id) in self.group_trainMatrix.keys():
            group_items_map.setdefault(group_id, set()).add(item_id)

        # Collect positive samples
        positive_samples = []
        for (group_id, item_id), rating in self.group_trainMatrix.items():
            positive_samples.append((group_id, item_id, 1.0))

        # Generate popularity-weighted negative samples
        negative_samples = []
        for (group_id, item_id), _ in self.group_trainMatrix.items():
            group_items = group_items_map.get(group_id, set())

            # Zero out interacted items so they cannot be sampled
            weights = popularity.copy()
            for gi in group_items:
                weights[gi] = 0.0
            total = weights.sum()
            if total == 0:
                continue
            weights /= total   # normalise to probability distribution

            neg_items = np.random.choice(
                all_items, size=self.num_negatives, replace=False, p=weights)
            for neg_item in neg_items:
                negative_samples.append((group_id, int(neg_item), 0.0))

        # Combine and shuffle
        all_samples = positive_samples + negative_samples
        np.random.shuffle(all_samples)
        return all_samples
    
    def load_group_membership(self, group_member_file: str) -> Optional[torch.Tensor]:
        """Load group membership information with fixed dimensions"""
        print(f"Loading group membership from {group_member_file}...")
        
        if not os.path.exists(group_member_file):
            print(f"Group membership file not found: {group_member_file}")
            return None
        
        try:
            df = pd.read_csv(group_member_file, header=None, names=['group_id', 'members'])
            
            group_members = {}
            
            for _, row in df.iterrows():
                group_id = int(row['group_id'])
                members_str = str(row['members'])
                
                # Parse member list
                if ',' in members_str:
                    members = [int(x.strip()) for x in members_str.split(',') if x.strip().isdigit()]
                else:
                    members = [int(members_str)] if members_str.isdigit() else []
                
                group_members[group_id] = members
            
            print(f"Loaded {len(group_members)} groups")
            
            # Determine actual max group size from data
            max_group_size = max(len(members) for members in group_members.values()) if group_members else 4
            print(f"Maximum group size in data: {max_group_size}")
            
            # Use NUM_USERS as padding value (consistent with embedding padding_idx)
            padding_value = self.NUM_USERS
            group_user_tensor = torch.full((self.NUM_GROUPS, max_group_size), padding_value, dtype=torch.long)
            
            # Fill in actual group memberships
            for group_id, members in group_members.items():
                if group_id < self.NUM_GROUPS:  # Ensure valid group index
                    for i, member_id in enumerate(members[:max_group_size]):  # Limit to max_group_size
                        if member_id < self.NUM_USERS:  # Ensure valid user index
                            group_user_tensor[group_id, i] = member_id
            
            print(f"Created group membership tensor: {group_user_tensor.shape}")
            return group_user_tensor
            
        except Exception as e:
            print(f"Error loading group membership: {e}")
            return None
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        # Get unique users and items from training data
        user_ids = set()
        item_ids = set()
        
        for (user_id, item_id) in self.user_trainMatrix.keys():
            user_ids.add(user_id)
            item_ids.add(item_id)
        
        # Get unique groups and items from group data
        group_ids = set()
        group_item_ids = set()
        
        for (group_id, item_id) in self.group_trainMatrix.keys():
            group_ids.add(group_id)
            group_item_ids.add(item_id)
        
        stats = {
            'n_users': len(user_ids),
            'n_items': len(item_ids),
            'n_groups': len(group_ids),
            'n_user_interactions': len(self.user_trainMatrix),
            'n_group_interactions': len(self.group_trainMatrix),
            'n_user_test': len(self.user_testRatings),
            'n_group_test': len(self.group_testRatings),
            'max_user_id': max(user_ids) if user_ids else 0,
            'max_item_id': max(item_ids.union(group_item_ids)) if item_ids.union(group_item_ids) else 0,
            'max_group_id': max(group_ids) if group_ids else 0,
        }
        
        return stats
    
    def generate_negative_samples_if_needed(self, num_negatives: int = 100) -> None:
        """Generate negative samples for test data if not available"""
        if len(self.group_testNegatives) != len(self.group_testRatings):
            print(f"Generating {num_negatives} negative samples for test data...")
            
            # Get all item IDs
            all_items = set(range(self.NUM_ITEMS))
            
            # Generate negative samples for group test data
            self.group_testNegatives = []
            for group_id, pos_item in self.group_testRatings:
                # Get items this group has interacted with
                group_items = set()
                for (g_id, item_id) in self.group_trainMatrix.keys():
                    if g_id == group_id:
                        group_items.add(item_id)
                group_items.add(pos_item)  # Add positive test item
                
                # Sample negative items
                available_items = list(all_items - group_items)
                if len(available_items) >= num_negatives:
                    negatives = np.random.choice(available_items, size=num_negatives, replace=False).tolist()
                else:
                    # If not enough unique negatives, sample with replacement
                    negatives = np.random.choice(list(all_items), size=num_negatives, replace=True).tolist()
                
                self.group_testNegatives.append(negatives)
            
            print(f"Generated {len(self.group_testNegatives)} negative sample lists")

# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    dataset = GDataset(
        user_path="./Data/Mafengwo/userRating",
        group_path="./Data/Mafengwo/groupRating",
        num_negatives=4,
        random_seed=42
    )
    
    # Print statistics
    stats = dataset.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test group membership loading
    group_user_ids = dataset.load_group_membership("./Data/Mafengwo/groupMember.csv")
    if group_user_ids is not None:
        print(f"\nGroup membership shape: {group_user_ids.shape}")
    
    # Test train/val split
    train_matrix, val_matrix = dataset.create_train_val_split(val_ratio=0.2)
    print(f"\nTrain/Val split:")
    print(f"   Train matrix shape: {train_matrix.shape}")
    print(f"   Val matrix shape: {val_matrix.shape}")
    print(f"   Train nnz: {train_matrix.nnz}")
    print(f"   Val nnz: {val_matrix.nnz}")
    
    # Test training sample generation
    training_samples = dataset.generate_training_samples()
    print(f"\nTraining samples generated: {len(training_samples)}")
    
    # Generate negative samples if needed
    dataset.generate_negative_samples_if_needed(num_negatives=100)