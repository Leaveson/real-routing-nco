import torch


class New_Cluster:
    """
    Multiple gaussian distributed clusters, as in the Solomon benchmark dataset
    Following the setting in Bi et al. 2022 (https://arxiv.org/abs/2210.07686)

    Args:
        n_cluster: Number of the gaussian distributed clusters
    """

    def __init__(self, n_cluster: int = 3):
        super().__init__()
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07
        self.n_cluster = n_cluster

    def sample(self, size):

        batch_size, num_loc, _ = size
        # Generate the centers of the clusters
        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster * 2
        )

        # Pre-define the coordinates
        coords = torch.zeros(batch_size, num_loc, 2)

        # Set the first coordinate as the center of the first cluster (depot)
        coords[:, 0, :] = center[:, 0:2]

        # Calculate the size of each cluster (for customers only, excluding depot)
        num_customer = num_loc - 1
        cluster_sizes = [num_customer // self.n_cluster] * self.n_cluster
        for i in range(num_customer % self.n_cluster):
            cluster_sizes[i] += 1

        # Generate the coordinates (starting from index 1)
        current_index = 1
        for i in range(self.n_cluster):
            means = center[:, i * 2 : (i + 1) * 2]
            stds = torch.full((batch_size, 2), self.std)
            points = torch.normal(
                means.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
                stds.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
            )
            coords[:, current_index : current_index + cluster_sizes[i], :] = points
            current_index += cluster_sizes[i]

        # Confine the coordinates to range [0, 1]
        coords.clamp_(0, 1)

        # Scale coordinates so max distance between locs <= 0.3
        # Max diagonal distance in [0,1]^2 is sqrt(2) ≈ 1.414
        # To make it 0.3: scale = 0.3 / sqrt(2) ≈ 0.21
        scale = 0.21
        coords = coords * scale 
        
        # Shift so that the first node (depot) is at (0.5, 0.5)
        depot_shift = 0.5 - coords[:, 0:1, :]
        coords = coords + depot_shift

        # Verify all pairwise distances are <= 0.3
        pairwise_dist = torch.cdist(coords, coords, p=2)  # [batch_size, num_loc, num_loc]
        max_dist = pairwise_dist.max().item()
        assert max_dist <= 0.3, f"Max distance {max_dist:.4f} exceeds 0.3!"

        return coords


# import torch


# class New_Cluster:
#     """
#     Multiple gaussian distributed clusters, as in the Solomon benchmark dataset
#     Following the setting in Bi et al. 2022 (https://arxiv.org/abs/2210.07686)

#     Args:
#         n_cluster: Number of the gaussian distributed clusters
#     """

#     def __init__(self, n_cluster: int = 3):
#         super().__init__()
#         self.lower, self.upper = 0.2, 0.8
#         self.std = 0.07
#         self.n_cluster = n_cluster

#     def sample(self, size):

#         batch_size, num_loc, _ = size
#         # Generate the centers of the clusters
#         center = self.lower + (self.upper - self.lower) * torch.rand(
#             batch_size, self.n_cluster * 2
#         )

#         # Pre-define the coordinates
#         coords = torch.zeros(batch_size, num_loc, 2)

#         # Set the first coordinate as the center of the first cluster (depot)
#         coords[:, 0, :] = center[:, 0:2]

#         # Calculate the size of each cluster (for customers only, excluding depot)
#         num_customer = num_loc - 1
#         cluster_sizes = [num_customer // self.n_cluster] * self.n_cluster
#         for i in range(num_customer % self.n_cluster):
#             cluster_sizes[i] += 1

#         # Generate the coordinates (starting from index 1)
#         current_index = 1
#         for i in range(self.n_cluster):
#             means = center[:, i * 2 : (i + 1) * 2]
#             stds = torch.full((batch_size, 2), self.std)
#             points = torch.normal(
#                 means.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
#                 stds.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
#             )
#             coords[:, current_index : current_index + cluster_sizes[i], :] = points
#             current_index += cluster_sizes[i]

#         # Confine the coordinates to range [0, 1]
#         coords.clamp_(0, 1)

#         # Scale coordinates so max distance from depot <= 0.3 (dynamic scaling)
#         max_target_dist = 0.3
        
#         # Get depot coordinates
#         depot = coords[:, 0:1, :]  # [batch_size, 1, 2]
        
#         # Calculate distances from depot to all other nodes
#         dist_from_depot = torch.norm(coords - depot, dim=2)  # [batch_size, num_loc]
        
#         # Get max distance from depot per batch
#         max_dist_per_batch = dist_from_depot.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
#         max_dist_per_batch = max_dist_per_batch.unsqueeze(-1)  # [batch_size, 1, 1]
        
#         # Avoid division by zero
#         max_dist_per_batch = torch.clamp(max_dist_per_batch, min=1e-6)
        
#         # Compute scale factor per batch
#         scale = max_target_dist / max_dist_per_batch  # [batch_size, 1, 1]
        
#         # Scale coordinates around depot, then shift depot to (0.5, 0.5)
#         coords = (coords - depot) * scale + 0.5

#         return coords