import os
import sys
import random
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from typing import Union, Callable
from torch.distributions import Uniform
from .distribution import New_Cluster

def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP),
    i.e. 260 at 2K nodes, 350 at 5K nodes and 500 at 10K nodes.
    Note that this serves as a demand scaler.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap

log = get_pylogger(__name__)

class SMTVRPGenerator(Generator):
    """SMTVRP Generator.
    Generates instances based on vrp-benchmarks logic with dynamic travel times.
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1000.0,  # Not used directly, but kept for compatibility
        loc_distribution: Union[int, float, str, type, Callable] = "cluster",
        capacity: float = None,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        num_cities: int = None,
        num_depots: int = 1,
        is_dynamic: bool = False, # Default to dynamic as per requirements
        scale_demand: bool = True,
        **kwargs,
    ) -> None:
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.capacity = capacity if capacity is not None else get_vehicle_capacity(num_loc) # Default for spec
        self.num_cities = num_cities if num_cities else max(1, num_loc // 50)
        self.num_depots = num_depots
        self.is_dynamic = is_dynamic
        self.scale_demand = scale_demand
        self.max_demand = max_demand
        self.demand_distribution = demand_distribution

        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = New_Cluster(n_cluster=self.num_cities)

        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand - 1, max_demand - 1
            )

    def _check_feasibility(
        self, 
        dist_matrix: torch.Tensor, 
        dur_matrix: torch.Tensor, 
        tws: torch.Tensor,
        max_depot_dist: float = 300.0,
        depot_end_time: float = 1440.0,
    ) -> torch.Tensor:
        """
        Feasibility 체크: 각 인스턴스가 feasible한지 확인
        
        조건:
        1. depot(index 0)과 모든 노드 사이의 거리 <= max_depot_dist
        2. max(depot까지 가는 시간, 노드의 start_time_window) + depot으로 돌아가는 시간 <= depot_end_time
        
        Args:
            dist_matrix: (B, N+1, N+1) 거리 행렬
            dur_matrix: (B, N+1, N+1) 시간 행렬
            tws: (B, N+1, 2) time windows [start, end]
            max_depot_dist: depot과 노드 사이 최대 허용 거리
            depot_end_time: depot의 end time window (기본값 1440분 = 24시간)
            
        Returns:
            feasible: (B,) bool 텐서, 각 인스턴스의 feasibility
        """
        batch_size = dist_matrix.shape[0]
        
        # 조건 1: depot(index 0)과 모든 노드 사이의 거리 <= max_depot_dist
        # dist_matrix[:, 0, :] -> depot에서 모든 노드까지의 거리 (B, N+1)
        depot_to_all_dist = dist_matrix[:, 0, :]  # (B, N+1)
        dist_condition = (depot_to_all_dist <= max_depot_dist).all(dim=1)  # (B,)
        
        # 조건 2: max(depot_to_node_dur, start_tw) + node_to_depot_dur <= depot_end_time
        # dur_matrix[:, 0, :] -> depot에서 노드까지 duration
        # dur_matrix[:, :, 0] -> 노드에서 depot까지 duration
        depot_to_node_dur = dur_matrix[:, 0, :]  # (B, N+1)
        node_to_depot_dur = dur_matrix[:, :, 0]  # (B, N+1)
        
        # tws[:, :, 0] -> start time window
        start_tw = tws[:, :, 0]  # (B, N+1)
        
        # 노드 도착 시간과 time window 시작 시간 중 최대값 + depot으로 돌아가는 시간
        service_start_time = torch.maximum(depot_to_node_dur, start_tw)  # (B, N+1)
        return_time = service_start_time + node_to_depot_dur  # (B, N+1)
        
        # 각 노드에 대해: max(depot_to_node_dur, start_tw) + node_to_depot_dur <= depot_end_time
        time_condition = (return_time <= depot_end_time).all(dim=1)  # (B,)
        
        # 두 조건 모두 만족해야 feasible
        feasible = dist_condition & time_condition
        
        return feasible

    def _generate_single_batch(self, batch_size_val: int) -> dict:
        """
        단일 배치 생성 (TensorDict로 변환하기 전의 raw 데이터)
        
        Returns:
            dict: 생성된 데이터 딕셔너리
        """
        batch_size = (batch_size_val,)
        
        # 1. Locations: (B, N+1, 2)
        locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2)) * self.max_loc
        
        # 2. Demands: (B, N)
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()
        
        # 3. Capacity: (B, 1)
        capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)
        
        # 4. Appear Times: (B, N+1) -> All 0 for now
        appear_times = torch.zeros((*batch_size, self.num_loc + 1), dtype=torch.float32)
        
        # 5. Distance Matrix: (B, N+1, N+1)
        dist_matrix = torch.cdist(locs, locs, p=2)
        
        # 6. Time Windows: (B, N+1, 2)
        tws = self._generate_time_windows_batch(batch_size_val, self.num_loc + 1, appear_times)
        
        # 7. Service Time: (B, N+1) -> All 0
        service_time = torch.zeros((*batch_size, self.num_loc + 1), dtype=torch.float32)
        
        # 8. Duration Matrix: (B, N+1, N+1)
        dur_matrix = self._generate_duration_matrix_batch(dist_matrix, current_time=0)
        
        # 9. Other fields
        batch_size_tensor = torch.Size([batch_size_val])
        open_route = torch.zeros((*batch_size_tensor, 1), dtype=torch.bool)
        distance_limit = torch.full((*batch_size_tensor, 1), float('inf'), dtype=torch.float32)
        backhaul_class = torch.ones((*batch_size_tensor, 1), dtype=torch.int32)
        speed = torch.ones((*batch_size_tensor, 1), dtype=torch.float32)
        
        return {
            "locs": locs,
            "demand": demand,
            "capacity": capacity,
            "appear_times": appear_times,
            "dist_matrix": dist_matrix,
            "tws": tws,
            "service_time": service_time,
            "dur_matrix": dur_matrix,
            "open_route": open_route,
            "distance_limit": distance_limit,
            "backhaul_class": backhaul_class,
            "speed": speed,
        }

    def _generate(self, batch_size, max_regenerate_iter: int = 100) -> TensorDict:
        batch_size_val = batch_size[0]
        
        # 초기 배치 생성
        data = self._generate_single_batch(batch_size_val)
        
        # Feasibility 체크
        feasible = self._check_feasibility(
            data["dist_matrix"], data["dur_matrix"], data["tws"]
        )
        
        # Infeasible한 인스턴스가 있으면 regenerate
        iter_count = 0
        while not feasible.all() and iter_count < max_regenerate_iter:
            num_infeasible = (~feasible).sum().item()
            infeasible_indices = torch.where(~feasible)[0]
            
            if iter_count == 0 or iter_count % 10 == 0:
                log.debug(f"Regenerating {num_infeasible} infeasible instances (iter {iter_count})")
            
            # Infeasible한 인스턴스 수만큼 새로 생성
            new_data = self._generate_single_batch(num_infeasible)
            
            # 새로 생성한 데이터의 feasibility 체크
            new_feasible = self._check_feasibility(
                new_data["dist_matrix"], new_data["dur_matrix"], new_data["tws"]
            )
            
            # Feasible한 새 인스턴스로 infeasible한 기존 인스턴스 교체
            new_feasible_indices = torch.where(new_feasible)[0]
            
            if len(new_feasible_indices) > 0:
                # 교체할 인스턴스 수 (새로 생성한 feasible 인스턴스 수와 기존 infeasible 인스턴스 수 중 작은 값)
                num_to_replace = min(len(new_feasible_indices), len(infeasible_indices))
                
                replace_target_indices = infeasible_indices[:num_to_replace]
                replace_source_indices = new_feasible_indices[:num_to_replace]
                
                # 각 텐서에 대해 교체
                for key in data.keys():
                    data[key][replace_target_indices] = new_data[key][replace_source_indices]
                
                # Feasibility 다시 체크
                feasible = self._check_feasibility(
                    data["dist_matrix"], data["dur_matrix"], data["tws"]
                )
            
            iter_count += 1
        
        if not feasible.all():
            log.warning(f"Could not make all instances feasible after {max_regenerate_iter} iterations. "
                       f"{(~feasible).sum().item()} instances remain infeasible.")
        
        # TensorDict 생성을 위한 후처리
        locs = data["locs"]
        demand = data["demand"]
        capacity = data["capacity"]
        tws = data["tws"]
        service_time = data["service_time"]
        dur_matrix = data["dur_matrix"]
        dist_matrix = data["dist_matrix"]
        open_route = data["open_route"]
        distance_limit = data["distance_limit"]
        backhaul_class = data["backhaul_class"]
        speed = data["speed"]
        appear_times = data["appear_times"]
        
        # Demand Linehaul/Backhaul
        demand_linehaul = demand  # (B, N)
        demand_backhaul = torch.zeros_like(demand_linehaul)  # (B, N)

        # Scaling
        if self.scale_demand:
            demand_linehaul = demand_linehaul / capacity
            demand_backhaul = demand_backhaul / capacity
            capacity = capacity / capacity

        return TensorDict(
            {
                "locs": locs,
                "demand_linehaul": demand_linehaul,
                "demand_backhaul": demand_backhaul,
                "vehicle_capacity": capacity,
                "capacity_original": capacity.clone(),
                "time_windows": tws,
                "service_time": service_time,
                "duration_matrix": dur_matrix,
                "distance_matrix": dist_matrix,
                "open_route": open_route,
                "distance_limit": distance_limit,
                "backhaul_class": backhaul_class,
                "speed": speed,
                "appear_times": appear_times,
            },
            batch_size=batch_size_val,
        )

    def _generate_time_windows_batch(self, batch_size: int, num_nodes: int, appear_times: torch.Tensor) -> torch.Tensor:
        """
        Generates time windows in batch.
        Replicates logic from vrp_bench/time_windows_generator.py
        """
        # Constants
        morning_peak = 8 * 60
        evening_peak = 19 * 60
        business_peak = 13 * 60
        delivery_day_start = 0
        delivery_day_end = 24 * 60
        
        # Sample customer types: 0 (residential) or 1 (commercial)
        # Shape: (B, N) - we generate for all nodes, then fix depot
        customer_type = torch.randint(0, 2, (batch_size, num_nodes), device=appear_times.device).float()
        
        # Residential (type 0)
        # 50% chance of morning vs evening
        is_morning = torch.rand(batch_size, num_nodes, device=appear_times.device) < 0.5
        
        start_time_res_morning = torch.normal(mean=torch.tensor(morning_peak, device=appear_times.device), std=torch.tensor(90, device=appear_times.device), size=(batch_size, num_nodes))
        start_time_res_evening = torch.normal(mean=torch.tensor(evening_peak, device=appear_times.device), std=torch.tensor(120, device=appear_times.device), size=(batch_size, num_nodes))
        start_time_res = torch.where(is_morning, start_time_res_morning, start_time_res_evening)
        
        window_length_res = torch.randint(1, 3, (batch_size, num_nodes), device=appear_times.device).float() * 60
        
        # Commercial (type 1)
        start_time_com = torch.normal(mean=torch.tensor(business_peak, device=appear_times.device), std=torch.tensor(90, device=appear_times.device), size=(batch_size, num_nodes))
        window_length_com = torch.randint(1, 2, (batch_size, num_nodes), device=appear_times.device).float() * 60
        
        # Combine based on type
        start_time = torch.where(customer_type == 0, start_time_res, start_time_com)
        window_length = torch.where(customer_type == 0, window_length_res, window_length_com)
        
        # Constraints
        # delivery_day_start = max(delivery_day_start, appear_time)
        # For now appear_times is 0, so max is just 0. If appear_times varies, we need:
        effective_start = torch.maximum(torch.tensor(delivery_day_start, device=appear_times.device), appear_times)
        
        # Ensure start time is within delivery hours
        # For residential: max(effective_start, min(start_time, delivery_day_end - window_length))
        # For commercial: max(effective_start, min(start_time, delivery_day_end - 60)) -> Wait, original code says 60 for commercial limit?
        # Original code:
        # Res: start_time = max(delivery_day_start, min(start_time, delivery_day_end - window_length))
        # Com: start_time = max(delivery_day_start, min(start_time, delivery_day_end - 60))
        
        limit_res = torch.tensor(delivery_day_end, device=appear_times.device) - window_length
        limit_com = torch.tensor(delivery_day_end - 60.0, device=appear_times.device)
        limit = torch.where(customer_type == 0, limit_res, limit_com)
        
        start_time = torch.minimum(start_time, limit)
        start_time = torch.maximum(effective_start, start_time)
        
        end_time = start_time + window_length
        end_time = torch.minimum(end_time, torch.tensor(delivery_day_end, device=end_time.device).float())
        
        # Stack
        tws = torch.stack([start_time, end_time], dim=-1) # (B, N, 2)
        
        # Fix Depot (Index 0) -> (0, 1440)
        tws[:, 0, 0] = 0
        tws[:, 0, 1] = 1440
        
        return tws

    def _generate_duration_matrix_batch(self, distance_matrix: torch.Tensor, current_time: float) -> torch.Tensor:
        """
        Generates duration matrix using torch (vectorized).
        Replicates logic from vrp_bench/travel_time_generator.py
        """
        # 1. Time Factor
        def normal_dist(x, mean, std):
            if isinstance(x, (int, float)):
                x = torch.tensor(x, dtype=torch.float32, device=distance_matrix.device)
            if isinstance(mean, (int, float)):
                mean = torch.tensor(mean, dtype=torch.float32, device=distance_matrix.device)
            if isinstance(std, (int, float)):
                std = torch.tensor(std, dtype=torch.float32, device=distance_matrix.device)
            return torch.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * torch.sqrt(torch.tensor(2 * np.pi, device=distance_matrix.device)))

        t = current_time
        # t is scalar 0 usually
        morning_peak = normal_dist(t, 480, 90)
        evening_peak = normal_dist(t, 1020, 90)
        time_fac = 0.5 + 2 * (morning_peak + evening_peak)
        
        # 2. Distance Factor
        dist_fac = 1 - torch.exp(-distance_matrix / 50.0)
        
        # 3. Base Delay
        base_delay = 0.25 * time_fac * dist_fac
        
        # 4. Random Factor
        rush_hour_effect = normal_dist(t, 480, 90) + normal_dist(t, 1020, 90)
        mu = 0 + 0.1 * rush_hour_effect
        sigma = 0.3 + 0.2 * rush_hour_effect
        
        # Sample lognormal
        # torch.distributions.LogNormal expects loc (mu) and scale (sigma) of the underlying normal distribution
        # numpy.random.lognormal same.
        # We need to sample for each element in distance_matrix
        rand_factor = torch.distributions.LogNormal(mu, sigma).sample(distance_matrix.shape).to(distance_matrix.device)
        
        delay = base_delay * rand_factor
        
        # 5. Accidents
        acc_rate = 0.05 * normal_dist(t, 1260, 120)
        acc_rate = max(0, acc_rate)
        
        # Poisson sample
        if acc_rate > 0:
            num_accidents = torch.distributions.Poisson(acc_rate).sample(distance_matrix.shape).to(distance_matrix.device)
        else:
            num_accidents = torch.zeros_like(distance_matrix)
            
        # Accident delay
        accident_delay = num_accidents * 75.0
        
        delay += accident_delay
        
        # Total Duration
        duration = distance_matrix + delay
        
        return torch.round(duration)
