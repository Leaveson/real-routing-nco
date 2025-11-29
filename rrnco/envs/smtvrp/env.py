import os
import sys
import math
import torch
import numpy as np
from typing import Optional, Union

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from .selectstartnodes import get_select_start_nodes_fn
from .generator import SMTVRPGenerator

log = get_pylogger(__name__)

class SMTVRPEnv(RL4COEnvBase):
    """
    SMTVRPEnv: Dynamic VRP Environment with Travel Time Rewards.
    Based on RMTVRPEnv but with dynamic duration updates and travel time objective.
    """
    name = "smtvrp"

    def __init__(
        self,
        generator: SMTVRPGenerator = None,
        generator_params: dict = {},
        normalize: bool = False,
        select_start_nodes_fn: Union[str, callable] = "all",
        check_solution: bool = False,
        **kwargs,
    ):
        super().__init__(check_solution=check_solution, **kwargs)
        if generator is None:
            if isinstance(generator_params, SMTVRPGenerator):
                generator = generator_params
            else:
                generator = SMTVRPGenerator(**generator_params)
        self.generator = generator
        if isinstance(select_start_nodes_fn, str):
            self.select_start_nodes_fn = get_select_start_nodes_fn(select_start_nodes_fn)
        else:
            self.select_start_nodes_fn = select_start_nodes_fn
        self.normalize = normalize
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        prev_node, curr_node = td["current_node"], td["action"]
        b_idx = torch.arange(td.batch_size[0], device=td.device)
        # print(td["action"])
        
        # Get duration for the taken action based on CURRENT duration matrix
        duration = td["duration_matrix"][b_idx, prev_node, curr_node]
        distance = td["distance_matrix"][b_idx, prev_node, curr_node]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        
        # Arrival time
        arrival_time = td["current_time"] + duration[:, None]
        # Start service at max(arrival, start_window)
        start_service_time = torch.max(arrival_time, start_times)
        # End service (ready to leave)
        end_service_time = start_service_time + service_time
        
        # If returning to depot (node 0), we don't necessarily have service time or wait time logic 
        # same as customers, but usually depot has wide TW. 
        # Logic:
        # If returning to depot, we should still update time to arrival time.
        # But usually in VRP, depot return ends the route.
        # However, for our penalty check, we need the time.
        
        # If curr_node != 0: use end_service_time
        # If curr_node == 0: use arrival_time (no service time at depot usually, or handled differently)
        
        # curr_time = torch.where(
        #     curr_node[:, None] != 0,
        #     end_service_time,
        #     arrival_time # At depot, time is arrival time
        # )
        # 
        curr_time = end_service_time * (curr_node[:, None] != 0)

        # Update current route length
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance[:, None]
        )

        # Update accumulated travel time (Reward Objective)
        # We want to minimize TOTAL travel time (duration).
        # This includes waiting time? 
        # "Travel Time(Duration) 기반의 보상 체계"
        # "모든 경로의 Travel Time(Duration) 합 최소화"
        # Usually this means completion time or sum of travel durations.
        # If it's "Travel Time", it might just be the sum of edge weights (durations).
        # If it includes waiting, it's "Makespan" or "Total Time".
        # Given "Travel Time", I will stick to sum of durations on edges.
        # But wait, if traffic is dynamic, waiting might be beneficial? 
        # No, usually VRP minimizes cost.
        # Let's accumulate the duration of the edge traversed.
        accumulated_travel_time = td.get("accumulated_travel_time", torch.zeros_like(curr_time)) + duration[:, None]

        # Dynamic Update of Duration Matrix
        # Update duration matrix based on NEW curr_time
        # We assume curr_time is a scalar per batch element (B, 1)
        # We need to update the whole (B, N, N) matrix.
        new_duration_matrix = self._update_duration_matrix(
            td["distance_matrix"], 
            curr_time.squeeze(-1) # (B,)
        )
        
        # Capacity updates (same as RMTVRP)
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        
        # Done
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        # log.info(f"DEBUG: visited={visited.sum(-1)}, done={done}")

        # Infeasible solution check:
        # If we are at depot (curr_node == 0) AND current_time > depot.late_tw AND not done
        # Then it means we returned to depot late (forced or chosen) without visiting all customers.
        # 1. Mark as done (visited = True for all)
        # 2. Set accumulated_travel_time += 10000
        
        depot_late_tw = td["time_windows"][..., 0, 1] # (B,)
        is_at_depot = (curr_node == 0)
        # Use curr_time (updated time) for check
        is_late = curr_time.squeeze(-1) > depot_late_tw
        
        # Apply penalty for already identified infeasible solutions (from action mask)
        is_infeasible_mask = td.get("is_infeasible", torch.zeros_like(done)).squeeze(-1)
        if is_infeasible_mask.any():
             accumulated_travel_time[is_infeasible_mask] += 10000.0
             visited[is_infeasible_mask] = True
             done[is_infeasible_mask] = True
        
        
        
        # Check for Depot -> Depot loop (stuck at depot)
        # If we are at depot and choose depot again, it means we can't go anywhere else.
        # We must terminate to prevent infinite loop.
        is_looping = (prev_node == 0) & (curr_node == 0) & ~done
        
        if is_looping.any():
            accumulated_travel_time[is_looping] += 10000.0
            visited[is_looping] = True
            done[is_looping] = True

        reward = torch.zeros_like(
            done
        ).float()  # we use the `get_reward` method to compute the reward
        
        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "accumulated_travel_time": accumulated_travel_time,
                "duration_matrix": new_duration_matrix, # UPDATED MATRIX
                "done": done,
                "reward": reward, # Step reward
                "used_capacity_linehaul": used_capacity_linehaul,
                "visited": visited,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        # print(f"DEBUG: Keys in _step return: {td.keys()}")
        return td

    def get_num_starts(self, td):
        return self.select_start_nodes_fn.get_num_starts(td)

    def select_start_nodes(self, td, num_starts):
        return self.select_start_nodes_fn(td, num_starts, self.get_num_starts(td))

    def _reset(self, td: Optional[TensorDict], batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        
        # Initialize static fields
        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]], dim=1
        )
        # No backhaul for now in generator, but keep structure
        demand_backhaul = torch.zeros_like(demand_linehaul)
        
        time_windows = td["time_windows"]
        service_time = td["service_time"]
        
        # Distance Matrix
        distance_matrix = td["distance_matrix"]
        
        # Initial Duration Matrix (already generated by generator, but let's ensure it matches time 0)
        # Generator generated it. We can use it.
        duration_matrix = td["duration_matrix"]
        
        # if self.normalize:
        #     duration_matrix = duration_matrix / duration_matrix.max()
        #     distance_matrix = distance_matrix / duration_matrix.max()
        

        # Reset dynamic fields
        current_time = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "distance_matrix": distance_matrix,
                "duration_matrix": duration_matrix,
                "demand_linehaul": demand_linehaul,
                "demand_backhaul": demand_backhaul,
                "vehicle_capacity": td["vehicle_capacity"],
                "time_windows": time_windows,
                "service_time": service_time,
                "current_node": torch.zeros((*batch_size,), dtype=torch.long, device=device),
                "current_route_length": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "current_time": current_time,
                "accumulated_travel_time": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "used_capacity_linehaul": torch.zeros((*batch_size, 1), device=device),
                "used_capacity_backhaul": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros((*batch_size, td["locs"].shape[-2]), dtype=torch.bool, device=device),
                "open_route": td["open_route"],
                "distance_limit": td["distance_limit"],
                "backhaul_class": td["backhaul_class"],
                "is_infeasible": torch.zeros((*batch_size, 1), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        # if self.normalize:
        #     td_reset.update(
        #         {
        #             "normal_factor_duration": torch.full(
        #                 (*batch_size, 1), 
        #                 duration_matrix.max(), 
        #                 dtype=torch.float32, 
        #                 device=device
        #             )
        #         }
        #     )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _update_duration_matrix(self, distance_matrix: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        """
        Updates the duration matrix based on current_time using logic ported from vrp_bench.
        distance_matrix: (B, N, N)
        current_time: (B,)
        Returns: (B, N, N)
        """
        with torch.no_grad():
            # Logic from travel_time_generator.py
            # 1. Time Factor
            
            def normal_dist_torch(x, mean, std):
                return torch.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))

            # current_time is (B,), broadcast to (B, N, N)
            t = current_time.view(-1, 1, 1)
            
            morning_peak = normal_dist_torch(t, 480, 90)
            evening_peak = normal_dist_torch(t, 1020, 90)
            time_fac = 0.5 + 2 * (morning_peak + evening_peak) # (B, 1, 1)
            
            # 2. Distance Factor
            # distance_factor = 1 - math.exp(-distance / 50)
            dist_fac = 1 - torch.exp(-distance_matrix / 50.0) # (B, N, N)
            
            # 3. Base Delay
            base_delay = 0.25 * time_fac * dist_fac # (B, N, N)
            
            # 4. Random Factor
            # rush_hour_effect = normal(t, 480, 90) + normal(t, 1020, 90)
            rush_hour_effect = normal_dist_torch(t, 480, 90) + normal_dist_torch(t, 1020, 90)
            mu = 0 + 0.1 * rush_hour_effect
            sigma = 0.3 + 0.2 * rush_hour_effect
            
            # Sample lognormal
            # We need to sample for each edge? 
            # "random.lognormvariate" is called per edge in original code.
            # So we sample (B, N, N)
            rand_factor = torch.distributions.LogNormal(mu.expand_as(distance_matrix), sigma.expand_as(distance_matrix)).sample()
            
            delay = base_delay * rand_factor
            
            # 5. Accidents
            # accident_rate = 0.05 * normal(t, 1260, 120)
            acc_rate = 0.05 * normal_dist_torch(t, 1260, 120)
            acc_rate = torch.clamp(acc_rate, min=0)
            
            # Poisson sample
            # num_accidents = poisson(acc_rate)
            num_accidents = torch.poisson(acc_rate.expand_as(distance_matrix))
            
            # Accident delay
            # Let's use: accident_delay = num_accidents * 75.0
            accident_delay = num_accidents * 75.0
            
            delay += accident_delay
            
            # Total Duration
            # distance / velocity + delay. Velocity = 1.
            duration = distance_matrix + delay
            
            # Debug logging for batch size
            # log.info(f"DEBUG: distance_matrix shape: {distance_matrix.shape}")
            
            # Optimize return: round in place if possible, but we need new tensor for float
            # duration.int().float() creates 2 copies.
            # duration.floor_() is in-place.
            duration = (distance_matrix + delay).detach()
            duration.floor_()
            return duration

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Reward is negative total travel time
        # We can compute it from the history of durations or just sum the step rewards.
        # RL4CO calls _get_reward at the end.
        # If we used step rewards, we might not need this if the algorithm uses step rewards.
        # But usually _get_reward returns the final return.
        # We tracked `accumulated_travel_time`.
        return -td["accumulated_travel_time"].squeeze(-1)

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # Reuse RMTVRP logic or simplify
        # We need to check Time Windows, Capacity, etc.
        # Since we inherit from RL4COEnvBase, we need to implement this.
        # I can copy `RMTVRPEnv.get_action_mask` but adapt for our state.
        
        curr_node = td["current_node"]
        b_idx = torch.arange(td.batch_size[0], device=td.device)
        
        # Current duration to next node
        dur_ij = td["duration_matrix"][b_idx, curr_node, :]
        dur_j0 = td["duration_matrix"][:, :, 0] # From next to depot (approximation using current time)
        
        # Time Windows
        early_tw = td["time_windows"][..., 0]
        late_tw = td["time_windows"][..., 1]
        
        arrival_time = td["current_time"] + dur_ij
        can_reach_customer = arrival_time < late_tw
        
        # Return to depot check (if not open route)
        # We use current duration estimate for return
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + dur_j0
        ) * ~td["open_route"] < late_tw[..., 0:1]
        
        # Capacity
        exceeds_cap = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        
        # Visited
        visited = td["visited"]
        
        # Combine
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & ~exceeds_cap
            & ~visited
        )
        
        # Mask depot
        # Can visit depot if coming from depot? No.
        # Can visit depot if all customers visited? Yes (handled by done).
        # Standard VRP: can visit depot to end route?
        # If open route, we don't need to return.
        # If closed route, we must return.
        # Usually depot is masked until all customers visited?
        # Or we can return to refill? (Multi-trip).
        # RMTVRP seems to be Single-trip per vehicle?
        # "Done when all customers are visited".
        # RMTVRPEnv: "Done when all customers are visited".
        # So we mask depot unless we are done?
        # Actually RMTVRPEnv logic:
        # can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))
        # If at depot and there are customers left, cannot visit depot (must go out).
        # If at customer, can we go to depot?
        # If we go to depot, is it done?
        # RMTVRPEnv seems to imply we just visit nodes.
        # If we return to depot, does it end?
        # RMTVRPEnv: "Done when all customers are visited".
        # So we can't end at depot if customers remain.
        # So depot is only valid if all customers visited?
        # But RMTVRPEnv allows returning to depot?
        # "Vehicles are not required to return to the depot after serving all customers" (Open Route).
        # If not open route, we must return.
        # But the action mask logic in RMTVRPEnv seems to allow depot visit only if...
        # Actually RMTVRPEnv mask logic for depot is:
        # can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))
        # This only prevents Depot -> Depot if customers exist.
        # It doesn't prevent Customer -> Depot if customers exist.
        # If Customer -> Depot is allowed, it implies Multi-Trip or just ending early?
        # But `done` is only when `visited.sum() == all`.
        # So if we go to depot early, we get stuck?
        # Let's assume standard CVRP where we visit all nodes.
        # I will use the same logic: prevent staying at depot if work remains.
        
        can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))
        
        # Relax constraints for already late (infeasible) agents
        # If we are already late (current_time > depot_late_tw), we allow visiting any unvisited node
        # (subject to capacity). We ignore TWs because we are already penalized.
        
        is_already_late = td["current_time"] > late_tw[..., 0:1] # Check against depot late TW
        td.update({"is_infeasible": is_already_late})

        if is_already_late.any():
            # Relaxed mask: just capacity and visited
            can_visit_relaxed = (
                ~exceeds_cap
                & ~visited
            )
            # Depot rule for relaxed: can always go to depot? 
            # Or same rule: don't go to depot if customers remain.
            can_visit_relaxed[:, 0] = ~((td["current_node"] == 0) & (can_visit_relaxed[:, 1:].sum(-1) > 0))
            
            # Apply relaxed mask where is_already_late is True
            # is_already_late is (B, 1). Broadcast to (B, N+1).
            mask_late = is_already_late.expand_as(can_visit)
            can_visit = torch.where(mask_late, can_visit_relaxed, can_visit)
            
        
        # Infeasible check in mask:
        # If current_time + duration_to_depot > depot.late_tw, we are already late to return.
        # We MUST return to depot to end the episode (and take penalty).
        # So we mask everything except depot.
        
        # Duration from current to depot
        dur_to_depot = td["duration_matrix"][b_idx, curr_node, 0] # (B,)
        arrival_at_depot = td["current_time"].squeeze(-1) + dur_to_depot
        
        depot_late_tw = td["time_windows"][..., 0, 1] # (B,)
        
        # Check if we are late to return
        # Note: If we are already at depot, dur_to_depot is 0. 
        # If we are at depot and late, we are handled by _step penalty.
        # Here we handle being at a customer and realizing we can't make it back.
        will_be_late = arrival_at_depot > depot_late_tw
        
        # If will_be_late, force depot visit (if not already done)
        # If done, we don't care (can_visit is all False or handled).
        # If not done and will be late:
        # IMPORTANT: Only force depot if we are NOT at depot.
        # If we are at depot, we must be allowed to leave (even if late) to continue visiting,
        # otherwise we are stuck (since depot->depot is masked).
        mask_force_depot = will_be_late & (visited.sum(-1) < visited.size(-1)) & (curr_node != 0)
        
        if mask_force_depot.any():
            # For these batches, mask all customers
            can_visit[mask_force_depot, 1:] = False
            # Ensure depot is unmasked (it might have been masked by the "don't return to depot" rule)
            can_visit[mask_force_depot, 0] = True
        
        # Also, if we are at customer, can we go to depot?
        # If we go to depot, we are just visiting node 0.
        # If we haven't visited all, we shouldn't go to depot if it ends the episode?
        # But RL4CO usually handles "depot as refill" or "depot as end".
        # If capacity is high enough (which it is in generator), it's single trip.
        # So we should force visiting all customers.
        # So depot should be masked if customers remain?
        # RMTVRPEnv doesn't explicitly mask depot if customers remain, except the "Depot->Depot" check.
        # But if we go Customer -> Depot, and not done, what happens?
        # We are at depot. Then we can go to other customers?
        # Yes, if it's multi-trip.
        # But our capacity is generated to cover all demands?
        # "Capacity should be 30 + num_loc/5 ... Note that this serves as a demand scaler."
        # It seems capacity is tight?
        # If capacity is tight, we might need multi-trip.
        # So Customer -> Depot -> Customer is allowed.
        # So I will keep the logic: allow depot visit.
        
        return can_visit

    def _make_spec(self, generator: SMTVRPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
                device=self.device,
            ),
            demand_linehaul=BoundedTensorSpec(
                low=-float('inf'),
                high=float('inf'),
                shape=(generator.num_loc, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            current_time=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            ),
            duration_matrix=UnboundedContinuousTensorSpec(
                shape=(generator.num_loc + 1, generator.num_loc + 1),
                dtype=torch.float32,
                device=self.device,
            ),
            accumulated_travel_time=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=generator.num_loc + 1,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, device=self.device
        )
        self.done_spec = UnboundedDiscreteTensorSpec(
            shape=(1,), dtype=torch.bool, device=self.device
        )
