"""A=2 Carry + teammate 토큰 하나. adapt 스캐폴드를 그대로 쓴다.

**왜 HumanoidAdaptCarryGround2Terrain 을 상속하지 않는가**: 그 파일은 stage1 코드의
*복사본* 이라 A=2 이식이 안 돼 있고 (같은 종류의 수정이 75곳), 게다가 우리에게 필요
없는 지형 기계가 절반이다. 반면 stage1 클래스 HumanoidTrajSitCarryClimb 는 A=2 이식이
**끝났고 검증됐다** (A=2 성공률 0.9645 = A=1 0.9648). 그래서 그쪽을 상속하고 adapt 가
요구하는 것만 덮어쓴다.

adapt 빌더(amp_network_builder_transformer_adapt.py)와의 계약:
  - each_subtask_name 은 ["new_extra", "new_carry", "old_carry"] — **new 가 앞**이어야
    한다. 빌더가 task_obs_each_size[how_many_new_tasks:] 로 old 쪽만 골라 로드한다.
  - old_carry 의 크기는 stage1 의 carry 블록과 **같아야**(42) 사전학습 토크나이저가
    크기 매칭으로 로드된다.
  - major_task_name="carry", has_extra=True

그래서 관측은 [self(223)] + [teammate(21)] + [carry(42)] + [carry(42)] 가 된다.
carry 블록을 두 번 내보내는 건 adapt 의 설계 그대로다 — old 는 얼린 사전학습 토큰,
new 는 같은 입력을 보는 학습 가능한 토큰이다.
"""

import os
import torch

from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from utils import torch_utils
from isaacgym.torch_utils import *


# stage1 의 task obs 안에서 carry 블록이 차지하는 구간. traj(20) + sit(38) 다음이다.
CARRY_LO, CARRY_HI = 58, 100
TEAMMATE_DIM = 21          # 상대 1명당. 아래 _teammate_obs 의 주석 참조


class HumanoidMACarry(HumanoidTrajSitCarryClimb):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # MA_TOKEN=zero 면 토큰 자리는 그대로 두고 값만 0 으로 채운다. obs 크기가 같아야
        # zero 와 live 가 **같은 네트워크**로 비교된다 -- 크기가 달라지면 그 비교는
        # "토큰 내용" 이 아니라 "네트워크 구조" 를 비교하는 게 되어 무의미해진다.
        # MA_TOKEN=mask 는 live 21-D 관측을 그대로 계산하되 adapt 빌더가 teammate
        # 토큰을 attention key/value 에서 제외한다. zero 토큰도 softmax 분모를 바꾸므로
        # "토큰이 아예 없음" 을 재려면 이 모드를 써야 한다.
        # 입력축 ablation. **차원 수(21)는 항상 같고 내용만 0 으로 덮는다** --
        # 크기가 바뀌면 토크나이저가 달라져서 "입력 내용"이 아니라 "네트워크 구조"를
        # 비교하게 된다. zero/live 를 같은 크기로 둔 것과 같은 이유다.
        #   zero  전부 0                                   (0 차원 유효)
        #   r12   상대 root pos·vel·heading 까지            (12)
        #   r18   + 상대 박스 pos·vel                       (18)
        #   live  + 상대 박스 목표 = 전체                    (21)
        #   mask  live 와 같은 값, actor attention 에서는 완전 제외 (21)
        self._token_mode = os.environ.get("MA_TOKEN", "live")
        assert self._token_mode in ("live", "zero", "r12", "r18", "mask"), self._token_mode
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

    def get_task_obs_size(self):
        # 부모의 get_obs_size 가 self obs + 이 값으로 계산한다. adapt 는 task mask 를
        # 안 쓰므로(mask 는 4개 태스크를 가릴 때 쓰는 것) 여기에 더하지 않는다.
        return TEAMMATE_DIM * (self.num_agents - 1) + 2 * (CARRY_HI - CARRY_LO)

    def get_multi_task_info(self):
        # 이 함수는 두 소비자에게 서로 다른 답을 줘야 한다.
        #   - 네트워크(adapt 빌더): [new_extra, new_carry, old_carry] 3 토큰
        #   - 부모의 관측 계산: 원래의 4 태스크(traj/sit/carry/climb) 마스크
        # 부모는 자기 _task_mask(rows, 4) 와 이 마스크 행렬을 곱하므로 3 을 주면 터진다.
        if getattr(self, "_use_base_info", False):
            return HumanoidTrajSitCarryClimb.get_multi_task_info(self)

        each = [TEAMMATE_DIM * (self.num_agents - 1),
                CARRY_HI - CARRY_LO,
                CARRY_HI - CARRY_LO]
        names = ["new_extra", "new_carry", "old_carry"]
        n = len(each)
        mask = torch.zeros(n, sum(each), dtype=torch.bool, device=self.device)
        index = torch.cumsum(torch.tensor([0] + each), dim=0).to(self.device)
        for i in range(n):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": n,
            "tota_subtask_obs_size": sum(each),
            "each_subtask_obs_size": each,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": False,
            "each_subtask_name": names,
            "major_task_name": "carry",
            "has_extra": True,
        }

    def _teammate_obs(self, rows):
        """상대 1명당 21 차원. ego root yaw 프레임, 10 m 클립.

            상대 root pos(3) + vel(3)         반응적 회피
            상대 heading tan-norm(6)          cos/sin 이 아니라 6d -- TokenHSI 관례
            상대 box pos(3) + vel(3)          상자는 독립적으로 길을 막는다
            상대 box target(3)                예측

        **파지 여부는 넣지 않는다.** 상대 box 위치에 이미 들어 있고, 위상을 따로 주입하면
        기준점이 튄다 (steer 에서 기준점 전환이 0.393 m 점프 -> 이탈 78 % 급등을 만들었다).

        프레임은 **파지 여부와 무관하게 항상 ego root yaw** 다. 거리는 10 m 로 클립해서
        먼 값이 토크나이저를 지배하지 못하게 한다.
        """
        A = self.num_agents
        if A == 1:
            return torch.zeros((len(rows), 0), device=self.device)

        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        me = rows
        e, a = torch.div(rows, A, rounding_mode="floor"), rows % A

        my_pos = h[me, 0:3]
        heading_inv = torch_utils.calc_heading_quat_inv(h[me, 3:7])

        out = []
        for k in range(1, A):
            other = e * A + (a + k) % A          # 상대적 순번이라 A 가 커져도 그대로 쓴다
            def local(v, clip=True):
                d = quat_rotate(heading_inv, v)
                return torch.clamp(d, -10.0, 10.0) if clip else d
            o_rot = h[other, 3:7]
            out.append(torch.cat([
                local(h[other, 0:3] - my_pos),                       # 3
                local(h[other, 7:10], clip=False),                   # 3
                torch_utils.quat_to_tan_norm(quat_mul(heading_inv, o_rot)),  # 6
                local(b[other, 0:3] - my_pos),                       # 3
                local(b[other, 7:10], clip=False),                   # 3
                local(self._box_tar_pos[other, 0:3] - my_pos),       # 3
            ], dim=-1))
        obs = torch.cat(out, dim=-1)
        keep = {
            "zero": 0, "r12": 12, "r18": 18,
            "live": TEAMMATE_DIM, "mask": TEAMMATE_DIM,
        }[self._token_mode]
        if keep < TEAMMATE_DIM:
            # 상대 1명당 블록 안에서 뒤쪽을 0 으로 덮는다 (A>=3 이어도 블록마다 적용)
            v = obs.view(obs.shape[0], -1, TEAMMATE_DIM).clone()
            v[:, :, keep:] = 0.0
            obs = v.view(obs.shape[0], -1)
        return obs

    def _compute_task_obs(self, env_ids=None):
        # 부모가 4개 블록을 다 계산한다. 그중 carry 만 떼어내 두 번 쓴다.
        # A>=2 는 carry 전용이라(_task_init_prob 고정) 마스크가 carry 를 그대로 통과시킨다.
        self._use_base_info = True
        try:
            parent = super()._compute_task_obs(env_ids)
        finally:
            self._use_base_info = False
        carry = parent[:, CARRY_LO:CARRY_HI]
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        return torch.cat([self._teammate_obs(rows), carry, carry], dim=-1)
