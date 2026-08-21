"""A=2 Carry + teammate 토큰 + steering 창. ma 와 steer 를 합친 것.

**상속 구조**: `HumanoidMACarry` (A=2 이식 완료 + teammate 토큰) 를 상속하고
steer 의 창·보상만 얹는다. steer 원본(`humanoid_steer_carry.py`, 1311 줄)은
`HumanoidAdaptCarryGround2Terrain` 을 상속해 A=1 전용이고 지형 기계가 절반이라
그대로 옮기지 않는다. 필요한 것은 경로 수학(`steer_path.py`, 배치 무관)과
창·보상 두 덩어리뿐이다.

**속도 명령 = 창의 길이** (docs/HANDOFF_steer_time.md):

    창    = path[s + (M/6)·i],  i=1..6      s = 단조 호길이 래칫
    v_tar = M / 1.6 s                       원본의 1.5 상수를 대체
    M=2.4 -> 1.5 m/s,  M=1.2 -> 0.75,  M=0 -> 정지

공간 지평이 아니라 **시간 지평(1.6 초)이 고정**된다. 창이 내 투영 기준이라
자기교정이 남고 시계·재anchor·집기핀 같은 장치가 필요 없다.

**월드모델 입장에서는 등간격 6점을 주는 것이 전부다.** 간격을 줄이면 느려진다.

관측 = [self 223] [teammate 21] [steer 12] [new_carry 42] [old_carry 42] = 340
"""

import os
import numpy as np
import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_carry import (
    HumanoidMACarry, CARRY_LO, CARRY_HI, TEAMMATE_DIM,
)
from tokenhsi.utils import steer_path as sp
from isaacgym.torch_utils import *


MS_L_DEFAULT = 12.0      # 시나리오 경로 길이. 창 계산과 배치가 같이 쓴다


def _f(name, default):
    return float(os.environ.get(name, default))


class HumanoidMASteerCarry(HumanoidMACarry):

    # ---- 시나리오 -----------------------------------------------------------
    #
    #   MS_SCEN = free | parallel | cross | solo
    #
    #     free      지금 방식. 각자 랜덤 경로라 겹칠지 말지가 운이다
    #     parallel  나란히 MS_GAP 간격, 둘 다 평속. 반응만으로 되는 한계
    #     cross     직각 교차. a1 만 교차점 앞 MS_W 에서 감속해 MS_DT 초 늦게 온다
    #     solo      cross 와 **같은 감속 명령**, 짝은 MS_SEP 밖. 감속 실행 능력만
    #
    # 시나리오는 **직선 경로**를 쓴다. 곡선이면 교차점까지 호길이가 4.30 m 대 4.60 m
    # 로 어긋나고 경로가 원점을 1.86 m 빗나가 통제가 안 된다 (실측).

    def _scen_env(self):
        """시나리오 하나가 배치 환경변수 여러 개를 일관되게 정한다.

        따로 주게 두면 조합이 어긋난다 -- 특히 MA_SPAWN_GAP(기본 1.0 m)이 켜져 있으면
        apply_layout **뒤에** 돌면서 parallel 0.5 m 를 통째로 망가뜨린다.
        """
        sc = os.environ.get("MS_SCEN", "free")
        if sc == "free":
            return sc
        e = os.environ
        want = {"parallel": "parallel", "cross": "side", "solo": "far"}.get(sc)
        if want is None:
            raise ValueError(f"unknown MS_SCEN={sc}")
        # **setdefault 면 안 된다.** 평가는 사이드카를 source 한 뒤 python 을 띄우므로
        # 낡은 MA_LAYOUT 을 물려받는 게 정상 경로다. 그러면 기하가 조용히 바뀌는데
        # _scen_xp(env 원점)와 _win_hi(L/2)는 없어진 교차점 기준으로 계속 계산되고
        # 아무것도 죽지 않은 채 모든 숫자가 그럴듯하게 나온다.
        old = e.get("MA_LAYOUT")
        if old not in (None, "", want):
            raise ValueError(f"MS_SCEN={sc} 는 MA_LAYOUT={want} 인데 {old} 가 이미 있다")
        e["MA_LAYOUT"] = want
        if sc == "parallel":
            e["MA_LAYOUT_D"] = str(_f("MS_GAP", 1.0) / 2.0)   # 간격 = 2d
        elif sc == "solo":
            e["MA_LAYOUT_S"] = str(_f("MS_SEP", 9.0))
        # **기본값을 두 곳에 쓰면 안 된다.** 여기가 9.0, 창 계산이 12.0 이면
        # 실제 경로는 9 m 인데 교차셀은 6.0 m 로 잡혀 회복 구간이 통째로 사라진다.
        e["MA_LAYOUT_L"] = str(_f("MS_L", MS_L_DEFAULT))
        # 직선이어야 교차점까지 호길이가 양쪽 같아지고, 그래야 dt=0 이 M 조작 없는
        # 순수 대조군이 된다. MS_SCEN_CURVE=1 은 그걸 포기하고 **휘어진 경로**로 둔다 --
        # 보기용이자 강건성 확인용이다. 호길이가 어긋나므로(실측 4.30 대 4.60 m)
        # 명령한 지연과 실제 도착 시간차가 정확히 안 맞는다. 판정에는 쓰지 않는다.
        if not int(_f("MS_SCEN_CURVE", 0)):
            e["MS_LAT_MAX"] = "0.0"
        e["MS_MRAND"] = "0"          # 랜덤 M 이 시나리오 프로파일을 덮어쓴다
        e["MA_SPAWN_GAP"] = "0"      # apply_layout 뒤에 돌며 배치를 흔든다
        e["MA_SEP"] = "0"
        return sc

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.scen = self._scen_env()
        self.scen_L = _f("MS_L", MS_L_DEFAULT)
        self.scen_w = _f("MS_W", 3.0)          # 감속 구간 길이 (m)
        # 감속 구간과 교차점 사이의 **평속 회복 구간**. 이게 0 이면 감속한 쪽이
        # 느린 채로 교차점을 지나 교차 구역 체류시간이 2 배가 된다. 그러면 "늦게
        # 왔다" 와 "느리게 지났다" 가 같은 조작이 되어 여유거리를 해석할 수 없다.
        self.scen_recov = _f("MS_RECOV", 1.5)
        self.scen_decel = os.environ.get("MS_DECEL", "a1")     # a1 | both
        self.scen_placebo = int(_f("MS_PLACEBO", 0))           # 창을 교차점 뒤로
        self.scen_dt = _f("MS_DT", 0.0)        # a1 을 몇 초 늦출까. 0 이면 sync
        self.scen_encr = 2.5                # 조우 반경 (지표 임계)
        self.steer_k = 6                    # 창의 점 개수. **관측 크기 = 2K** 라 env 로 두면 안 된다
        self.steer_m_nom = 2.4              # 2.4 m = 1.5 m/s. 속도 눈금의 정의 그 자체
        self.steer_m_lo = _f("MS_M_LO", 0.25)           # 배수 하한
        self.steer_mrand = int(_f("MS_MRAND", 0))       # 0 이면 M 고정 = 원본과 동일
        self.steer_pos_c = _f("MS_POS_C", 2.0)          # latpen 계수
        # 속도 보상과 이탈 벌점의 **저울**. 기본값에서 스텝당 최대 기여가
        #   속도  2*(0.2) + 2*(0.2) = 0.8
        #   이탈  2*(2.0) + 2*(2.0) = 8.0     <- 10 배
        # 라서 최적 정책이 "천천히 가고 정확히 따라가기" 가 된다. 실측이 정확히 그렇다
        # (lat 0.109 / v_real 0.58, 참조 모션 중앙값 0.932 의 13 백분위).
        self.steer_vel_w = _f("MS_VEL_W", 1.0)         # 속도항 배수
        # exp(-k*dv^2) 의 k. 기본 5 면 sigma=0.32 m/s 라, 명령 1.5 에 실제 0.58 이면
        # 2.9 sigma 밖이고 보상이 0.0145 -- 올라갈 언덕이 안 느껴지는 평지다.
        self.steer_vel_k = _f("MS_VEL_K", 5.0)
        # 목표 도달 뒤에도 창은 계속 전진을 명령하는데 정답은 정지다 (maxIETSteps=60).
        # 매 성공 에피소드마다 "명령을 무시하라" 를 60 스텝씩 가르치는 셈이라,
        # 경로 끝 이후의 M 을 0(=정지 명령)으로 만드는 대조군을 둔다.
        self.steer_endclamp = int(_f("MS_ENDCLAMP", 0))
        # MS_CLIP=1 : 호길이·조준점·창을 **실제 경로 끝에서 포화**시킨다.
        # TokenHSI traj 의 clip(t/dur, 0, 1) 과 같은 보호를 거리 버전으로 넣는 것이다.
        # 원본 carry 는 명령이 절대 위치(목표)라 도착하면 저절로 "여기 있어라" 가 되는데,
        # steer 는 "내 앞쪽" 이라 끝을 넘어 가짜 구간을 가리킨다.
        # ENDCLAMP 이 M 만 막는 우회책이라면 이쪽이 조준점·이탈거리까지 정상화한다.
        self.steer_clip = int(_f("MS_CLIP", 0))
        self.steer_back = 0.5               # 래칫 후진 허용
        self.steer_pin = _f("MS_PIN", 0.5)
        self.steer_zero = bool(int(os.environ.get("MS_ZERO", "0")))
        self.steer_seed = int(_f("MS_SEED", 0))
        self._steer_tick = 0
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

    # ---- 관측 크기 ---------------------------------------------------------

    def steer_dim(self):
        return 2 * self.steer_k

    def get_task_obs_size(self):
        return (TEAMMATE_DIM * (self.num_agents - 1) + self.steer_dim()
                + 2 * (CARRY_HI - CARRY_LO))

    def get_multi_task_info(self):
        # 부모의 관측 계산 중에는 원래 4 태스크 마스크를 그대로 돌려줘야 한다
        # (부모가 자기 _task_mask(rows,4) 와 곱한다).
        if getattr(self, "_use_base_info", False):
            return super().get_multi_task_info()
        each = [TEAMMATE_DIM * (self.num_agents - 1),
                self.steer_dim(),
                CARRY_HI - CARRY_LO,
                CARRY_HI - CARRY_LO]
        # **new 가 앞이어야 한다.** 빌더가 task_obs_each_size[how_many_new:] 로
        # old 쪽만 골라 사전학습 토크나이저를 싣는다.
        names = ["new_extra_0", "new_extra_1", "new_carry", "old_carry"]
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

    # ---- 경로 상태 ---------------------------------------------------------

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        R = self._rows
        # **전부 row 당 하나다.** 사람마다 자기 경로를 받는다.
        self._gt_path = torch.zeros(R, sp.V, 2, device=self.device)
        self._arc_root = torch.zeros(R, device=self.device)
        self._arc_box = torch.zeros(R, device=self.device)
        self._lat_box = torch.zeros(R, device=self.device)
        self._mscale = torch.ones(R, sp.V, device=self.device)
        self._prev_arc = torch.zeros(R, device=self.device)
        self._scen_xp = torch.zeros(R, 2, device=self.device)   # 교차점 (world)
        # 실제 경로가 끝나는 호길이. 뒤는 직선 외삽이라 여기서 막아야 한다.
        self._s_end = torch.full((R,), sp.V * sp.DS, device=self.device)
        # 시나리오 지표. 전부 row 당 하나이고 스텝마다 누적, 에피소드 끝에 덤프한다.
        self._ep_xtime = torch.full((R,), -1, device=self.device, dtype=torch.long)
        self._ep_xdist = torch.full((R,), 1e6, device=self.device)
        self._ep_encd = torch.full((R,), 1e6, device=self.device)
        self._ep_wn = torch.zeros(R, device=self.device, dtype=torch.long)
        self._ep_wv = torch.zeros(R, device=self.device)
        self._ep_wc = torch.zeros(R, device=self.device)
        self._ep_dtcmd = torch.zeros(R, device=self.device)     # 이 에피소드에 명령한 지연
        self._ep_wdone = torch.zeros(R, device=self.device, dtype=torch.long)  # 창을 끝까지 지났나
        self._ep_wnbox = torch.zeros(R, device=self.device, dtype=torch.long)
        self._ep_latw = torch.zeros(R, device=self.device)      # 창 안 |lat| 합 -- 우회 반증용
        self._ep_encn = torch.zeros(R, device=self.device, dtype=torch.long)
        self._lat_root = torch.zeros(R, device=self.device)
        # **시나리오와 무관하게 늘 쌓는다.** free 기준선(reg/zero/m4/m8)이야말로
        # "경로를 따라갔나 · 명령 속도를 지켰나" 를 물어야 하는데, 성공률만으로는
        # 경로 추종과 목표로 직진하는 것을 구분할 수 없다.
        self._ep_latr = torch.zeros(R, device=self.device)      # |lat_root| 합
        self._ep_latb = torch.zeros(R, device=self.device)      # |lat_box| 합
        self._ep_spd = torch.zeros(R, device=self.device)       # |v_real - v_cmd| 합
        self._ep_vr = torch.zeros(R, device=self.device)        # v_real 합
        self._ep_steps = torch.zeros(R, device=self.device, dtype=torch.long)
        # 명령 속도별 실제 속도. **절대오차 합만으로는 "전 구간에서 느림" 과
        # "낮은 명령은 따르고 높은 것만 못 따름" 이 구분되지 않는다.** 그 구분이
        # steer 주장의 핵심이라 명령 구간별로 갈라 쌓는다. 경계는 MRAND 의
        # 배수 {.25,.5,.75,1.0} -> v_cmd {0.375,0.75,1.125,1.5} 사이를 가른다.
        self._VBINS = torch.tensor([0.5625, 0.9375, 1.3125], device=self.device)
        self._ep_vbin_s = torch.zeros(R, 4, device=self.device)   # 구간별 v_real 합
        self._ep_vbin_n = torch.zeros(R, 4, device=self.device)   # 구간별 스텝 수
        # 감속 창을 **셀 인덱스로** 잡는다. _m_at 이 0.1 m 셀 단위 계단이라
        # 미터로 자르면 실효 W 가 격자만큼 어긋나 오프셋이 3% 빗나간다.
        self._x_cell = int(self.scen_L / 2.0 / sp.DS)          # 교차점이 든 셀
        self._win_hi = self._x_cell - int(self.scen_recov / sp.DS)
        self._win_lo = max(self._win_hi - int(self.scen_w / sp.DS), 0)
        if self.scen_placebo:                                  # 교차점 **뒤**로 옮긴다
            self._win_lo = self._x_cell + int(self.scen_recov / sp.DS)
            self._win_hi = self._win_lo + int(self.scen_w / sp.DS)
        self._w_eff = (self._win_hi - self._win_lo) * sp.DS
        return

    def _decel_mult(self, dt=None):
        """구간 W_eff 에서만 감속해 dt 초 늦게 도착하는 M 배수.

            v_slow = W / (W/v_nom + dt)      구간 통과 시간이 정확히 dt 만큼 는다

        dt 가 텐서면 배수도 텐서로 돌려준다 (에피소드마다 다른 지연).
        """
        if dt is None:
            dt = self.scen_dt
        v_nom = self.steer_m_nom / 1.6
        v_slow = self._w_eff / (self._w_eff / v_nom + dt)
        mult = v_slow * 1.6 / self.steer_m_nom
        if torch.is_tensor(mult):
            return mult.clamp(min=self.steer_m_lo, max=1.0)
        return min(max(mult, self.steer_m_lo), 1.0)

    def _sample_dt(self, n, seed):
        """MS_DT_RAND=1 이면 에피소드마다 지연을 새로 뽑는다.

        **고정 dt 로 학습하면 안 된다.** M 프로파일이 매 에피소드 똑같아서 정책이
        "호길이 1.5~4.5 에서 느려져라" 를 창을 읽지 않고 외워버린다. 그러면
        "타이밍이 된다" 는 결론이 공허해진다 -- 창을 0 으로 해도 같은 값이 나온다.
        지연을 흔들면 얼마나 늦출지는 창에서만 알 수 있다.
        """
        if not int(os.environ.get("MS_DT_RAND", 0)):
            return torch.full((n,), self.scen_dt, device=self.device)
        lv = torch.tensor([float(x) for x in
                           os.environ.get("MS_DT_SET", "0,0.5,1.0,1.5,2.0").split(",")],
                          device=self.device)
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        return lv[torch.randint(len(lv), (n,), generator=g).to(self.device)]

    def _post_object_reset(self, env_ids):
        """부모가 물체를 시뮬에 반영한 뒤, 관측을 만들기 전에 부른다.

        **_reset_envs 뒤에 하면 안 된다** -- 그 시점엔 부모가 이미
        _compute_observations 를 돌려서 첫 관측이 옛 경로로 계산되고,
        그게 _init_amp_obs 까지 흘러간다 (실측: 0.86 -> 0.43).
        """
        super()._post_object_reset(env_ids)
        if len(env_ids) > 0:
            self._reset_steer(self.agent_rows(env_ids))
        return

    def _reset_steer(self, rows):
        """row 별로 사람->박스->목표 경로를 그린다."""
        if len(rows) == 0:
            return
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        root = h[rows, 0:2]
        box = b[rows, 0:2]
        tar = self._box_tar_pos[rows, 0:2]
        self._steer_tick += 1
        seed = self.steer_seed + self._steer_tick + int(rows[0])
        path, s_box, n_end = sp.gen_full_v2(root, box, tar, seed,
                                     0.0, _f("MS_LAT_MAX", 2.2),
                                     120.0,
                                     p_two=0.1,
                                     skew=0.8,
                                     spread=(0.85, 1.8),
                                     lat_frac=0.25, with_end=True)
        self._gt_path[rows] = path
        self._s_end[rows] = (n_end.float() - 1.0) * sp.DS
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = s_box          # 박스는 두 다리가 만나는 곳에서 시작
        self._mscale[rows] = 1.0
        if self.steer_mrand > 0:
            # 경로를 n 구간으로 쪼개 구간마다 {1.0, .75, .5, .25} 중 하나.
            # **곡률의 함수로 두면 안 된다** -- 정책이 "커브가 보이니 느리게" 로
            # 보상을 만족시키고 간격 읽는 것을 영영 배우지 않는다. 배포에서 필요한
            # 건 정반대다: 직선에서 "상대가 지나가니 반속" 을 듣는 것.
            g = torch.Generator(device="cpu"); g.manual_seed(seed + 17)
            n = self.steer_mrand
            picks = torch.rand((len(rows), n), generator=g).to(self.device)
            mult = self.steer_m_lo + (1.0 - self.steer_m_lo) * (picks * 4).floor() / 3.0
            edge = torch.linspace(0, sp.V, n + 1, device=self.device).long()
            for j in range(n):
                cols = torch.arange(int(edge[j]), int(edge[j + 1]), device=self.device)
                self._mscale[rows[:, None], cols[None, :]] = mult[:, j:j + 1]

        self._prev_arc[rows] = 0.0
        if self.steer_endclamp:
            # 실제 경로가 끝나는 셀 이후를 정지 명령으로 만든다. 경로 버퍼 뒤쪽은
            # 직선 외삽이라 그대로 두면 목표 위에 서 있어야 할 때 전진을 명령한다.
            _end = torch.norm(self._gt_path[rows] - tar[:, None, :], dim=-1).argmin(dim=1)
            _cols = torch.arange(sp.V, device=self.device)[None, :]
            self._mscale[rows] = torch.where(_cols > _end[:, None],
                                             torch.zeros_like(self._mscale[rows]),
                                             self._mscale[rows])
        if self.scen != "free":
            # 교차점 = env 공통 원점. apply_layout 이 start/tar 를 원점 대칭으로 놓는다.
            org = self.agent_axis(self._initial_humanoid_root_states)[:, :, 0:2].mean(dim=1)
            self._scen_xp[rows] = org[torch.div(rows, self.num_agents, rounding_mode="floor")]
            # **a1 만** 감속한다. a0 은 늘 평속이라 기준이 된다.
            if self.scen in ("cross", "solo"):
                # a1|a0|both. a0 는 **역할 뒤바꾸기 대조군**이다 -- cross 에서 a1 이
                # dt=0 에서도 0.5 s 느린 비대칭이 나왔는데, 그게 학습된 역할인지
                # 기하 탓인지는 감속 대상을 바꿔봐야 갈린다.
                if self.scen_decel == "both":
                    sel = rows
                elif self.scen_decel == "rand":
                    # **누가 양보할지도 에피소드마다 뽑는다.** a1 로 고정하면 정책이
                    # "내가 a1 이면 감속" 을 위치 규칙으로 외운다 -- 실측: 같은 정책에
                    # a0 로 감속을 명령하면 창속도가 1.500->1.324 밖에 안 내려가고
                    # (a1 은 1.233->0.957) 기울기가 -0.158 로 뒤집힌다.
                    # 크기는 창에서 읽지만 역할은 외운 것이라, 월드모델이 누구를
                    # 늦출지 정하는 시나리오를 지탱하지 못한다.
                    _g = torch.Generator(device="cpu"); _g.manual_seed(seed + 137)
                    _e = torch.div(rows, self.num_agents, rounding_mode="floor").unique()
                    _pick = torch.randint(self.num_agents, (len(_e),), generator=_g).to(self.device)
                    sel = _e * self.num_agents + _pick
                else:
                    _who = 0 if self.scen_decel == "a0" else 1
                    sel = rows[(rows % self.num_agents) == _who]
                if len(sel) > 0:
                    dt = self._sample_dt(len(sel), seed + 91)
                    cols = torch.arange(self._win_lo, self._win_hi, device=self.device)
                    self._mscale[sel[:, None], cols[None, :]] = self._decel_mult(dt)[:, None]
                    self._ep_dtcmd[sel] = dt
            if os.environ.get("MS_DBG") and not getattr(self, "_dbg_done", False):
                self._dbg_done = True
                self._scen_selfcheck(rows)
        return

    def _scen_selfcheck(self, rows):
        """MS_DBG=1 : 첫 리셋에서 배치가 실제로 의도대로 됐는지 찍는다.

        조용히 틀리는 것들이 많다 -- 교차점까지 호길이가 어긋나거나(생성 오프셋),
        spawn_gap 이 배치를 흔들거나, 랜덤 M 이 프로파일을 덮거나.
        """
        A = self.num_agents
        p = self._gt_path[rows].cpu().numpy()
        xp = self._scen_xp[rows].cpu().numpy()
        ms = self._mscale[rows].cpu().numpy()
        h = self.humanoid_rows(self._humanoid_root_states)[rows, 0:2].cpu().numpy()
        tar = self._box_tar_pos[rows, 0:2].cpu().numpy()
        a = (rows % A).cpu().numpy()
        print(f"\n[scen] MS_SCEN={self.scen} L={self.scen_L} W={self._w_eff:.2f} "
              f"dt={self.scen_dt} mult={self._decel_mult():.4f} "
              f"cells[{self._win_lo}:{self._win_hi}] 교차셀={self._x_cell} "
              f"회복={self.scen_recov} 감속={self.scen_decel}"
              f"{' PLACEBO' if self.scen_placebo else ''} scen_id={self._scen_id()}", flush=True)
        for k in range(min(2 * A, len(rows))):
            end = int(np.linalg.norm(p[k] - tar[k], axis=-1).argmin())
            seg = np.linalg.norm(np.diff(p[k][:end + 1], axis=0), axis=-1)
            j = int(np.linalg.norm(p[k][:end + 1] - xp[k], axis=-1).argmin())
            lat = np.abs(np.cross(p[k][:end + 1] - p[k][0],
                                  (tar[k] - p[k][0]) / (np.linalg.norm(tar[k] - p[k][0]) + 1e-9))).max()
            mm = ms[k][:end + 1]
            print(f"  row={int(rows[k])} a{a[k]}  시작({h[k][0]:+.2f},{h[k][1]:+.2f}) "
                  f"총호={seg.sum():.2f}m  교차점까지호={seg[:j].sum():.3f}m "
                  f"(빗나감 {np.linalg.norm(p[k][j]-xp[k]):.3f}m)  곡률lat={lat:.3f} "
                  f"M배수 {mm.min():.3f}~{mm.max():.3f}", flush=True)
        exp = self.scen_L / 2.0
        got = float(np.linalg.norm(np.diff(p[0][:int(np.linalg.norm(p[0]-tar[0],axis=-1).argmin())+1],axis=0),axis=-1)[
            :int(np.linalg.norm(p[0][:int(np.linalg.norm(p[0]-tar[0],axis=-1).argmin())+1]-xp[0],axis=-1).argmin())].sum())
        # 휜 경로(MS_SCEN_CURVE=1)는 호길이가 안 맞는 게 정상이다 -- 그게 그 옵션의 대가다.
        # 검사를 그대로 두면 보기용 실행이 오히려 막힌다.
        if int(_f("MS_SCEN_CURVE", 0)):
            print(f"  [주의] 휜 경로 모드 -- 교차점까지 호길이 {got:.3f}m "
                  f"(직선이면 {exp:.3f}m). 판정에는 쓰지 않는다.", flush=True)
        elif abs(got - exp) > 0.2:
            # **원인을 같이 찍는다.** 값만 보면 배치가 틀린 건지 경로가 휜 건지
            # 환경변수가 안 넘어온 건지 구분이 안 된다.
            _lat = float(np.abs(np.cross(p[0][:end + 1] - p[0][0],
                        (tar[0] - p[0][0]) / (np.linalg.norm(tar[0] - p[0][0]) + 1e-9))).max())
            hint = ("경로가 휘었다 (MS_LAT_MAX 가 0 이 아님)" if _lat > 0.05 else
                    "경로는 직선인데 길이가 다르다 (MA_LAYOUT_L 이 MS_L 과 불일치)")
            raise ValueError(
                f"교차점까지 호길이 {got:.3f}m 가 MS_L/2={exp:.3f}m 와 다르다 -- {hint}\n"
                f"  MS_L={os.environ.get('MS_L')} MA_LAYOUT_L={os.environ.get('MA_LAYOUT_L')} "
                f"MA_LAYOUT={os.environ.get('MA_LAYOUT')} MS_LAT_MAX={os.environ.get('MS_LAT_MAX')} "
                f"MS_SCEN_CURVE={os.environ.get('MS_SCEN_CURVE')}\n"
                f"  실측 곡률lat={_lat:.3f}  총호={seg.sum():.3f}m")
        if A == 2:
            # **root state 로 잰다.** agent_min_dist 는 _rigid_body_pos 를 쓰는데
            # 리셋 시점엔 아직 물리 스텝 전이라 **생성 시점 값(원 반지름 1.5 -> 3.0 m)**
            # 이 그대로 남아 있다. 실제로 모든 시나리오에서 2.98 이 찍혀서, 배치가
            # 깨진 것으로 오판하기 딱 좋다. root state 는 방금 쓴 값이라 정확하다.
            h2 = self.agent_axis(self._humanoid_root_states)[:, :, 0:2]
            e2 = torch.div(rows, A, rounding_mode="floor").unique()
            d = torch.norm(h2[e2, 0] - h2[e2, 1], dim=-1).cpu().numpy()
            print(f"  초기 사람간 거리(root) 중앙값 {np.median(d):.2f}m  "
                  f"1%={np.percentile(d, 1):.2f}m  최소={d.min():.2f}m", flush=True)
        return

    # ---- 창 ---------------------------------------------------------------

    def _ratchet(self, xy, arc):
        """경로에 투영하되 **앞쪽만** 본다.

        전체 경로는 박스에서 되꺾이므로, 제한 없는 최근접점은 운반 중인 박스를
        접근 다리로 되돌려 호길이를 거꾸로 흐르게 한다. 단조 스칼라가 그걸 막는다.
        """
        lo = arc - self.steer_back
        hi = arc + 2.0
        s, lat, _ = sp.project(xy, self._gt_path, lo, hi)
        s = torch.maximum(s, lo)
        if self.steer_clip:
            s = torch.minimum(s, self._s_end)
        return s, lat

    def _m_at(self, arc, rows=None):
        """arc 가 부분집합이면 **rows 를 반드시 줘야 한다.**

        arange(len(arc)) 로 인덱싱하면 리셋 때(_compute_task_obs(env_ids)) 부분 rows 가
        들어오면서 **남의 M 프로파일**을 읽는다. 시나리오는 프로파일이 패리티에만
        의존하고 agent_rows 가 패리티 순서를 보존해 우연히 무해하지만,
        MS_MRAND>0 은 행마다 다르게 뽑으므로 실제로 틀린 값이 첫 관측에 들어간다.
        """
        j = (arc / sp.DS).long().clamp(0, sp.V - 1)
        ar = torch.arange(arc.shape[0], device=self.device) if rows is None else rows
        return self.steer_m_nom * self._mscale[ar, j]

    def _aim_pt(self, arc, look):
        tgt = arc + look
        if self.steer_clip:
            tgt = torch.minimum(tgt, self._s_end)
        q = (tgt / sp.DS).clamp(0, sp.V - 2)
        lo = q.floor().long()
        frac = (q - lo.float())[:, None]
        ar = torch.arange(arc.shape[0], device=self.device)
        return self._gt_path[ar, lo] + frac * (self._gt_path[ar, lo + 1] - self._gt_path[ar, lo])

    def _steer_obs(self, rows):
        """등간격 K 점. **길이 M 이 곧 속도 명령**이고 형태는 원본과 같다."""
        n = len(rows)
        if self.steer_zero:
            return torch.zeros(n, self.steer_dim(), device=self.device)
        h = self.humanoid_rows(self._humanoid_root_states)[rows]
        arc = self._arc_root[rows]
        M = self._m_at(arc, rows)
        off = torch.arange(1, self.steer_k + 1, device=self.device)[None, :] / self.steer_k
        pts_arc = arc[:, None] + M[:, None] * off
        if self.steer_clip:
            # 끝에 다가가면 점들이 끝에 쌓여 **창이 저절로 짧아진다** = 감속 명령.
            pts_arc = torch.minimum(pts_arc, self._s_end[rows][:, None])
        q = (pts_arc / sp.DS).clamp(0, sp.V - 2)
        lo = q.floor().long(); frac = (q - lo.float())[..., None]
        ar = torch.arange(n, device=self.device)[:, None]
        pts = self._gt_path[rows][ar, lo] + frac * (
            self._gt_path[rows][ar, lo + 1] - self._gt_path[rows][ar, lo])
        root_xy = h[:, 0:2]
        heading = torch.atan2(
            2.0 * (h[:, 6] * h[:, 5] + h[:, 3] * h[:, 4]),
            1.0 - 2.0 * (h[:, 4] ** 2 + h[:, 5] ** 2))
        return sp.to_local(pts, root_xy, heading).reshape(n, -1)

    def _compute_task_obs(self, env_ids=None):
        # 부모(HumanoidMACarry)가 [teammate | carry | carry] 를 만든다.
        # 그 사이에 steer 창을 끼운다.
        base = super()._compute_task_obs(env_ids)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        t_dim = TEAMMATE_DIM * (self.num_agents - 1)
        teammate = base[:, :t_dim]
        carry = base[:, t_dim:t_dim + (CARRY_HI - CARRY_LO)]
        return torch.cat([teammate, self._steer_obs(rows), carry, carry], dim=-1)

    # ---- 보상 -------------------------------------------------------------

    def _compute_reward(self, actions):
        """원본 carry 보상에서 **두 줄만** 바꾼다.

            v_tar = M / 1.6       원본은 1.5 상수
            u     = 경로상 조준점 방향   원본은 목표로 직진

        그 위에 latpen `c·(exp(-0.5·e²) − 1)` 을 더한다. **-1 이 붙어야** 한다 --
        더하기형은 에피소드 리턴을 상수만큼 부풀리고 런 간 산포도 같이 커져서
        신호/노이즈가 c 에 안 붙는다 (실측: 성공률 0.58 붕괴).

        pos_near / handheld / putdown / power 는 전부 원본 그대로다.
        """
        A = self.num_agents
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        root_pos = h[..., 0:3]
        box_pos = b[..., 0:3]
        tar = self._box_tar_pos

        self._arc_root, lat_root = self._ratchet(root_pos[:, 0:2], self._arc_root)
        self._arc_box, lat_box = self._ratchet(box_pos[:, 0:2], self._arc_box)
        self._lat_box = lat_box
        self._lat_root = lat_root

        root_vel = (root_pos - self._prev_root_pos)[:, 0:2] / self.dt
        box_vel3 = (box_pos - self._prev_box_pos) / self.dt

        m_root = self._m_at(self._arc_root)
        m_box = self._m_at(self._arc_box)
        v_root, v_box = m_root / 1.6, m_box / 1.6
        l_root, l_box = m_root * 0.5, m_box * 0.5      # L 은 늘 창의 절반

        u_root = torch.nn.functional.normalize(
            self._aim_pt(self._arc_root, l_root) - root_pos[:, 0:2], dim=-1)
        u_box = torch.nn.functional.normalize(
            self._aim_pt(self._arc_box, l_box) - box_pos[:, 0:2], dim=-1)

        def vel_term(vel_xy, direction, pin, v_tar):
            along = (direction * vel_xy).sum(dim=-1)
            r = torch.exp(-self.steer_vel_k * (v_tar - along) ** 2)
            r = torch.where(along <= 0, torch.zeros_like(r), r)
            # v_tar=0 이면 along-track 투영이 의미를 잃는다. "아무 방향으로도
            # 움직이지 마라" 가 곧 정지다.
            stopped = v_tar < 0.05
            r = torch.where(stopped,
                            torch.exp(-self.steer_vel_k * vel_xy.pow(2).sum(dim=-1)), r)
            return torch.where(pin, torch.ones_like(r), r)

        def latpen(lat, pin):
            r = torch.exp(-0.5 * lat ** 2) - 1.0
            return torch.where(pin, torch.zeros_like(r), r)

        # --- walk: root 가 접근 다리를 간다
        pos_err = ((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1)
        pin_walk = pos_err < 0.5 ** 2
        _vw = 0.2 * self.steer_vel_w
        walk_r = _vw * vel_term(root_vel, u_root, pin_walk, v_root)
        walk_r = walk_r + self.steer_pos_c * latpen(lat_root, pos_err < self.steer_pin ** 2)

        # --- carry: 박스가 운반 다리를 간다
        diff = tar - box_pos
        err_xy = (diff[:, 0:2] ** 2).sum(dim=-1)
        pin_carry = err_xy < 0.5 ** 2
        pos_near = torch.exp(-10.0 * (diff ** 2).sum(dim=-1))
        carry_r = _vw * vel_term(box_vel3[:, 0:2], u_box, pin_carry, v_box) + 0.2 * pos_near
        carry_r = carry_r + self.steer_pos_c * latpen(lat_box, err_xy < self.steer_pin ** 2)
        if self._carry_rwd_box_vel_penalty:
            # 원본 그대로 -- "뻣뻣하게 잡는 것을 막기 위해" 박스를 내던지는 걸 벌한다.
            thre = self._carry_rwd_box_vel_pen_thre
            spd = box_vel3.norm(dim=-1).clamp(min=thre)
            carry_r = carry_r - self._carry_rwd_box_vel_pen_coeff * (
                1.0 - torch.exp(-2.0 * (thre - spd) ** 2))

        rb = self.humanoid_rows(self._rigid_body_pos)
        hands = rb[:, self._key_body_ids[[0, 1]]].mean(dim=1)
        handheld = torch.exp(-5.0 * ((hands - box_pos) ** 2).sum(dim=-1))
        handheld = torch.where(
            ((box_pos[:, 0:2] - root_pos[:, 0:2]) ** 2).sum(dim=-1) > 0.7 ** 2,
            torch.zeros_like(handheld), handheld)

        putdown = (torch.abs(box_pos[:, -1] - tar[:, -1]) <= 0.001).float()
        putdown = torch.where(err_xy > 0.1 ** 2, torch.zeros_like(putdown), putdown)

        self.rew_buf[:] = 2.0 * walk_r + 2.0 * carry_r + 0.2 * handheld + 0.2 * putdown
        if self._power_reward:
            power = torch.abs(self.dof_force_tensor
                              * self.humanoid_rows(self._dof_vel)).sum(dim=-1)
            self.rew_buf[:] = self.rew_buf - self._power_coefficient * power

        self._apply_team_reward()      # ma 의 팀 항 (c=beta=0 이면 아무것도 안 한다)
        return

    # ---- 시나리오 지표 -------------------------------------------------------

    def update_metrics(self):
        """부모 지표 위에 시나리오 전용을 얹는다. **_compute_reward 뒤**라
        _arc_root 가 이번 스텝 값이다 (_compute_reset -> update_metrics 순서)."""
        super().update_metrics()

        # --- 늘 재는 것: 경로 이탈과 속도 추종 ---
        arc0 = self._arc_root
        v_real0 = ((arc0 - self._prev_arc) / self.dt).clamp(min=0.0)
        v_cmd0 = self._m_at(arc0) / 1.6
        self._ep_latr += self._lat_root.abs()
        self._ep_latb += self._lat_box.abs()
        self._ep_spd += (v_real0 - v_cmd0).abs()
        self._ep_vr += v_real0
        self._ep_steps += 1
        # **정지 스텝을 빼고 쌓는다.** 넣으면 보행속도가 아니라 "평균 진척률" 이 되어
        # 실제 1.02 m/s 를 0.58 로 보이게 만든다 (그 값으로 원인을 오진했다).
        _mv = v_real0 > self._metric_still_v
        _b = torch.bucketize(v_cmd0, self._VBINS)                 # 0..3
        _ar = torch.arange(v_cmd0.shape[0], device=self.device)
        self._ep_vbin_s[_ar, _b] += torch.where(_mv, v_real0, torch.zeros_like(v_real0))
        self._ep_vbin_n[_ar, _b] += _mv.float()

        if self.scen == "free":
            self._prev_arc = arc0.clone()      # free 도 다음 스텝 차분을 위해 갱신한다
            return
        h = self.humanoid_rows(self._humanoid_root_states)
        step = self.progress_rows()

        # ① 교차점 도달 시각: 교차점에 가장 가까웠던 순간. 경로를 정확히 통과하지
        #    않아도 정의된다. 명령한 오프셋이 실제로 나오는지가 이 트랙의 주 지표다.
        dx = torch.norm(h[:, 0:2] - self._scen_xp, dim=-1)
        closer = dx < self._ep_xdist
        self._ep_xdist = torch.where(closer, dx, self._ep_xdist)
        self._ep_xtime = torch.where(closer, step, self._ep_xtime)

        # ② 조우 중 상대와의 최소거리. 에피소드 전체 평균은 **경로 대부분이 서로
        #    멀어서 평평해진다** -- ma 에서 17 조건 전부 1.62~1.93 으로 나온 이유다.
        #    교차점 근처로 자르면 그 희석이 사라진다.
        d = self.agent_min_dist()
        near = dx < self.scen_encr
        self._ep_encd = torch.where(near & (d < self._ep_encd), d, self._ep_encd)

        # ③ 감속 구간에서 명령 속도를 지켰나. 호길이 증가율이 곧 경로상 속도다.
        arc = self._arc_root
        j = (arc / sp.DS).long().clamp(0, sp.V - 1)
        inw = (j >= self._win_lo) & (j < self._win_hi)
        v_real = ((arc - self._prev_arc) / self.dt).clamp(min=0.0)
        v_cmd = self._m_at(arc) / 1.6
        self._ep_wn += inw.long()
        self._ep_wv += torch.where(inw, v_real, torch.zeros_like(v_real))
        self._ep_wc += torch.where(inw, v_cmd, torch.zeros_like(v_cmd))
        self._prev_arc = arc.clone()

        # ④ 창을 **끝까지** 지났나. wn 은 창을 못 끝낸 행에서도 값이 쌓이므로
        #    그대로 쓰면 "느려서 오래 걸림" 과 "중간에 넘어짐" 이 섞인다.
        self._ep_wdone = torch.maximum(self._ep_wdone, (j >= self._win_hi).long())
        jb = (self._arc_box / sp.DS).long().clamp(0, sp.V - 1)
        self._ep_wnbox += ((jb >= self._win_lo) & (jb < self._win_hi)).long()

        # ⑤ 창 안에서의 횡방향 이탈. **타이밍 대 우회의 반증 지표다** --
        #    여유가 늘었는데 이것도 늘었으면 옆으로 비켜서 번 것이지 늦게 와서가 아니다.
        #    d_min 은 env 당 대칭 스칼라 하나라 누가 양보했는지 영영 말할 수 없다.
        self._ep_latw += torch.where(inw, self._lat_root.abs(),
                                     torch.zeros_like(self._lat_root))
        self._ep_encn += near.long()
        return

    def _scen_id(self):
        base = {"free": 0, "parallel": 1, "cross": 2, "solo": 3}[self.scen]
        return base + (10 if self.scen_decel == "both" else 0) + (20 if self.scen_placebo else 0)

    def _metric_extra_cols(self, rows):
        # **열 13~19 는 고정이다.** train.sh 가 위치로 읽으므로 새 열은 20 부터 붙인다.
        sid = torch.full_like(self._ep_latw[rows], float(self._scen_id()))
        cols = [self._ep_xtime[rows].float(),
                torch.clamp(self._ep_xdist[rows], max=99.0),
                torch.clamp(self._ep_encd[rows], max=99.0),
                self._ep_wn[rows].float(),
                self._ep_wv[rows],
                self._ep_wc[rows],
                self._ep_dtcmd[rows],
                self._ep_wdone[rows].float(),      # 20
                self._ep_wnbox[rows].float(),      # 21
                self._ep_latw[rows],               # 22
                self._ep_encn[rows].float(),       # 23
                self._arc_root[rows],              # 24  패딩 구간까지 갔는지 진단
                sid,                               # 25  조건 오라벨 방지
                self._ep_latr[rows],               # 26  경로 이탈 |lat_root| 합
                self._ep_latb[rows],               # 27  박스 이탈 |lat_box| 합
                self._ep_spd[rows],                # 28  |v_real - v_cmd| 합
                self._ep_vr[rows],                 # 29  v_real 합
                self._ep_steps[rows].float()]      # 30  분모
        cols += [self._ep_vbin_s[rows, i] for i in range(4)]      # 31~34 구간별 v_real 합
        cols += [self._ep_vbin_n[rows, i] for i in range(4)]      # 35~38 구간별 스텝 수
        cols.append(self._s_end[rows])                            # 39 실제 경로 끝 (호길이)
        return cols

    def _metric_reset_extra(self, rows):
        self._ep_xtime[rows] = -1
        self._ep_xdist[rows] = 1e6
        self._ep_encd[rows] = 1e6
        self._ep_wn[rows] = 0
        self._ep_wv[rows] = 0.0
        self._ep_wc[rows] = 0.0
        self._ep_wdone[rows] = 0
        self._ep_wnbox[rows] = 0
        self._ep_latw[rows] = 0.0
        self._ep_encn[rows] = 0
        for b in (self._ep_latr, self._ep_latb, self._ep_spd, self._ep_vr):
            b[rows] = 0.0
        self._ep_steps[rows] = 0
        self._ep_vbin_s[rows] = 0.0
        self._ep_vbin_n[rows] = 0.0
        return

    # ---- 뷰어 -------------------------------------------------------------

    def _init_camera(self):
        """MS_CAM=top 이면 env 전체가 들어오는 높은 시점으로 고정한다.

        기본 카메라는 agent 0 **뒤 3 m, 높이 1 m** 에 붙어 따라다녀서, 12 m 배치에서는
        한 사람만 크게 잡히고 상대도 경로도 화면 밖으로 나간다. 영상용으로는 못 쓴다.
        """
        if os.environ.get("MS_CAM") != "top":
            return super()._init_camera()
        from isaacgym import gymapi
        self.gym.refresh_actor_root_state_tensor(self.sim)
        org = self.agent_axis(self._initial_humanoid_root_states)[0, :, 0:2].mean(dim=0).cpu().numpy()
        H = _f("MS_CAM_H", 17.0)          # 높이
        B = _f("MS_CAM_B", 9.0)           # 뒤로 물러난 거리. 0 이면 완전 수직
        self._cam_prev_char_pos = np.array([org[0], org[1], 1.0], dtype=np.float32)
        self.gym.viewer_camera_look_at(
            self.viewer, None,
            gymapi.Vec3(float(org[0]), float(org[1] - B), H),
            gymapi.Vec3(float(org[0]), float(org[1]), 0.0))
        return

    def _update_camera(self):
        # 고정 시점이면 따라가지 않는다. 따라가면 배치가 화면 안에서 흔들린다.
        if os.environ.get("MS_CAM") == "top":
            return
        return super()._update_camera()


    def _draw_task(self):
        """기본 오버레이 대신 **경로 두 벌**을 그린다. A=2 라 사람마다 색이 다르다.

            바닥의 넓은 띠   그 사람에게 명령된 전체 경로 (목표에서 잘림)
            허리 높이 띠     정책에 실제로 들어가는 K 점 창. **길이가 곧 속도 명령**이다
            기둥            보상이 조준하는 점

        add_lines 는 굵기가 없어서, 띠는 얇은 선을 촘촘히 겹쳐 만든다.
        MS_DRAW_TASK=1 이면 원본 오버레이(박스 bbox 등)도 같이 그린다.
        """
        if os.environ.get("MS_DRAW_TASK"):
            super()._draw_task()
            if self.viewer is None:
                return
        else:
            self._update_marker()
            if self.viewer is None:
                return
            self.gym.clear_lines(self.viewer)
        if self.steer_zero:
            return

        A = self.num_agents
        rows = self.all_rows()
        h = self.humanoid_rows(self._humanoid_root_states)
        arc = self._arc_root
        M = self._m_at(arc)
        # 창: 등간격 K 점. 길이 M 이 속도 명령이다
        off = torch.arange(1, self.steer_k + 1, device=self.device)[None, :] / self.steer_k
        q = ((arc[:, None] + M[:, None] * off) / sp.DS).clamp(0, sp.V - 2)
        lo = q.floor().long(); frac = (q - lo.float())[..., None]
        ar = torch.arange(len(rows), device=self.device)[:, None]
        win = self._gt_path[ar, lo] + frac * (self._gt_path[ar, lo + 1] - self._gt_path[ar, lo])
        aim = self._aim_pt(arc, M * 0.5)

        gt_all = self._gt_path.cpu().numpy()
        tar_xy = self._box_tar_pos[:, 0:2].cpu().numpy()
        win_np = win.cpu().numpy()
        aim_np = aim.cpu().numpy()
        root_z = h[:, 2].cpu().numpy()
        s_end_np = self._s_end.cpu().numpy()
        draw_speed = int(_f("MS_DRAW_SPEED", 1))
        mscale_np = self._mscale.cpu().numpy()

        gt_w = 0.30
        win_w = 0.16
        # 사람마다 다른 색. 안 그러면 누구 경로인지 알 수 없다.
        GT_COL = [[1.00, 0.15, 0.65], [1.00, 0.55, 0.10], [0.70, 0.30, 1.00], [0.20, 0.80, 0.20]]
        WIN_COL = [[0.10, 0.95, 1.00], [0.35, 1.00, 0.35], [1.00, 1.00, 0.40], [0.50, 0.80, 1.00]]

        def band(pts, z, width, spacing=0.02):
            n = max(int(width / spacing), 3)
            a, b = pts[:-1], pts[1:]
            d = b - a
            nrm = np.stack([-d[:, 1], d[:, 0]], axis=-1)
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
            out = []
            for o in np.linspace(-width / 2, width / 2, n):
                aa, bb = a + nrm * o, b + nrm * o
                out.append(np.concatenate([
                    aa, np.full((len(aa), 1), z),
                    bb, np.full((len(bb), 1), z)], axis=1))
            return np.concatenate(out).astype(np.float32)

        def _runs(v, eps=1e-6):
            """같은 값이 이어지는 구간을 [(시작, 끝)] 로 묶는다."""
            out, i = [], 0
            for j in range(1, len(v) + 1):
                if j == len(v) or abs(v[j] - v[i]) > eps:
                    out.append((i, j - 1)); i = j
            return out

        def col(rgb, n):
            return np.tile(np.array([rgb], np.float32), (n, 1))

        for i, env_ptr in enumerate(self.envs):
            for a in range(A):
                r = i * A + a
                # **argmin 을 쓰면 안 된다.** 휘어진 경로가 목표 근처를 스쳐 지나가면
                # 엉뚱한 데를 끝으로 잡아 띠가 중간에서 끊긴다. s_end 가 진짜 끝이다.
                end = int(s_end_np[r] / sp.DS) + 1
                g = gt_all[r][:max(end, 2)][::2]
                if draw_speed:
                    # **속도 명령을 바닥 띠 색으로 보여준다.** 창 길이만으로는 순간값이라
                    # 앞으로 어디서 느려지는지가 안 보인다. 배수가 낮을수록 어둡다.
                    ms = mscale_np[r][:max(end, 2)][::2]
                    for lo_i, hi_i in _runs(ms):
                        if hi_i - lo_i < 2:
                            continue
                        seg = g[lo_i:hi_i + 1]
                        v = band(seg, 0.04 + 0.01 * a, gt_w)
                        f = float(ms[lo_i])
                        base = GT_COL[a % len(GT_COL)]
                        shade = [c * (0.28 + 0.72 * f) for c in base]
                        self.gym.add_lines(self.viewer, env_ptr, len(v), v, col(shade, len(v)))
                else:
                    v = band(g, 0.04 + 0.01 * a, gt_w)
                    self.gym.add_lines(self.viewer, env_ptr, len(v), v,
                                       col(GT_COL[a % len(GT_COL)], len(v)))

                zw = float(root_z[r])
                v = band(win_np[r], zw, win_w)
                self.gym.add_lines(self.viewer, env_ptr, len(v), v,
                                   col(WIN_COL[a % len(WIN_COL)], len(v)))

                pt = aim_np[r]
                o = np.linspace(-0.04, 0.04, 9)[:, None]
                v = np.tile(np.array([[pt[0], pt[1], 0.02, pt[0], pt[1], zw]], np.float32), (9, 1))
                v[:, [0, 3]] += o
                self.gym.add_lines(self.viewer, env_ptr, len(v), v,
                                   col([1.0, 0.90, 0.10], len(v)))
        return
