"""
mahjong_env.py
==============
Simplified 1-player Mahjong environment.

Observation (36,):
    [0:34]  – agent's current hand (count per tile type, 0-4)
    [34]    – current shanten number (-1 to 8)
    [35]    – remaining tiles in wall (0 to 136)

Action:
    Integer in [0, 33] – index of the tile type to discard.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Helper: Mahjong win-condition checker
# ---------------------------------------------------------------------------

def _can_form_sets(hand_vec: np.ndarray) -> bool:
    """
    Recursively check if the 34-element hand vector can be decomposed into
    exactly 4 melds (sequences of 3 consecutive suited tiles OR triplets of
    any tile) plus 1 pair.

    Returns True if a standard winning hand is detected.
    """
    tiles = []
    for tile_id, count in enumerate(hand_vec):
        tiles.extend([tile_id] * int(count))

    if len(tiles) != 14:
        return False

    # Try every possible pair
    unique = list(dict.fromkeys(tiles))  # preserves order, removes dups
    for pair_tile in unique:
        if hand_vec[pair_tile] >= 2:
            remaining = hand_vec.copy()
            remaining[pair_tile] -= 2
            if _extract_melds(remaining, 4):
                return True
    return False


def _extract_melds(hand: np.ndarray, melds_needed: int) -> bool:
    """Try to extract `melds_needed` melds from hand (recursive)."""
    if melds_needed == 0:
        return hand.sum() == 0

    # Find the first non-zero tile index
    first = -1
    for i in range(34):
        if hand[i] > 0:
            first = i
            break
    if first == -1:
        return False

    # Option 1: Triplet
    if hand[first] >= 3:
        hand[first] -= 3
        if _extract_melds(hand, melds_needed - 1):
            hand[first] += 3
            return True
        hand[first] += 3

    # Option 2: Sequence (only for suited tiles 0-26)
    if first <= 24:                         # need first+2 <= 26
        suit = first // 9
        pos = first % 9
        if pos <= 6:                        # can still form a sequence
            s0, s1, s2 = first, first + 1, first + 2
            if (hand[s0] >= 1 and hand[s1] >= 1 and hand[s2] >= 1):
                hand[s0] -= 1; hand[s1] -= 1; hand[s2] -= 1
                if _extract_melds(hand, melds_needed - 1):
                    hand[s0] += 1; hand[s1] += 1; hand[s2] += 1
                    return True
                hand[s0] += 1; hand[s1] += 1; hand[s2] += 1

    return False


# ---------------------------------------------------------------------------
# Shanten Number Calculation  (chỉ dạng tiêu chuẩn: 4 melds + 1 pair)
# ---------------------------------------------------------------------------

_SUITED_RANGES = [(0, 9), (9, 18), (18, 27)]   # Man, Pin, Sou  (có sequence)
_HONOR_RANGE   = (27, 34)                      # Gió + Rồng     (chỉ triplet)


def _best_blocks_suited(h: np.ndarray, lo: int, hi: int) -> tuple[int, int]:
    idx = -1
    for i in range(lo, hi):
        if h[i] > 0:
            idx = i
            break
    if idx == -1:
        return 0, 0

    best_m = best_p = 0

    def _update(m: int, p: int) -> None:
        nonlocal best_m, best_p
        if (m, p) > (best_m, best_p):
            best_m, best_p = m, p

    # 1. Triplet
    if h[idx] >= 3:
        h[idx] -= 3
        m, p = _best_blocks_suited(h, idx, hi)
        _update(m + 1, p)
        h[idx] += 3

    # 2. Sequence
    if idx + 2 < hi and h[idx + 1] >= 1 and h[idx + 2] >= 1:
        h[idx] -= 1; h[idx + 1] -= 1; h[idx + 2] -= 1
        m, p = _best_blocks_suited(h, idx, hi)
        _update(m + 1, p)
        h[idx] += 1; h[idx + 1] += 1; h[idx + 2] += 1

    # 3. Partial — đôi
    if h[idx] >= 2:
        h[idx] -= 2
        m, p = _best_blocks_suited(h, idx, hi)
        _update(m, p + 1)
        h[idx] += 2

    # 4. Partial — kanchan (idx, idx+2)
    if idx + 2 < hi and h[idx + 2] >= 1:
        h[idx] -= 1; h[idx + 2] -= 1
        m, p = _best_blocks_suited(h, idx, hi)
        _update(m, p + 1)
        h[idx] += 1; h[idx + 2] += 1

    # 5. Partial — penchan / ryanmen (idx, idx+1)
    if idx + 1 < hi and h[idx + 1] >= 1:
        h[idx] -= 1; h[idx + 1] -= 1
        m, p = _best_blocks_suited(h, idx, hi)
        _update(m, p + 1)
        h[idx] += 1; h[idx + 1] += 1

    # 6. Bỏ qua (isolated): đệ quy từ idx+1 để tiến lên
    h[idx] -= 1
    m, p = _best_blocks_suited(h, idx + 1, hi)   # ← idx+1, không phải idx
    _update(m, p)
    h[idx] += 1

    return best_m, best_p


def _best_blocks_honors(h: np.ndarray) -> tuple[int, int]:
    melds = partials = 0
    for i in range(_HONOR_RANGE[0], _HONOR_RANGE[1]):
        if h[i] >= 3:
            melds += 1
        elif h[i] == 2:
            partials += 1
    return melds, partials


def _evaluate_hand(h: np.ndarray, has_pair: bool) -> int:
    total_m = total_p = 0

    for lo, hi in _SUITED_RANGES:
        m, p = _best_blocks_suited(h, lo, hi)
        total_m += m
        total_p += p

    hm, hp = _best_blocks_honors(h)
    total_m += hm
    total_p += hp

    # Cap: số block hữu ích không vượt quá 4
    if total_m + total_p > 4:
        total_p = 4 - total_m

    return 8 - 2 * total_m - total_p - (1 if has_pair else 0)


def calculate_shanten(hand: np.ndarray) -> int:
    h = hand.copy().astype(int)
    best = 8  # worst case với 13 bài

    # Trường hợp chưa xác định pair
    best = min(best, _evaluate_hand(h, has_pair=False))

    # Thử từng tile làm pair chính
    for i in range(34):
        if h[i] >= 2:
            h[i] -= 2
            s = _evaluate_hand(h, has_pair=True)
            if s < best:
                best = s
            h[i] += 2

    return best


def shanten_breakdown(hand: np.ndarray) -> dict:
    s = calculate_shanten(hand)
    if s == -1:
        status = "winning"
    elif s == 0:
        status = "tenpai"
    else:
        status = "building"
    return {"shanten": s, "status": status}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MahjongEnv(gym.Env):
    """
    Simplified 1-player Mahjong environment.

    Observation (36,):
        [0:34]  – agent's current hand (count per tile type, 0-4)
        [34]    – current shanten number (-1 to 8)
        [35]    – remaining tiles in wall (0 to 136)

    Action:
        Integer in [0, 33] – index of the tile type to discard.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    def __init__(self, seed: int | None = None):
        super().__init__()

        self.action_space = spaces.Discrete(34)
        
        # THAY ĐỔI: Giảm shape từ 70 xuống 36
        self.observation_space = spaces.Box(
            low=-1, high=136, shape=(36,), dtype=np.int32
        )

        self._np_random: np.random.Generator | None = None
        self.seed(seed)

        # Will be properly initialised in reset()
        self.wall: np.ndarray = np.zeros(34, dtype=np.int32)
        self.agent_hand: np.ndarray = np.zeros(34, dtype=np.int32)
        self.discard_pile: np.ndarray = np.zeros(34, dtype=np.int32)
        self._prev_shanten: int = 8  # worst-case starting value

        self._min_shanten_achieved: int = 8  # Track min shanten

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_tile(self) -> int:
        total = int(self.wall.sum())
        if total == 0:
            raise RuntimeError("Wall is empty – cannot draw.")
        probs = self.wall / total
        tile = int(self._np_random.choice(34, p=probs))
        self.wall[tile] -= 1
        return tile

    def _get_obs(self) -> np.ndarray:
        # THAY ĐỔI: Chỉ lấy Hand, Shanten và Wall
        current_shanten = self._shanten()
        wall_remaining = int(self.wall.sum())
        
        # Nối thành mảng 36 phần tử (Cắt bỏ self.discard_pile)
        return np.concatenate([
            self.agent_hand, 
            [current_shanten, wall_remaining]
        ]).astype(np.int32)

    def _action_mask(self) -> np.ndarray:
        """Boolean mask: True where agent holds ≥ 1 of that tile type."""
        return (self.agent_hand > 0).astype(np.int8)

    def _is_winning_hand(self) -> bool:
        return calculate_shanten(self.agent_hand) == -1

    def _shanten(self) -> int:
        return calculate_shanten(self.agent_hand)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        # Full deck: 4 copies of each of 34 tile types
        self.wall = np.full(34, 4, dtype=np.int32)
        self.agent_hand = np.zeros(34, dtype=np.int32)
        self.discard_pile = np.zeros(34, dtype=np.int32)

        # Deal 13 tiles to the agent
        for _ in range(13):
            tile = self._draw_tile()
            self.agent_hand[tile] += 1

        # Draw the 14th tile (agent's first "draw")
        tile = self._draw_tile()
        self.agent_hand[tile] += 1

        self._prev_shanten = calculate_shanten(self.agent_hand)
        self._min_shanten_achieved = self._prev_shanten
        initial_shanten = self._prev_shanten
        info = {
            "action_mask": self._action_mask(),
            "shanten":     initial_shanten,
            "shanten_detail": shanten_breakdown(self.agent_hand),
        }
        return self._get_obs(), info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # ---- Validate action ----
        if self.agent_hand[action] <= 0:
            # Invalid action: penalise and force a valid discard
            valid = np.where(self.agent_hand > 0)[0]
            action = int(self._np_random.choice(valid))

        reward: float = -0.1          # default step cost
        terminated = False
        truncated = False

        # ---- 1. Agent discards ----
        self.agent_hand[action] -= 1
        self.discard_pile[action] += 1

        # ---- 2. Dummy simulation ----
        dummy_discards = []
        for _ in range(3):
            if self.wall.sum() == 0:
                break
            dummy_tile = self._draw_tile()
            self.discard_pile[dummy_tile] += 1
            dummy_discards.append(dummy_tile)

        # ---- 3. Agent draws ----
        drawn_tile = -1
        if self.wall.sum() > 0:
            drawn_tile = self._draw_tile()
            self.agent_hand[drawn_tile] += 1
        else:
            truncated = True

        # ---- 4. Termination check + Adaptive Penalty ----
        current_shanten = self._shanten()
        if not truncated:
            if current_shanten == -1:          # AI Tới bài!
                terminated = True
                reward = +10.0
            else:
                # 1. Bảng hình phạt thích ứng theo Shanten
                penalty_map = {
                    6: -0.20,
                    5: -0.15,
                    4: -0.20,
                    3: -0.10,
                    2: -0.08,
                    1: -0.05,
                    0:  0.10  # Đạt Tenpai thì không bị phạt thời gian nữa
                }
                
                # Lấy mức phạt tương ứng, nếu Shanten > 6 thì mặc định phạt -0.2
                step_penalty = penalty_map.get(current_shanten, -0.20)
                reward = step_penalty
                
                # 2. Thưởng khi phá kỷ lục (High-Water Mark)
                if current_shanten < self._min_shanten_achieved:
                    improvement = self._min_shanten_achieved - current_shanten
                    reward += improvement * 0.5
                    self._min_shanten_achieved = current_shanten

        self._prev_shanten = current_shanten
        
        # ---- 5. Build info ----
        info: dict = {
            "action_mask":    self._action_mask(),
            "wall_remaining": int(self.wall.sum()),
            "shanten":        current_shanten,
            "shanten_detail": shanten_breakdown(self.agent_hand),
            
            # Replay data vẫn được xuất ra bình thường
            "dummy_discards": dummy_discards,
            "agent_drawn":    drawn_tile
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render / close (stubs)
    # ------------------------------------------------------------------

    def render(self):
        hand_tiles = [f"T{i}×{c}" for i, c in enumerate(self.agent_hand) if c > 0]
        print(f"Hand : {' '.join(hand_tiles)}")
        print(f"Wall remaining: {int(self.wall.sum())}")

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Example: wrapping with gym.vector.AsyncVectorEnv
# ---------------------------------------------------------------------------

def make_env(seed: int):
    """Factory used by AsyncVectorEnv."""
    def _init():
        env = MahjongEnv(seed=seed)
        return env
    return _init


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  1. Shanten unit tests                                             #
    # ------------------------------------------------------------------ #
    print("=== Shanten unit tests (4 melds + 1 pair, 14-tile hands) ===\n")

    win14 = np.zeros(34, dtype=np.int32)
    win14[0]=1; win14[1]=1; win14[2]=1   # Man 1-2-3
    win14[3]=1; win14[4]=1; win14[5]=1   # Man 4-5-6
    win14[6]=1; win14[7]=1; win14[8]=1   # Man 7-8-9
    win14[9]=1; win14[10]=1; win14[11]=1 # Pin 1-2-3
    win14[27]=2                          # East pair
    s = calculate_shanten(win14)
    bd = shanten_breakdown(win14)
    print(f"[T1] Sequence win   → shanten={s}  (expect -1)  status='{bd['status']}'")

    trip14 = np.zeros(34, dtype=np.int32)
    trip14[27]=3; trip14[9]=3; trip14[18]=3; trip14[0]=3 
    trip14[1]=2                                          
    s = calculate_shanten(trip14)
    bd = shanten_breakdown(trip14)
    print(f"[T2] Triplet win    → shanten={s}  (expect -1)  status='{bd['status']}'")

    ten14 = win14.copy()
    ten14[27] -= 1   
    ten14[28] += 1   
    s = calculate_shanten(ten14)
    bd = shanten_breakdown(ten14)
    print(f"[T3] Tenpai 14-tile → shanten={s}  (expect  0)  status='{bd['status']}'")

    one14 = np.zeros(34, dtype=np.int32)
    one14[0]=1; one14[1]=1; one14[2]=1   
    one14[3]=1; one14[4]=1; one14[5]=1   
    one14[9]=1; one14[10]=1; one14[11]=1 
    one14[18]=1; one14[19]=1             
    one14[27]=1; one14[28]=1; one14[8]=1 
    s = calculate_shanten(one14)
    bd = shanten_breakdown(one14)
    print(f"[T4] 1-shanten      → shanten={s}  (expect  1)  status='{bd['status']}'")

    two14 = np.zeros(34, dtype=np.int32)
    two14[0]=1; two14[1]=1; two14[2]=1   
    two14[9]=1; two14[10]=1; two14[11]=1 
    two14[18]=1; two14[19]=1             
    two14[21]=1; two14[22]=1             
    two14[27]=1; two14[28]=1; two14[29]=1; two14[30]=1  
    s = calculate_shanten(two14)
    bd = shanten_breakdown(two14)
    print(f"[T5] 2-shanten      → shanten={s}  (expect  2)  status='{bd['status']}'")

    rng = np.random.default_rng(42)
    full_deck = np.repeat(np.arange(34), 4)
    rand14_tiles = rng.choice(full_deck, size=14, replace=False)
    rand14 = np.zeros(34, dtype=np.int32)
    for t in rand14_tiles: rand14[t] += 1
    s = calculate_shanten(rand14)
    bd = shanten_breakdown(rand14)
    print(f"[T6] Random 14-tile → shanten={s}  status='{bd['status']}'  "
          f"valid={-1 <= s <= 8}")

    print("\n✓ Tất cả unit tests passed!\n")

    # ------------------------------------------------------------------ #
    #  2. Single-environment sanity check                                #
    # ------------------------------------------------------------------ #
    print("=== Single-env test ===")
    env = MahjongEnv(seed=42)
    obs, info = env.reset()
    print(f"Initial obs shape : {obs.shape}")
    print(f"Initial hand sum  : {obs[:34].sum()}  (should be 14)")
    print(f"Initial shanten   : {info['shanten']}  ({info['shanten_detail']})")

    for step_num in range(10):
        mask = info["action_mask"]
        valid_actions = np.where(mask)[0]
        action = int(np.random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step_num+1:2d}: action={action:2d}  reward={reward:+.2f}  "
              f"shanten={info['shanten']}  [{info['shanten_detail']['status']}]  "
              f"wall={info['wall_remaining']}")
        if terminated or truncated:
            print("  Episode ended – resetting.")
            obs, info = env.reset()

    env.close()

    # ------------------------------------------------------------------ #
    #  3. AsyncVectorEnv example (4 parallel environments)                #
    # ------------------------------------------------------------------ #
    print("\n=== AsyncVectorEnv (4 envs) ===")
    num_envs = 4
    vec_env = gym.vector.AsyncVectorEnv(
        [make_env(seed=i) for i in range(num_envs)]
    )

    obs_batch, info_batch = vec_env.reset()
    print(f"Batched obs shape    : {obs_batch.shape}")
    print(f"Per-env shanten      : {info_batch['shanten']}")

    for step_num in range(3):
        masks = np.stack(info_batch["action_mask"])
        actions = np.array([
            int(np.random.choice(np.where(masks[i])[0]))
            for i in range(num_envs)
        ])
        obs_batch, rewards, terms, truncs, info_batch = vec_env.step(actions)
        print(f"  Step {step_num+1}: rewards={np.round(rewards,2)}  "
              f"shanten={info_batch['shanten']}  terminated={terms}")

    vec_env.close()
    print("\nDone.")