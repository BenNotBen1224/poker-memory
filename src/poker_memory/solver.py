"""River solver: the equilibrium of a heads-up river betting tree, every hand at once.

Chips are counted from the start of the river: the pot P is already in the middle, and a
player's result is what he gets back minus what he puts in on the river. Results always add
up to P, so the game is constant-sum and "distance from perfect play" is well defined.

Method: discounted counterfactual regret minimisation (DCFR, alpha 1.5, beta 0, gamma 2) with
alternating updates, over vectors of hands instead of one hand at a time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .handrank import COMBOS, Showdown, board_strength, touching

OOP, IP = 0, 1  # OOP acts first on the river


@dataclass(frozen=True)
class Sizes:
    bets: tuple[float, ...] = (0.33, 0.75)  # opening bets as shares of the pot, plus all-in
    allin: bool = True
    max_raises: int = 2
    first_can_bet: bool = True  # False only for textbook games where the first player must check
    extra: tuple[int, ...] = ()  # river totals actually used, so the real line is in the tree


@dataclass
class Node:
    actor: int  # OOP, IP, or -1 at the end of the hand
    put: tuple[int, int]  # chips each player has put in on the river so far
    moves: list[tuple[str, int]] = field(default_factory=list)  # (verb, river total after the move)
    kids: list[int] = field(default_factory=list)
    folder: int | None = None  # end of hand only: who folded; None means showdown


def _amounts(low: int, stack: int, pot_now: int, shares: tuple[float, ...], sizes: Sizes) -> list[int]:
    """River totals above `low` a player may bet or raise to. Real sizes win over generated ones within 10%."""
    picks = sorted({x for x in sizes.extra if low < x <= stack})
    if sizes.allin and stack > low:
        picks.append(stack)
    for share in shares:
        x = min(low + round(share * pot_now), stack)
        if x > low and all(abs(x - y) > 0.1 * y for y in picks):
            picks.append(x)
    return sorted(set(picks))


def build_tree(pot: int, stack: int, sizes: Sizes) -> list[Node]:
    nodes: list[Node] = []

    def add(node: Node) -> int:
        nodes.append(node)
        return len(nodes) - 1

    def grow(actor: int, put: tuple[int, int], raises: int) -> int:
        i = add(Node(actor=actor, put=put))
        other = 1 - actor

        def move(verb: str, to: int, kid: int) -> None:
            nodes[i].moves.append((verb, to))
            nodes[i].kids.append(kid)

        def with_put(to: int) -> tuple[int, int]:
            return (to, put[1]) if actor == OOP else (put[0], to)

        if put[actor] == put[other]:
            move("check", put[actor], add(Node(actor=-1, put=put)) if actor == IP else grow(IP, put, raises))
            if actor == IP or sizes.first_can_bet:
                for to in _amounts(put[other], stack, pot + sum(put), sizes.bets, sizes):
                    move("bet", to, grow(other, with_put(to), raises))
        else:
            move("fold", put[actor], add(Node(actor=-1, put=put, folder=actor)))
            move("call", put[other], add(Node(actor=-1, put=(put[other], put[other]))))
            if raises < sizes.max_raises and put[other] < stack:
                for to in _amounts(put[other], stack, pot + sum(put), (), sizes):
                    move("raise", to, grow(other, with_put(to), raises + 1))
        return i

    grow(OOP, (0, 0), 0)
    return nodes


@dataclass
class Solution:
    strategy: dict[int, np.ndarray]  # decision node -> (hands of its actor, moves), equilibrium mix
    exploitability: float  # how much a perfect opponent could still win, as a share of the pot
    iterations: int
    seconds: float


class River:
    def __init__(self, board: list[int], pot: int, stack: int, ranges: tuple[np.ndarray, np.ndarray], sizes: Sizes):
        self.pot, self.stack = pot, stack
        self.nodes = build_tree(pot, stack, sizes)
        live = ~touching(board)
        self.hands = [np.flatnonzero((r > 0) & live) for r in ranges]
        self.prior = [ranges[p][self.hands[p]].astype(np.float32) for p in (OOP, IP)]
        self._showdown = Showdown(board_strength(board))
        self.decisions = [i for i, n in enumerate(self.nodes) if n.actor >= 0]
        self.ends = [i for i, n in enumerate(self.nodes) if n.actor < 0]
        self._showdowns = np.array([j for j, e in enumerate(self.ends) if self.nodes[e].folder is None])
        self._mass = float(self.prior[OOP] @ self._against(OOP, self.prior[IP][:, None])[1][:, 0])
        # Chips at each end of the hand for player p: a * (wins - losses) + b * (opponent hands faced).
        self._coef = [np.zeros((2, len(self.ends)), np.float32) for _ in (OOP, IP)]
        for j, e in enumerate(self.ends):
            n = self.nodes[e]
            for p in (OOP, IP):
                if n.folder is None:
                    self._coef[p][:, j] = (pot / 2 + n.put[p], pot / 2)
                else:
                    f = n.folder
                    self._coef[p][1, j] = -n.put[f] if p == f else pot + n.put[f]

    def reach(self, sigma: dict[int, np.ndarray]) -> list[list[np.ndarray]]:
        reach: list = [None] * len(self.nodes)
        reach[0] = list(self.prior)
        for i in self.decisions:
            n = self.nodes[i]
            for a, kid in enumerate(n.kids):
                reach[kid] = list(reach[i])
                reach[kid][n.actor] = reach[i][n.actor] * sigma[i][:, a]
        return reach

    def _against(self, p: int, theirs: np.ndarray, ranked: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """For each of p's hands and each column of opponent weights `theirs` (his hands, k):
        (weight it beats minus weight it loses to, weight it faces), skipping hands sharing a card.
        Only the columns in `ranked` (all by default) get the first number; the rest are left at zero."""
        full = np.zeros((len(COMBOS), theirs.shape[1]), np.float32)
        full[self.hands[1 - p]] = theirs
        h = self.hands[p]
        faced = Showdown.faced(full)[h]
        ranked = slice(None) if ranked is None else ranked
        win, tie, _ = self._showdown.counts(full[:, ranked])
        net = np.zeros_like(faced)
        net[:, ranked] = 2 * win[h] + tie[h] - faced[:, ranked]
        return net, faced

    def _values(self, p: int, sigma: dict[int, np.ndarray], reach: list, best: bool = False) -> list[np.ndarray]:
        """Counterfactual chips for each of p's hands at every node, given the opponent's reach."""
        net, faced = self._against(p, np.column_stack([reach[e][1 - p] for e in self.ends]), self._showdowns)
        ends = self._coef[p][0] * net + self._coef[p][1] * faced
        v: list = [None] * len(self.nodes)
        for j, e in enumerate(self.ends):
            v[e] = ends[:, j]
        for i in reversed(self.decisions):
            n = self.nodes[i]
            kids = np.column_stack([v[c] for c in n.kids])
            if n.actor != p:
                v[i] = kids.sum(axis=1)
            else:
                v[i] = kids.max(axis=1) if best else (sigma[i] * kids).sum(axis=1)
        return v

    def _uniform(self) -> dict[int, np.ndarray]:
        return {i: np.full((len(self.hands[self.nodes[i].actor]), len(self.nodes[i].kids)),
                           1 / len(self.nodes[i].kids), np.float32) for i in self.decisions}

    def exploitability(self, sigma: dict[int, np.ndarray]) -> float:
        reach = self.reach(sigma)
        best = sum(self.prior[p] @ self._values(p, sigma, reach, best=True)[0] for p in (OOP, IP))
        return float((best / self._mass - self.pot) / 2 / self.pot)

    def solve(self, target: float = 0.005, max_iterations: int = 2000, check_every: int = 10) -> Solution:
        start = time.perf_counter()
        sigma = self._uniform()
        regret = {i: np.zeros_like(s) for i, s in sigma.items()}
        total = {i: np.zeros_like(s) for i, s in sigma.items()}

        def normalize(x: np.ndarray) -> np.ndarray:
            norm = x.sum(axis=1, keepdims=True)
            return np.divide(x, norm, out=np.full_like(x, 1 / x.shape[1]), where=norm > 0)

        def average() -> dict[int, np.ndarray]:
            return {i: normalize(s) for i, s in total.items()}

        t, gap = 0, float("inf")
        while t < max_iterations:
            t += 1
            keep, fade = np.float32(t**1.5 / (t**1.5 + 1)), np.float32((t / (t + 1)) ** 2)
            for p in (OOP, IP):
                reach = self.reach(sigma)
                v = self._values(p, sigma, reach)
                for i in self.decisions:
                    if self.nodes[i].actor != p:
                        continue
                    kids = np.column_stack([v[c] for c in self.nodes[i].kids])
                    total[i] = total[i] * fade + reach[i][p][:, None] * sigma[i]
                    regret[i] = regret[i] * np.where(regret[i] > 0, keep, np.float32(0.5)) + kids - v[i][:, None]
                    sigma[i] = normalize(np.maximum(regret[i], 0))
            if t % check_every == 0:
                gap = self.exploitability(average())
                if gap <= target:
                    break
        avg = average()
        if t % check_every:
            gap = self.exploitability(avg)
        return Solution(strategy=avg, exploitability=gap, iterations=t, seconds=time.perf_counter() - start)

    def move_values(self, p: int, combo: int, node: int, sigma: dict[int, np.ndarray]) -> list[float]:
        """Average chips (from the start of the river) for one hand after each move at `node`,
        when both players then follow `sigma`."""
        reach = self.reach(sigma)
        v = self._values(p, sigma, reach)
        j = int(np.flatnonzero(self.hands[p] == combo)[0])
        faced = reach[node][1 - p] @ ~touching(COMBOS[combo])[self.hands[1 - p]]
        return [float(v[c][j] / faced) for c in self.nodes[node].kids]

    def mix(self, node: int, sigma: dict[int, np.ndarray], combos: np.ndarray | None = None) -> np.ndarray:
        """How often each move is chosen at `node`, over the hands that get there (optionally only `combos`)."""
        p = self.nodes[node].actor
        weight = self.reach(sigma)[node][p]
        if combos is not None:
            weight = weight * np.isin(self.hands[p], combos)
        return weight @ sigma[node] / weight.sum()
