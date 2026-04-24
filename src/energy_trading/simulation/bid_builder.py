from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class DABid:
    ts: pd.Timestamp
    side: str  # "buy" | "sell"
    quantity_mw: float
    price_eur_mwh: float
    mode: str  # "price_taker" | "limit"
    reason: str = ""


@dataclass(frozen=True)
class AFRRCapacityBid:
    ts: pd.Timestamp
    side: str  # "pos" | "neg"
    quantity_mw: float
    capacity_price_eur_mw: float
    energy_price_eur_mwh: float


@dataclass(frozen=True)
class AFRREnergyBid:
    ts: pd.Timestamp
    side: str  # "pos" | "neg"
    quantity_mw: float
    energy_price_eur_mwh: float
    gate_closure_min_before_delivery: int = 25


@dataclass(frozen=True)
class BidPricingPolicy:
    cap_risk_lambda: float = 0.2
    act_risk_lambda: float = 0.2
    da_buy_limit_price_eur_mwh: float = 3000.0
    da_sell_limit_price_eur_mwh: float = -500.0

    def _center(self, pred: float, q50: float | None) -> float:
        return float(q50) if q50 is not None and math.isfinite(q50) else float(pred)

    def _spread(self, q10: float | None, q90: float | None) -> float:
        if q10 is None or q90 is None or not math.isfinite(q10) or not math.isfinite(q90):
            return 0.0
        return max(0.0, float(q90) - float(q10))

    def capacity_price(
        self,
        *,
        pred: float,
        q10: float | None = None,
        q50: float | None = None,
        q90: float | None = None,
    ) -> float:
        center = self._center(pred, q50)
        spread = self._spread(q10, q90)
        return max(0.0, center - self.cap_risk_lambda * spread)

    def energy_price(
        self,
        *,
        side: str,
        pred: float,
        mc_pos: float,
        mc_neg: float,
        q10: float | None = None,
        q50: float | None = None,
        q90: float | None = None,
    ) -> float:
        center = self._center(pred, q50)
        spread = self._spread(q10, q90)
        if side == "pos":
            return max(float(mc_pos), center - self.act_risk_lambda * spread)
        return min(center + self.act_risk_lambda * spread, -float(mc_neg))


class BidBuilder:
    def __init__(
        self,
        *,
        pricing_policy: BidPricingPolicy,
        da_step_mw: float,
        afrr_step_mw: float,
        eta_in: float,
        eta_out: float,
        degradation_cost_eur_mwh: float,
        transaction_cost_eur_mwh: float,
        da_mode: str = "price_taker",
        da_arb_mode: str = "limit",
        afrr_energy_bid_strategy: str = "forecast",
        link_da_to_awarded_afrr: bool = True,
    ) -> None:
        self.pricing = pricing_policy
        self.da_step_mw = float(da_step_mw)
        self.afrr_step_mw = float(afrr_step_mw)
        self.eta_in = float(eta_in)
        self.eta_out = float(eta_out)
        self.mc_pos = float(degradation_cost_eur_mwh) + float(transaction_cost_eur_mwh)
        self.mc_neg = float(degradation_cost_eur_mwh) + float(transaction_cost_eur_mwh)
        self.da_mode = da_mode
        self.da_arb_mode = da_arb_mode
        self.afrr_energy_bid_strategy = afrr_energy_bid_strategy
        self.link_da_to_awarded_afrr = bool(link_da_to_awarded_afrr)

    def _qfloor(self, x: float, step: float) -> float:
        if x <= 0.0:
            return 0.0
        units = int((x / step) + 1e-12)
        return float(units * step)

    def build_afrr_capacity_bids(
        self,
        *,
        ts: pd.Timestamp,
        reserve_pos_mw: float,
        reserve_neg_mw: float,
        pred_cap_pos: float,
        pred_cap_neg: float,
        pred_act_pos: float,
        pred_act_neg: float,
        is_oracle: bool = False,
    ) -> list[AFRRCapacityBid]:
        bids: list[AFRRCapacityBid] = []
        q_pos = self._qfloor(float(reserve_pos_mw), self.afrr_step_mw)
        q_neg = self._qfloor(float(reserve_neg_mw), self.afrr_step_mw)
        if q_pos > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="pos",
                    quantity_mw=q_pos,
                    # Capacity is settled pay-as-bid: in oracle mode (pred=true),
                    # bid at forecast/true capacity price to both clear and keep
                    # economically consistent remuneration.
                    capacity_price_eur_mw=(
                        float(pred_cap_pos)
                        if is_oracle
                        else self.pricing.capacity_price(pred=float(pred_cap_pos))
                    ),
                    energy_price_eur_mwh=self.pricing.energy_price(
                        side="pos",
                        pred=float(pred_act_pos),
                        mc_pos=self.mc_pos,
                        mc_neg=self.mc_neg,
                    ),
                )
            )
        if q_neg > 0.0:
            bids.append(
                AFRRCapacityBid(
                    ts=ts,
                    side="neg",
                    quantity_mw=q_neg,
                    capacity_price_eur_mw=(
                        float(pred_cap_neg)
                        if is_oracle
                        else self.pricing.capacity_price(pred=float(pred_cap_neg))
                    ),
                    energy_price_eur_mwh=self.pricing.energy_price(
                        side="neg",
                        pred=float(pred_act_neg),
                        mc_pos=self.mc_pos,
                        mc_neg=self.mc_neg,
                    ),
                )
            )
        return bids

    def dynamic_afrr_energy_price(
        self,
        *,
        side: str,
        pred_act_price: float,
        soc_now_mwh: float,
        soc_min_mwh: float,
        soc_max_mwh: float,
        obligation_mw: float = 0.0,
        delivery_duration_h: float = 1.0,
        q10: float | None = None,
        q50: float | None = None,
        q90: float | None = None,
        is_oracle: bool = False,
        true_act_price: float | None = None,
    ) -> float:
        """T-25 dynamic energy bid update based on latest forecast + current SoC."""
        # Defensive bidding: if physical delivery capability is insufficient,
        # bid out of merit order to avoid activation/default penalties.
        ob = max(0.0, float(obligation_mw))
        dt = max(1e-12, float(delivery_duration_h))
        if side == "pos":
            required_internal_mwh = ob * dt / max(self.eta_out, 1e-12)
            available_internal_mwh = max(0.0, float(soc_now_mwh) - float(soc_min_mwh))
            if available_internal_mwh + 1e-9 < required_internal_mwh:
                return 9999.0
        else:
            required_headroom_mwh = ob * dt * self.eta_in
            available_headroom_mwh = max(0.0, float(soc_max_mwh) - float(soc_now_mwh))
            if available_headroom_mwh + 1e-9 < required_headroom_mwh:
                return -9999.0

        if is_oracle and true_act_price is not None and math.isfinite(float(true_act_price)):
            return float(true_act_price)

        strategy = str(self.afrr_energy_bid_strategy).lower().strip()
        if strategy == "forecast":
            base = float(pred_act_price)
        elif strategy == "marginal_cost":
            base = float(self.mc_pos if side == "pos" else -self.mc_neg)
        else:  # hybrid
            base = self.pricing.energy_price(
                side=side,
                pred=float(pred_act_price),
                mc_pos=self.mc_pos,
                mc_neg=self.mc_neg,
                q10=q10,
                q50=q50,
                q90=q90,
            )
        denom = max(1e-12, float(soc_max_mwh) - float(soc_min_mwh))
        soc_ratio = (float(soc_now_mwh) - float(soc_min_mwh)) / denom
        soc_ratio = max(0.0, min(1.0, soc_ratio))
        # Mild state-dependent urgency adjustment (EUR/MWh).
        # pos (discharge): low SoC -> more conservative (higher ask).
        # neg (charge): high SoC -> more conservative (higher ask).
        if side == "pos":
            adj = (0.5 - soc_ratio) * 20.0
        else:
            adj = (soc_ratio - 0.5) * 20.0
        return float(base + adj)

    def build_da_bids_from_plan(
        self,
        *,
        ts: pd.Timestamp,
        planned_charge_mw: float,
        planned_discharge_mw: float,
        obligation_pos_mw: float,
        obligation_neg_mw: float,
        pred_da_price: float,
        is_oracle: bool = False,
    ) -> list[DABid]:
        # Use MILP schedule volumes directly (endogenous physics/hedging).
        # BidBuilder only sets execution mode/price policy.
        plan_buy_mw = self._qfloor(max(0.0, float(planned_charge_mw)), self.da_step_mw)
        plan_sell_mw = self._qfloor(max(0.0, float(planned_discharge_mw)), self.da_step_mw)

        bids: list[DABid] = []
        if plan_buy_mw > 0.0:
            buy_is_hedge = self.link_da_to_awarded_afrr and float(obligation_pos_mw) > 0.0
            buy_mode = "price_taker" if (is_oracle or buy_is_hedge) else self.da_arb_mode
            buy_price = (
                self.pricing.da_buy_limit_price_eur_mwh if buy_mode == "price_taker" else float(pred_da_price)
            )
            bids.append(
                DABid(
                    ts=ts,
                    side="buy",
                    quantity_mw=plan_buy_mw,
                    price_eur_mwh=buy_price,
                    mode=buy_mode,
                    reason="afrr_hedge" if buy_is_hedge else "da_arbitrage",
                )
            )
        if plan_sell_mw > 0.0:
            sell_is_hedge = self.link_da_to_awarded_afrr and float(obligation_neg_mw) > 0.0
            sell_mode = "price_taker" if (is_oracle or sell_is_hedge) else self.da_arb_mode
            sell_price = (
                self.pricing.da_sell_limit_price_eur_mwh if sell_mode == "price_taker" else float(pred_da_price)
            )
            bids.append(
                DABid(
                    ts=ts,
                    side="sell",
                    quantity_mw=plan_sell_mw,
                    price_eur_mwh=sell_price,
                    mode=sell_mode,
                    reason="afrr_hedge" if sell_is_hedge else "da_arbitrage",
                )
            )
        return bids
