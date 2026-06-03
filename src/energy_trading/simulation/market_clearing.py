from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .bid_builder import AFRRCapacityBid, DABid


@dataclass(frozen=True)
class DAClearingResult:
    submitted_buy_mw: float
    submitted_sell_mw: float
    executed_buy_mw: float
    executed_sell_mw: float
    buy_accepted: bool
    sell_accepted: bool
    reason_buy: str
    reason_sell: str


@dataclass(frozen=True)
class AFRRCapacityClearingResult:
    submitted_pos_mw: float
    submitted_neg_mw: float
    awarded_pos_mw: float
    awarded_neg_mw: float
    pos_awarded: bool
    neg_awarded: bool


@dataclass(frozen=True)
class AFRRActivationClearingResult:
    executed_rate_pos: float
    executed_rate_neg: float
    pos_accepted: bool
    neg_accepted: bool


class MarketClearingEngine:
    def __init__(
        self,
        *,
        da_mode_default: str = "price_taker",
        forecast_value_mode: str = "canonical_economic",
    ) -> None:
        self.da_mode_default = str(da_mode_default)
        self.forecast_value_mode = str(forecast_value_mode).strip().lower()
        if self.forecast_value_mode not in {"canonical_economic", "raw_signed"}:
            self.forecast_value_mode = "canonical_economic"

    def clear_afrr_capacity(
        self,
        bids: Iterable[AFRRCapacityBid],
        *,
        true_cap_pos: float,
        true_cap_neg: float,
    ) -> AFRRCapacityClearingResult:
        submitted_pos = 0.0
        submitted_neg = 0.0
        awarded_pos = 0.0
        awarded_neg = 0.0
        for b in bids:
            if b.side == "pos":
                submitted_pos += b.quantity_mw
                if float(b.capacity_price_eur_mw) <= float(true_cap_pos):
                    awarded_pos += b.quantity_mw
            elif b.side == "neg":
                submitted_neg += b.quantity_mw
                if float(b.capacity_price_eur_mw) <= float(true_cap_neg):
                    awarded_neg += b.quantity_mw
        return AFRRCapacityClearingResult(
            submitted_pos_mw=float(submitted_pos),
            submitted_neg_mw=float(submitted_neg),
            awarded_pos_mw=float(awarded_pos),
            awarded_neg_mw=float(awarded_neg),
            pos_awarded=awarded_pos > 0.0,
            neg_awarded=awarded_neg > 0.0,
        )

    def clear_da(
        self,
        bids: Iterable[DABid],
        *,
        true_da_price: float,
    ) -> DAClearingResult:
        submitted_buy = 0.0
        submitted_sell = 0.0
        executed_buy = 0.0
        executed_sell = 0.0
        buy_acc = False
        sell_acc = False
        reason_buy = "none"
        reason_sell = "none"

        for b in bids:
            mode = b.mode or self.da_mode_default
            if b.side == "buy":
                submitted_buy += b.quantity_mw
                if mode == "price_taker":
                    executed_buy += b.quantity_mw
                    buy_acc = buy_acc or b.quantity_mw > 0.0
                    reason_buy = "price_taker"
                else:
                    ok = float(b.price_eur_mwh) >= float(true_da_price)
                    if ok:
                        executed_buy += b.quantity_mw
                        buy_acc = True
                        reason_buy = "limit_cleared"
                    elif reason_buy == "none":
                        reason_buy = "limit_rejected"
            elif b.side == "sell":
                submitted_sell += b.quantity_mw
                if mode == "price_taker":
                    executed_sell += b.quantity_mw
                    sell_acc = sell_acc or b.quantity_mw > 0.0
                    reason_sell = "price_taker"
                else:
                    ok = float(b.price_eur_mwh) <= float(true_da_price)
                    if ok:
                        executed_sell += b.quantity_mw
                        sell_acc = True
                        reason_sell = "limit_cleared"
                    elif reason_sell == "none":
                        reason_sell = "limit_rejected"

        return DAClearingResult(
            submitted_buy_mw=float(submitted_buy),
            submitted_sell_mw=float(submitted_sell),
            executed_buy_mw=float(executed_buy),
            executed_sell_mw=float(executed_sell),
            buy_accepted=bool(buy_acc),
            sell_accepted=bool(sell_acc),
            reason_buy=reason_buy,
            reason_sell=reason_sell,
        )

    def clear_afrr_activation(
        self,
        bids: Iterable[AFRRCapacityBid],
        cap_res: AFRRCapacityClearingResult,
        *,
        true_act_pos: float,
        true_act_neg: float,
        true_rate_pos: float,
        true_rate_neg: float,
    ) -> AFRRActivationClearingResult:
        pos_ok = False
        neg_ok = False
        for b in bids:
            if b.side == "pos" and cap_res.pos_awarded:
                if float(b.energy_price_eur_mwh) <= float(true_act_pos):
                    pos_ok = True
            elif b.side == "neg" and cap_res.neg_awarded:
                if self.forecast_value_mode == "canonical_economic":
                    neg_clears = float(b.energy_price_eur_mwh) <= float(true_act_neg)
                else:
                    neg_clears = float(b.energy_price_eur_mwh) >= float(true_act_neg)
                if neg_clears:
                    neg_ok = True
        return AFRRActivationClearingResult(
            executed_rate_pos=float(true_rate_pos) if pos_ok else 0.0,
            executed_rate_neg=float(true_rate_neg) if neg_ok else 0.0,
            pos_accepted=bool(pos_ok),
            neg_accepted=bool(neg_ok),
        )
