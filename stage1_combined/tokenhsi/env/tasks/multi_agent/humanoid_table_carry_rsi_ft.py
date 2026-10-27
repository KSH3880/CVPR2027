"""Unchanged HOLDING_AT reward/78k inputs, opt-in mesh-adjusted Table carry RSI."""
import os
from env.tasks.multi_agent.humanoid_table_hold_at_ft import HumanoidTableHoldAtFT


class HumanoidTableCarryRsiFT(HumanoidTableHoldAtFT):
    def _reset_boxes(self, env_ids):
        super()._reset_boxes(env_ids)
        enabled=not self._is_eval or self.cfg['env']['tableCarryRsi'].get('audit',False) or os.environ.get('TABLE_RSI_VIEW')=='1'
        if enabled:
            if not hasattr(self,'_table_carry_rsi'):
                from utils.table_carry_rsi import TableCarryRsi
                self._table_carry_rsi=TableCarryRsi(self)
            self._table_carry_rsi.apply(env_ids)

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)
        if hasattr(self,'_table_carry_rsi'):self._table_carry_rsi.init_history(env_ids)

    def consume_relation_diagnostics(self):
        result=super().consume_relation_diagnostics()
        if hasattr(self,'_table_carry_rsi'):
            pool=self._table_carry_rsi
            result['table_rsi/active_fraction']=pool.active.float().mean()
            result['table_rsi/requested']=pool.requested
            result['table_rsi/applied']=pool.applied
            result['table_rsi/explicit_loco_fallbacks']=pool.missing
        return result
