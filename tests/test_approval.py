from pathlib import Path
import pytest
from vdie_moe_sim.config import ROOT
from vdie_moe_sim.run_all import verify_approval

def test_current_parameter_version_is_explicitly_approved():
    approval=verify_approval(ROOT/'VDIE_MOE_SIM_SPEC.md')
    assert approval['approval_message']=='用户：确认通过'

def test_altered_specification_cannot_inherit_existing_approval(tmp_path):
    original=(ROOT/'VDIE_MOE_SIM_SPEC.md').read_text()
    changed=tmp_path/'changed_spec.md'
    changed.write_text(original.replace('fp8_nrmse_limit: 0.05','fp8_nrmse_limit: 0.1'))
    assert changed.read_text()!=original
    with pytest.raises(PermissionError,match='user-approved'):
        verify_approval(changed)
