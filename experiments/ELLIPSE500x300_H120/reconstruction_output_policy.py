"""Future output scope; historical frozen releases remain read-only."""

POLICY_ID = 'separate_218_440_20261010'
EHE_CHANNELS = ('440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected')
JSCC_CHANNELS = EHE_CHANNELS + ('440_ComptonOnly', '440_SinglePlusCompton')
RETIRED_SUM_CHANNELS = ('440SinglePlus218Single', '440SingleComptonPlus218Single')


def require_separate_channels(config, system='EHE'):
    """Fail closed before a new solve if an obsolete sum contract is supplied."""
    expected = EHE_CHANNELS if system == 'EHE' else JSCC_CHANNELS if system == 'JSCC' else None
    if expected is None:
        raise ValueError('Unknown reconstruction system')
    if config.get('output_policy') != POLICY_ID or config.get('output_channels') != list(expected):
        raise ValueError('New executions require a separately frozen no-cross-energy-sum contract')
    return expected


def accepted_channels(run, config, legacy_channels):
    """Read-only acceptance may still inspect the exact historical output scope."""
    if 'output_policy' in config:
        expected = require_separate_channels(config)
        if run.get('output_policy') != config['output_policy']:
            raise ValueError('Recorded output policy differs from its frozen configuration')
    else:
        expected = tuple(legacy_channels)
    if run['channels'] != list(expected):
        raise ValueError('Recorded output scope differs from its frozen configuration')
    return expected
