from osmo.data_units import concentration_factor,time_factor,parse_dose,single_administration

def test_mass_molar_and_legacy_base_units():
    assert concentration_factor('mg/L')==1000
    assert concentration_factor('kg/l')==1e9
    assert concentration_factor('µmol/l',325.77)==325.77
    assert concentration_factor('nmol/l',325.77)==.32577
    assert concentration_factor('kg/kg',325.77) is None
    assert concentration_factor('umol/l') is None
    assert time_factor('min')*60==1

def test_dose_and_schedule_ambiguity_is_not_filled():
    assert parse_dose('200 ug/kg')==(.2,'mg/kg')
    assert parse_dose('15','mg')==(15.,'mg')
    assert parse_dose('10')==(None,None)
    assert parse_dose('10 mg daily')==(None,None)
    assert single_administration('1.0')==1
    assert single_administration('0-24-48') is None
