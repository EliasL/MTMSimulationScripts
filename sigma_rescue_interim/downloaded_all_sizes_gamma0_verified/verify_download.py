"""Verify downloaded drops against current gamma=0 convention; audit yield.

Does not regenerate source stresses or claim unavailable audited coverage.
Run from repository root using the project interpreter and PYTHONPATH=.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from Plotting.energyDropCalculations import _simple_shear_tangent_gamma0
from Plotting.standardPowerlaw import derive_sigma_rescue_yield_peaks

SOURCE = Path('/Users/elias/Downloads/drops_usable.csv.gz')
OUT = Path(__file__).resolve().parent / 'tables'
mu = float(_simple_shear_tangent_gamma0((1,))[0])
cols = ['run_name', 'size', 'seed', 'load_i', 'load_ip1', 'load_step_i',
        'load_step_ip1', 'delta_gamma', 'total_energy_i', 'total_energy_ip1',
        'total_energy_change_ip1', 'total_e_change_from_init_ip1',
        'avg_sigma12_i', 'avg_sigma12_ip1', 'reference_volume',
        'delta_E_I', 'delta_E_R', 'delta_E_S', 'usable', 'common_sigma_provider']
parts=[]
errors={key:0.0 for key in ['delta_E_I','delta_E_R','delta_E_S']}
counts={}
for c in pd.read_csv(SOURCE, usecols=cols, chunksize=250000):
    assert c.usable.eq(True).all()
    assert np.isfinite(c.select_dtypes(include='number')).all().all()
    assert (c[['avg_sigma12_i','avg_sigma12_ip1']] != -1).all().all()
    assert c.common_sigma_provider.notna().all()
    assert (c.load_step_ip1-c.load_step_i).eq(1).all()
    assert np.allclose(c.load_ip1-c.load_i,c.delta_gamma,rtol=1e-9,atol=1e-12)
    assert (c.delta_gamma>0).all() and np.allclose(c.reference_volume,c['size']**2)
    expected={
        'delta_E_I': -c.total_energy_change_ip1,
        'delta_E_R': -c.total_e_change_from_init_ip1,
        'delta_E_S': -(c.total_energy_ip1-(c.total_energy_i+c.reference_volume*c.avg_sigma12_i*c.delta_gamma+0.5*c.reference_volume*mu*c.delta_gamma**2)),
    }
    for key,value in expected.items():
        err=float(np.max(np.abs(c[key]-value)))
        errors[key]=max(errors[key],err)
        assert np.allclose(c[key],value,rtol=1e-10,atol=2e-10), (key,err)
    for size,n in c.groupby('size').size().items(): counts[int(size)]=counts.get(int(size),0)+int(n)
    parts.append(c[['run_name','size','seed','load_ip1','avg_sigma12_ip1','usable']])
frame=pd.concat(parts,ignore_index=True)
assert not frame.duplicated(['run_name','load_ip1']).any()
peaks=derive_sigma_rescue_yield_peaks(audited=frame)
OUT.mkdir(exist_ok=True)
peaks.to_csv(OUT/'yield_peaks.csv',index=False)
target=OUT/'drops_usable.csv.gz'
if not target.exists(): target.symlink_to(SOURCE)
with SOURCE.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
manifest={
 'delta_E_S_tangent':'gamma0', 'source':str(SOURCE), 'source_sha256':digest,
 'verification':'Every downloaded row checked against gamma=0 tangent and starting-step avg_sigma12; Delta E_I/R checked against stored energy changes. Raw stresses not independently recomputed from VTUs.',
 'gamma0_tangent':mu,'max_absolute_errors':errors,'usable_transition_count':len(frame),
 'rows_by_size':counts,'run_count':len(peaks),'established_yield_peak_count':int(peaks.peak_established.sum()),
 'yield_audit_source':'Downloaded usable transitions; standard >=100 valid samples on each side within 0.01 load of peak. Raw full stress curves unavailable.',
 'audited_transition_count':None,'stress_coverage_fraction_available':False,
}
# Attach the original audit when available; never fabricate missing transitions.
audit_source = SOURCE.parent / 'tables'
if (audit_source / 'drops_all_audited.csv.gz').is_file():
    original = json.loads((audit_source / 'build_manifest.json').read_text())
    assert original['delta_E_S_tangent'] == 'gamma0'
    with (audit_source / 'drops_usable.csv.gz').open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == digest
    audited = pd.read_csv(audit_source / 'drops_all_audited.csv.gz',
        usecols=['run_name','size','seed','load_ip1','avg_sigma12_ip1','usable'])
    assert len(audited) == original['audited_transition_count']
    assert int(audited.usable.sum()) == original['usable_transition_count']
    audited_peaks = derive_sigma_rescue_yield_peaks(audited=audited)
    pd.testing.assert_frame_equal(audited_peaks, peaks, check_exact=False, rtol=1e-10)
    supplied = pd.read_csv(audit_source / 'yield_peaks.csv')
    pd.testing.assert_frame_equal(audited_peaks, supplied, check_exact=False, rtol=1e-10)
    target = OUT / 'drops_all_audited.csv.gz'
    if not target.exists(): target.symlink_to(audit_source / 'drops_all_audited.csv.gz')
    shutil.copy2(audit_source / 'build_manifest.json', OUT / 'source_build_manifest.json')
    manifest.update(original_build_manifest=str(audit_source / 'build_manifest.json'),
        audited_source=str(audit_source / 'drops_all_audited.csv.gz'),
        audited_transition_count=len(audited), stress_coverage_fraction_available=True,
        yield_audit_source='Recomputed from original all-audited transition table; matches supplied yield audit.')
(OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2),flush=True)
print(peaks.to_string(index=False),flush=True)
