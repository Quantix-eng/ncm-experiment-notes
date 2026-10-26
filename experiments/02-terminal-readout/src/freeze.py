from base import *
def main():
    assert not (D/'results/protocol_frozen.json').exists(),'Do not refreeze after outcomes'
    roots=['','directions_mecanistes','generalisation_architecturale','correction_pooling','dynamique_temporelle','mecanismes_revision','compensation_conditionnelle']
    prior={str(Path(r)/'results/final_manifest.json'):sha(ROOT/r/'results/final_manifest.json') for r in roots}
    write(D/'sources/previous_manifests.json',prior)
    write(D/'results/protocol_frozen.json',dict(protocol=Q,protocol_sha256=sha(D/'protocol.json'),objectives_sha256=sha(D/'PROTOCOLE_GELE.md'),tables_sha256=sha(D/'results/discovery_tables.json'),pilot_sha256=sha(D/'results/discovery_pilot.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    log('positive_study_frozen')
if __name__=='__main__':main()
