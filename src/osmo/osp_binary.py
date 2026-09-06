"""Read observed arrays from legacy PK-Sim SQLite projects without executing .NET deserialization."""
import base64,struct,zlib,sqlite3,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def unpack_content(blob):
    if blob[:4] != b'PK\x03\x04':
        raise ValueError('Unsupported content container')
    method=struct.unpack_from('<H',blob,8)[0]
    name_length,extra_length=struct.unpack_from('<HH',blob,26)
    start=30+name_length+extra_length
    if method != 8: raise ValueError('Expected deflate content')
    return zlib.decompress(blob[start:],-15)

def primitive_array(encoded):
    b=base64.b64decode(encoded,validate=True)
    # Narrow parser for SerializedStreamHeader + ArraySinglePrimitive + MessageEnd.
    # General BinaryFormatter deserialization is intentionally never invoked.
    if len(b)<28 or b[:17] != bytes.fromhex('0001000000ffffffff0100000000000000') or b[17]!=15 or b[-1]!=11:
        raise ValueError('Unsupported serialized array')
    n=struct.unpack_from('<i',b,22)[0]
    dtype={11:'<f4',6:'<f8'}.get(b[26])
    if dtype is None or n<0 or len(b)!=28+n*np.dtype(dtype).itemsize:
        raise ValueError('Invalid primitive array size/type')
    return np.frombuffer(b,dtype=dtype,count=n,offset=27).astype(float).tolist()

def observed_repositories(path):
    con=sqlite3.connect('file:'+Path(path).resolve().as_posix()+'?mode=ro',uri=True)
    try:
        records=con.execute('SELECT d.Id,d.Name,c.Data FROM OBSERVED_DATA o JOIN DATA_REPOSITORIES d ON o.DataRepositoryId=d.Id JOIN CONTENTS c ON d.ContentId=c.Id').fetchall()
    finally:con.close()
    for identifier,name,blob in records:
        root=ET.fromstring(unpack_content(blob))
        meta={e.get('name'):e.get('value') for e in root.findall('./ExtendedProperties/*')}
        grids={e.get('id'):e for e in root.findall('./Columns/McBaseGrid')}
        for col in root.findall('./Columns/McDataColumn'):
            info=col.find('DataInfo')
            if info is None or info.get('origin')!='Observation' or info.get('auxiliaryType')!='Undefined':continue
            grid=grids[col.get('baseGrid')]
            t,y=primitive_array(grid.findtext('Values')),primitive_array(col.findtext('Values'))
            if len(t)!=len(y):raise ValueError('Mismatched observation array lengths')
            dimension=col.get('dimension')
            # Base units documented by OSP; displayUnitName is NOT the stored unit.
            unit={'Concentration (mass)':'kg/l','Concentration (molar)':'umol/l','Mass per tissue':'kg/kg'}.get(dimension,'unsupported:'+str(dimension))
            mw=info.get('molWeight')
            yield dict(identifier=identifier,name=name,column=col.get('name'),metadata=meta,
                       time=t,value=y,time_unit='min',unit=unit,dimension=dimension,
                       molecular_weight=float(mw)*1e9 if mw else None,
                       display_unit=info.get('displayUnitName'),source_field=info.get('source'))
