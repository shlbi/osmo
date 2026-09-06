import base64,struct,zlib
import pytest
from osmo.osp_binary import primitive_array,unpack_content

def test_narrow_primitive_array_parser_and_rejection():
    header=bytes.fromhex('0001000000ffffffff0100000000000000')
    raw=header+b'\x0f'+struct.pack('<iiB',1,3,11)+struct.pack('<fff',.5,1.,12.)+b'\x0b'
    assert primitive_array(base64.b64encode(raw).decode())==[.5,1.,12.]
    with pytest.raises(ValueError):primitive_array(base64.b64encode(raw[:-2]).decode())
    with pytest.raises(ValueError):primitive_array(base64.b64encode(b'arbitrary object stream').decode())

def test_local_deflate_container_without_central_directory():
    xml=b'<observed><value>3.5</value></observed>'
    compressor=zlib.compressobj(wbits=-15);payload=compressor.compress(xml)+compressor.flush()
    header=bytearray(30);header[:4]=b'PK\x03\x04';struct.pack_into('<H',header,8,8);struct.pack_into('<HH',header,26,4,0)
    assert unpack_content(bytes(header)+b'Name'+payload)==xml
