"""Read a vanilla structure template to identify the target release format."""
import io
import struct


def read_nbt(blob):
    stream=io.BytesIO(blob)
    def number(fmt):
        return struct.unpack('>'+fmt, stream.read(struct.calcsize('>'+fmt)))[0]
    def string():
        return stream.read(number('H')).decode('utf8',errors='replace')
    def tag(kind):
        if 1<=kind<=6: return number('bhiqfd'[kind-1])
        if kind==7: return stream.read(number('i'))
        if kind==8: return string()
        if kind==9:
            subtype=number('B')
            return [tag(subtype) for _ in range(number('i'))]
        if kind==10:
            result={}
            while subtype:=number('B'):
                name=string()
                result[name]=tag(subtype)
            return result
        if kind in (11,12): return [number('i' if kind==11 else 'q') for _ in range(number('i'))]
        raise ValueError(f'Unexpected NBT type {kind}')
    root=number('B')
    string()
    return tag(root)
