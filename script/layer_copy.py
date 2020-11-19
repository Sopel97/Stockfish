import sys
import os
import struct
import io

class DatabaseEntry:
    def __init__(self, size, name):
        self.size = size
        self.name = name

class Database:
    def __init__(self):
        self.database = dict()

    def __getitem__(self, key):
        return self.database[key]

    def __contains__(self, key):
        return key in self.database

    def add(self, hash, entries):
        self.database[hash] = entries

feature_set_database = Database()
feature_set_database.add(0x5d69d7b8, [DatabaseEntry(256*2 + 41024*256*2, 'FeatureSet[HalfKP]')])

layer_stack_database = Database()
layer_stack_database.add(0x63337156, [
    DatabaseEntry(512*32*1 + 32*4, 'AffineTransform[512->32]'),
    DatabaseEntry(32*32*1 + 32*4, 'AffineTransform[32->32]'),
    DatabaseEntry(32*1*1 + 1*4, 'AffineTransform[32->1]')
    ])

# 32 due to padding on input dimension
layer_stack_database.add(0x299148c2, [
    DatabaseEntry(128*16*1 + 16*4, 'AffineTransform[128->16]'),
    DatabaseEntry(32*16*1 + 16*4, 'AffineTransform[16->16]'),
    DatabaseEntry(32*1*1 + 1*4, 'AffineTransform[16->1]')
    ])




class Layer:
    def __init__(self, data, offset, size, name):
        self.data = data
        self.offset = offset
        self.size = size
        self.name = name

    def __str__(self):
        return self.name + " at " + str(self.offset) + ":" + str(self.offset + self.size)

    def __repr__(self):
        return self.__str__()

class Network:
    def __init__(self, data, version):
        self.data = data
        self.version = version
        self.layers = []
        self.fully_known = True

    def add_layer(self, layer):
        self.layers.append(layer)

    def __str__(self):
        return '\n'.join(str(l) for l in self.layers) + ("" if self.fully_known else "\nmore unknown layers...")

    def __repr__(self):
        return self.__str__()

def get_compatible_layer_pairs(lhs_net, rhs_net):
    compatible_layer_pairs = []
    for lhs_layer in lhs_net.layers:
        for rhs_layer in rhs_net.layers:
            if lhs_layer.size == rhs_layer.size and lhs_layer.name == rhs_layer.name:
                compatible_layer_pairs.append((lhs_layer, rhs_layer))
    return compatible_layer_pairs

def read_int32(stream, expected=None):
    b = stream.read(4)
    if len(b) != 4:
        return None
    v = struct.unpack("<i", b)[0]
    if expected is not None and v != expected:
        raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def parse_feature_transformer(stream):
    hash = read_int32(stream)
    if not hash in feature_set_database:
        raise Exception("Unknown feature set with hash " + hex(hash) + " at " + str(stream.tell() - 4))

    feature_set = feature_set_database[hash][0]
    offset = stream.tell()
    stream.read(feature_set.size)
    return Layer(stream.getbuffer(), offset, feature_set.size, feature_set.name)

def parse_layers(stream):
    hash = read_int32(stream)
    if hash is None:
        return []

    if not hash in layer_stack_database:
        raise Exception("Unknown layer stack with hash " + hex(hash) + " at " + str(stream.tell() - 4))

    layer_stack = layer_stack_database[hash]
    layers = []
    offset = stream.tell()
    for layer_proto in layer_stack:
        layers.append(Layer(stream.getbuffer(), offset, layer_proto.size, layer_proto.name))
        stream.read(layer_proto.size)
        offset += layer_proto.size

    return layers

def parse_network(data):
    stream = io.BytesIO(data)

    version = read_int32(stream)
    hash = read_int32(stream)
    desc_size = read_int32(stream)
    stream.read(desc_size)

    net = Network(data, version)
    feature_transformer = parse_feature_transformer(stream)
    net.add_layer(feature_transformer)

    try:
        while True:
            layers = parse_layers(stream)
            if not layers:
                break

            for layer in layers:
                net.add_layer(layer)
    except Exception:
        net.fully_known = False

    return net

def copy_layer(data, layer_from, layer_to):
    assert layer_from.size == layer_to.size
    size = layer_from.size
    return data[:layer_to.offset] + layer_from.data[layer_from.offset:layer_from.offset+size] + data[layer_to.offset+size:]

if len(sys.argv) < 4:
    print("Usage: python layer_copy.py net_to_copy_from net_to_copy_to net_net_name")

from_net_path = sys.argv[1]
to_net_path = sys.argv[2]
new_net_path = sys.argv[3]

with open(from_net_path, 'rb') as from_net_file:
    with open(to_net_path, 'rb') as to_net_file:
        from_net = parse_network(from_net_file.read())
        to_net = parse_network(to_net_file.read())
        print('From net:')
        print(str(from_net))
        print('To net:')
        print(str(to_net))
        print('Copyable layers:')
        copyable_layers = get_compatible_layer_pairs(from_net, to_net)
        for i, layers in enumerate(copyable_layers):
            lhs_layer, rhs_layer = layers
            print('{}: {} {}:{} -> {}:{}'.format(i, lhs_layer.name, lhs_layer.offset, lhs_layer.offset + lhs_layer.size, rhs_layer.offset, rhs_layer.offset + rhs_layer.size))

        print('Input a space separated list on numbers corresponding to the list above to copy those layers.')
        print('Input "save" when you want to save the net.')

        resulting_net = to_net.data

        do_break = False
        while not do_break:
            inputs = input()
            for inp in inputs.split(' '):
                if inp == 'save':
                    do_break = True
                    break

                i = int(inp)
                if i < 0 or i >= len(copyable_layers):
                    print("The index must correspond to a copyable layer")

                lhs_layer, rhs_layer = copyable_layers[i]
                resulting_net = copy_layer(resulting_net, lhs_layer, rhs_layer)
                print('Copied {}: {} {}:{} -> {}:{}'.format(i, lhs_layer.name, lhs_layer.offset, lhs_layer.offset + lhs_layer.size, rhs_layer.offset, rhs_layer.offset + rhs_layer.size))

        with open(new_net_path, 'wb') as new_net_file:
            new_net_file.write(resulting_net)
