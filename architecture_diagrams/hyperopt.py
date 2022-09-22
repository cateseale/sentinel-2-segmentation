import sys

sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # input
    to_input('../examples/fcn8s/coast.png'),

    # Block 1
    to_Conv_nolabel(name='conv1', s_filer=256, n_filer=(16, 16), offset="(0,0,0)", to="(0,0,0)", width=2, height=40,
                    depth=40),
    to_Conv(name='conv2', s_filer=256, n_filer=(16, 16), offset="(0,0,0)", to="(conv1-east)", width=2, height=40,
            depth=40),
    to_Pool(name="pool_1", offset="(0,0,0)", to="(conv2-east)", width=1, height=32, depth=32, opacity=0.5),

    to_Dropout(name='drop1', offset="(0,0,0)", to="(pool_1-east)", height=32, depth=32),

    # Block 2
    to_Conv_nolabel(name='conv3', s_filer=128, n_filer=(32, 32), offset="(1,0,0)", to="(drop1-east)", width=3.5,
                    height=32, depth=32),
    to_Conv(name='conv4', s_filer=128, n_filer=(32, 32), offset="(0,0,0)", to="(conv3-east)", width=3.5, height=32,
            depth=32),
    to_Pool(name='pool_2', offset="(0,0,0)", to="(conv4-east)", width=1, height=25, depth=25, opacity=0.5),

    to_Dropout(name='drop2', offset="(0,0,0)", to="(pool_2-east)", height=25, depth=25),

    to_connection(of='drop1', to='conv3'),

    # Block 3
    to_Conv_nolabel(name='conv5', s_filer=64, n_filer=(64, 64), offset="(1,0,0)", to="(drop2-east)", width=4.5,
                    height=25, depth=25),
    to_Conv(name='conv6', s_filer=64, n_filer=(64, 64), offset="(0,0,0)", to="(conv5-east)", width=4.5, height=25,
            depth=25),
    to_Pool(name='pool_3', offset="(0,0,0)", to="(conv6-east)", width=1, height=16, depth=16, opacity=0.5),
    to_Dropout(name='drop3', offset="(0,0,0)", to="(pool_3-east)", height=16, depth=16),

    to_connection(of='drop2', to='conv5'),

    # Block 4
    to_Conv_nolabel(name='conv7', s_filer=32, n_filer=(128, 128), offset="(1,0,0)", to="(drop3-east)", width=5.5,
                    height=16, depth=16),
    to_Conv(name='conv8', s_filer=32, n_filer=(128, 128), offset="(0,0,0)", to="(conv7-east)", width=5.5, height=16,
            depth=16),
    to_Pool(name='pool_4', offset="(0,0,0)", to="(conv8-east)", width=1, height=8, depth=8, opacity=0.5),
    to_Dropout(name='drop4', offset="(0,0,0)", to="(pool_4-east)", height=8, depth=8),
    to_connection(of='drop3', to='conv7'),

    # Bottleneck
    # Block 5
    to_Conv_nolabel(name='conv9', s_filer=16, n_filer=(256, 256), offset="(2,0,0)", to="(pool_4-east)", width=5.5,
                    height=8, depth=8),
    to_Conv(name='conv10', s_filer=16, n_filer=(256, 256), offset="(0,0,0)", to="(conv9-east)", width=5.5, height=8,
            depth=8),

    to_connection("pool_4", "conv9"),

    # Decoder
    # Block 6
    to_ConvRes(name='up1', offset="(2.1,0,0)", to="(conv10-east)", n_filer=128, s_filer=32, width=5.5, height=16,
               depth=16, opacity=0.5),
    to_Conv_nolabel(name='conv11', s_filer=32, n_filer=(128, 128), offset="(0,0,0)", to="(up1-east)", width=5.5,
                    height=16, depth=16),
    to_Conv(name='conv12', s_filer=32, n_filer=(128, 128), offset="(0,0,0)", to="(conv11-east)", width=5.5, height=16,
            depth=16),

    to_connection("conv10", "up1"),

    # Block 7
    to_ConvRes(name='up2', offset="(2.1,0,0)", to="(conv12-east)", n_filer=64, s_filer=64, width=4.5, height=25, depth=25,
               opacity=0.5),

    to_Conv_nolabel(name='conv13', s_filer=64, n_filer=(64, 64), offset="(0,0,0)", to="(up2-east)", width=4.5,
                    height=25, depth=25),
    to_Conv(name='conv14', s_filer=64, n_filer=(64, 64), offset="(0,0,0)", to="(conv13-east)", width=4.5, height=25,
            depth=25),

    to_connection("conv12", "up2"),

    # Block 8

    to_ConvRes(name='up3', offset="(2.1,0,0)", to="(conv14-east)", n_filer=32, s_filer=128, width=3.5, height=32,
               depth=32, opacity=0.5),
    to_Conv_nolabel(name='conv15', s_filer=128, n_filer=(32, 32), offset="(0,0,0)", to="(up3-east)", width=3.5,
                    height=32, depth=32),
    to_Conv(name='conv16', s_filer=128, n_filer=(32, 32), offset="(0,0,0)", to="(conv15-east)", width=3.5, height=32,
            depth=32),

    to_connection("conv14", "up3"),
    # Block 9
    to_ConvRes(name='up4', offset="(2.1,0,0)", to="(conv16-east)", n_filer=16, s_filer=256, width=2, height=40,
               depth=40, opacity=0.5),
    to_Conv_nolabel(name='conv17', s_filer=256, n_filer=(16, 16), offset="(0,0,0)", to="(up4-east)", width=2, height=40,
                    depth=40),
    to_Conv(name='conv18', s_filer=256, n_filer=(16, 16), offset="(0,0,0)", to="(conv17-east)", width=2, height=40,
            depth=40),

    to_connection("conv16", "up4"),

    to_skip(of='conv8', to='up1', pos=1.25),
    to_skip(of='conv6', to='up2', pos=1.25),
    to_skip(of='conv4', to='up3', pos=1.25),
    to_skip(of='conv2', to='up4', pos=1.25),

    to_ConvSoftMax(name="soft1", s_filer=256, offset="(0.75,0,0)", to="(conv18-east)", width=1, height=40, depth=40,
                   caption="SOFTMAX"),
    to_connection("conv18", "soft1"),
    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()

