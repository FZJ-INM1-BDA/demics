from demics import Tensor, ops


tensor = Tensor.from_file('volumes/img_Slice.*.tif')
tensor.min_reduce(axis=2)

mask: Tensor = ops.adaptive_threshold(tensor, constant=32., tile_size=512, overlap=32)
label: Tensor = ops.label(mask, tile_size=512, overlap=128)
visualization: Tensor = ops.scramble(label, tile_size=512, overlap=128)

visualization.astype('uint8').to_tif('result.tif')
