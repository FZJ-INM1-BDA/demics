from demics import Tensor, ops


tensor = Tensor.from_file('volumes/img_Slice.*.tif')

mask: Tensor = ops.semantic_segmentation(tensor, 'my_model.pt', tile_size=128, overlap=32)
label: Tensor = ops.label(mask, tile_size=512, overlap=128)
visualization: Tensor = ops.scramble(label, tile_size=512, overlap=128)

visualization.astype('uint8').to_tif('result.tif')
