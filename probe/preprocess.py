import nibabel
import pyvista as pv

# import visualize_3d

epi_img = nibabel.load("../data/00495/BraTS2021_00495_t1.nii.gz")
data = epi_img.get_fdata()

vol = pv.wrap(data)

plotter = pv.Plotter()
plotter.add_volume(vol)
plotter.show()
