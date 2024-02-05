import numpy as np
from scipy import io
import util.loader as loader
import util.helper as helper
import util.drawer as drawer
from dataset.batch_data_generator import DataGenerator
import tensorflow as tf
import pickle
import os
import model.model as M
import model.model_cart as MCart

# pickle_file = '/projects/FHEIDE/RADDet/inference/gt/007961.pickle'
pickle_file = '/scratch/gpfs/dd7477/RADDet/000028.pickle'
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        gt = pickle.load(f)

gt_file = pickle_file
gt_instances = loader.readRadarInstances(gt_file)
config = loader.readConfig()
config_data = config["DATA"]
config_model = config["MODEL"]
config_train = config["TRAIN"]
anchor_boxes = loader.readAnchorBoxes() # load anchor boxes with order
anchor_cart = loader.readAnchorBoxes(anchor_boxes_file="./anchors_cartboxes.txt")


### NOTE: using the yolo head shape out from model for data generator ###
model = M.RADDet(config_model, config_data, config_train, anchor_boxes)
model_cart = MCart.RADDetCart(config_model, config_data, config_train, \
                            anchor_cart, list(model.backbone_fmp_shape))


data_generator = DataGenerator(config_data, config_train, config_model, \
                    model.features_shape, anchor_boxes, \
                    anchors_cart=anchor_cart, cart_shape=model_cart.features_shape)
# gt_instances = loader.readRadarInstances(config_data["gt_dir"], sequence_num, \
#                                         config_data["gt_name_format"])
gt_labels, has_label, raw_boxes = data_generator.encodeToLabels(gt_instances)
gt_labels_cart, has_label_cart, raw_boxes_cart = \
                        data_generator.encodeToCartBoxesLabels(gt_instances)

print('done!')
# dataname = '000009'
# mat = io.loadmat('../RadarSimResults/RADs/'+ dataname + '.mat')
# matnp = np.array(mat['RAD'],dtype=np.complex64)
# matnp = np.nan_to_num(matnp)
# matnp = matnp * 1e4
# RAD_file = '/projects/FHEIDE/RADDet/inference/RAD/' + dataname + '.npy'
# np.save(RAD_file, matnp)


# config = loader.readConfig()
# config_radar = config["RADAR_CONFIGURATION"]
# config_data = config["DATA"]
# config_inference = config["INFERENCE"]
# RAD_complex = loader.readRAD(RAD_file)
# ### NOTE: real time visualization ###
# RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
#                 power_order=2), target_axis=-1), scalar=10, log_10=True)
# RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
#                 power_order=2), target_axis=1), scalar=10, log_10=True)
# RA_cart = helper.toCartesianMask(RA, config_radar, \
#                         gapfill_interval_num=int(15))
# RA_img = helper.norm2Image(RA)[..., :3]

# RD_img = helper.norm2Image(RD)[..., :3]
# RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

# RAD_data = helper.complexTo2Channels(RAD_complex)
# RAD_data = (RAD_data - config_data["global_mean_log"]) / \
#                     config_data["global_variance_log"]
# data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)

# img_file = loader.imgfileFromRADfile(RAD_file, config_inference["inference_set_dir"])
# stereo_left_image = loader.readStereoLeft(img_file)

# fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
# axes[0].imshow(RD_img)
# axes[1].imshow(RA_img)
# axes[2].imshow(RA_cart_img)
# axes[2].imshow(RA_cart_img)


